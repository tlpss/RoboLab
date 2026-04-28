# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Server-side teleoperation script — runs on the GPU server.

Pairs with teleop_operator.py, which runs on the operator's laptop.
Receives SpaceMouse deltas over ZMQ, runs IK, steps the simulation,
and streams JPEG images back to the operator.

Usage:
    python examples/demo/teleop_sim.py --task BananaInBowlTask
    python examples/demo/teleop_sim.py --task BananaInBowlTask --port 5555 --headless
"""

import argparse
import os

import cv2

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RoboLab teleoperation server.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--port", type=int, default=5555, help="ZMQ port to listen on.")
parser.add_argument("--num-steps", type=int, default=1000, help="Max steps per episode.")
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--jpeg-quality", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import msgpack  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import apply_delta_pose, subtract_frame_transforms  # noqa: E402

import robolab.constants  # noqa: E402

# Record image observations into the HDF5 dataset (off by default in RoboLab).
# Must be set BEFORE create_env so the recorder cfg is built with the image term.
robolab.constants.RECORD_IMAGE_DATA = True

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import (  # noqa: E402
    auto_register_droid_envs_teleop,
)

auto_register_droid_envs_teleop(task=args_cli.task)

DEVICE = "cuda:0"
EE_BODY = "base_link"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def arm_joint_ids(robot):
    return [i for i, n in enumerate(robot.data.joint_names) if n in ARM_JOINTS]


def ee_pose_in_base(robot):
    idx = robot.data.body_names.index(EE_BODY)
    return subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx],
    )


def ee_jacobian(robot, jids):
    idx = robot.data.body_names.index(EE_BODY)
    jacs = robot.root_physx_view.get_jacobians()
    return jacs[:, idx - 1, :, :][:, :, jids]


def _get_image(obs):
    try:
        img = unpack_image_obs(obs).get("combined_image")
        if img is not None and img.max() == 0:
            print("[warn] image is all zeros — did you pass --headless?")
        return img
    except Exception as e:
        print(f"[warn] could not unpack image: {e}. obs keys: {list(obs.keys())}")
        return None


def encode_image(rgb_array, quality):
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def main():
    output_dir = args_cli.output_dir or os.path.join("output", "teleop", args_cli.task)
    os.makedirs(output_dir, exist_ok=True)
    env, env_cfg = create_env(scene=args_cli.task + "Teleop", device=DEVICE, num_envs=1)

    # Route the recorder's HDF5 output to a task-specific file under output_dir.
    rec = env.recorder_manager
    rec.cfg.dataset_export_dir_path = output_dir
    rec.set_hdf5_file(f"{args_cli.task}_demos.hdf5")
    # Disable streaming auto-flush: episodes are short and we want `discard`
    # to drop buffers cleanly without leaving partial demos in HDF5.
    rec.set_flush_interval(0)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        num_envs=1,
        device=DEVICE,
    )

    robot = env.scene["robot"]
    jids = arm_joint_ids(robot)

    # ZMQ server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://0.0.0.0:{args_cli.port}")
    print(f"\nListening on port {args_cli.port} — start teleop_operator.py on your laptop.")

    saved = 0
    episode = 0
    step_count = 0
    target_pos = target_quat = None

    def reset_episode():
        nonlocal target_pos, target_quat, step_count
        # Clear RobolabEnv's frozen-on-terminate eval state so _reset_idx takes
        # the normal-reset branch and reset-mode events (e.g. randomize_init_pose)
        # actually fire on every operator-triggered reset.
        env.reset_eval_state()
        env.reset()
        obs, _ = env.reset() # second reset - updated camera buffer.
        ik.reset()
        robot.update(env_cfg.sim.dt)
        target_pos, target_quat = ee_pose_in_base(robot)
        target_pos = target_pos.clone()
        target_quat = target_quat.clone()
        step_count = 0
        return obs

    obs = reset_episode()

    while simulation_app.is_running():
        # Receive request from operator
        raw = sock.recv()
        try:
            msg = msgpack.unpackb(raw, raw=False)
            print(f"message received: {msg}")
            cmd = msg.get("cmd", "step")

            if cmd == "quit":
                sock.send(msgpack.packb({"done": True, "saved": saved}))
                break

            if cmd in ("reset", "save", "discard"):
                if cmd == "save" and step_count > 0:
                    rec.set_episode_index(saved)
                    rec.export_episodes(env_ids=[0])
                    print(f"[ep {episode}] saved demo_{saved} ({step_count} steps) → {output_dir}/{args_cli.task}_demos.hdf5")
                    saved += 1
                else:
                    rec.clear(env_ids=[0])
                    print(f"[ep {episode}] {cmd}.")
                obs = reset_episode()
                episode += 1
                img = _get_image(obs)
                image_bytes = encode_image(img, args_cli.jpeg_quality) if img is not None else b""
                sock.send(msgpack.packb({"image": image_bytes, "done": False, "saved": saved, "step": 0}))
                continue

            # cmd == "step"
            delta = torch.tensor(msg["delta"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            gripper_open = bool(msg["gripper_open"])
            target_pos, target_quat = apply_delta_pose(target_pos, target_quat, delta)
            ik.set_command(torch.cat([target_pos, target_quat], dim=-1))

            cur_pos, cur_quat = ee_pose_in_base(robot)
            joint_cur = robot.data.joint_pos[:, jids]
            joint_target = ik.compute(cur_pos, cur_quat, ee_jacobian(robot, jids), joint_cur)

            gripper_val = torch.tensor([[0.0 if gripper_open else 1.0]], device=DEVICE)
            action = torch.cat([joint_target, gripper_val], dim=-1)

            # Print whenever SpaceMouse is actually pushed
            delta_mag = float(delta[0, :3].norm())
            if delta_mag > 0.01:
                joint_change = float((joint_target - joint_cur).abs().max())
                pos_error = float((target_pos - cur_pos).norm())
                print(f"[diag] delta_mag={delta_mag:.4f}  pos_error={pos_error:.4f}  "
                      f"max_joint_change={joint_change:.6f}")

            obs, _, term, trunc, _ = env.step(action)
            step_count += 1
            done = bool((term | trunc).any()) or step_count >= args_cli.num_steps

            img = _get_image(obs)
            image_bytes = encode_image(img, args_cli.jpeg_quality) if img is not None else b""

            # On termination: do NOT auto-save or auto-reset. Report done=True
            # and wait for the operator to send "save", "discard", or "reset".
            if done:
                print(f"[ep {episode}] env terminated at step {step_count} — awaiting save/discard/reset.")

            sock.send(msgpack.packb({
                "image": image_bytes,
                "done": done,
                "saved": saved,
                "step": step_count,
            }))

        except Exception as e:
            import traceback
            print(f"[error] {e}")
            traceback.print_exc()
            # Always reply so the client doesn't hang
            sock.send(msgpack.packb({"error": str(e), "done": False, "saved": saved, "step": step_count}))

    sock.close()
    ctx.term()
    try:
        env.close()
    except Exception:
        pass
    try:
        rec._dataset_file_handler.close() if rec._dataset_file_handler is not None else None
    except Exception:
        pass
    print(f"\nDone. {saved} demonstration(s) saved to {output_dir}/{args_cli.task}_demos.hdf5")


if __name__ == "__main__":
    main()
