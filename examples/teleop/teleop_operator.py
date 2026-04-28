"""
Operator-side teleoperation script — runs on your laptop.

Reads the SpaceMouse, shows the live camera feed, and sends actions to the
GPU server running teleop_sim.py over ZMQ.

No Isaac Lab or GPU needed on the laptop.

Install deps (once):
    pip install pyzmq hidapi numpy opencv-python scipy msgpack

SpaceMouse udev rule (Linux, one-time, then replug):
    echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"' \\
        | sudo tee /etc/udev/rules.d/99-spacemouse.rules
    sudo udevadm control --reload-rules

Controls:
    SpaceMouse axes        : translate / rotate end effector
    SpaceMouse left button : toggle gripper open/closed
    SpaceMouse right button: reset target pose
    Keyboard:
        r — reset the env (start a fresh episode, discarding the current one)
        d — discard the current episode and reset
        s — save the current episode (typically after success/termination) and reset
        q — quit
    On success/termination the server reports done=True; teleop pauses and
    waits for s (save) or d (discard) before resetting.

Usage:
    python teleop_operator.py --host <server-ip>
    python teleop_operator.py --host 192.168.1.42 --port 5555
"""

import argparse
import threading
import time
from collections import deque

import cv2
import msgpack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from spacemouse import SpaceMouseAgent as SpaceMouse

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SpaceMouse operator client.")
parser.add_argument("--host", type=str, required=True, help="GPU server IP or hostname.")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--translation-scale", type=float, default=0.02)
parser.add_argument("--rotation-scale", type=float, default=0.02)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# ZMQ helpers
# ---------------------------------------------------------------------------


def send(sock, cmd, delta6=None, gripper_open=True):
    payload = {"cmd": cmd}
    if delta6 is not None:
        payload["delta"] = delta6.tolist()
        payload["gripper_open"] = gripper_open
    sock.send(msgpack.packb(payload))
    return msgpack.unpackb(sock.recv(), raw=True)


def decode_image(image_bytes):
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    sm = SpaceMouse(
        deadzone=0.1,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to {args.host}:{args.port}")
    print("Keys: s=save  d=discard  q=quit\n")

    cv2.namedWindow("teleop", cv2.WINDOW_NORMAL)

    last_img = None
    saved = 0
    step = 0
    done = False  # True once the env reports termination — pauses stepping

    running = True
    while running:
        # When the episode is done, stop sending steps and wait for s/d/r/q.
        if not done:
            action = sm.get_action()
            delta6 = np.array(action[:6])
            gripper_open = action[6] > 0  # gripper action positive means open

            reply = send(sock, "step", delta6, gripper_open)
            img = decode_image(reply.get(b"image", b""))
            step = reply.get(b"step", 0)
            saved = reply.get(b"saved", 0)
            done = bool(reply.get(b"done", False))
            if img is not None:
                last_img = img
            if done:
                print(f"[ep done] terminated at step {step}. Press s=save, d=discard, r=reset.")
        else:
            # Episode is paused; keep the UI responsive without sending steps.
            gripper_open = True

        # Render last frame with status overlay
        if last_img is not None:
            disp = last_img.copy()
            status = "DONE — s/d/r" if done else f"step {step}  saved {saved}  gripper {'open' if gripper_open else 'CLOSED'}"
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("teleop", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            send(sock, "quit")
            running = False
        elif key == ord("s"):
            reply = send(sock, "save")
            saved = reply.get(b"saved", saved)
            img = decode_image(reply.get(b"image", b""))
            if img is not None:
                last_img = img
            done = False
            print(f"Saved. Total: {saved}")
        elif key == ord("d"):
            reply = send(sock, "discard")
            img = decode_image(reply.get(b"image", b""))
            if img is not None:
                last_img = img
            done = False
            print("Discarded.")
        elif key == ord("r"):
            reply = send(sock, "reset")
            img = decode_image(reply.get(b"image", b""))
            if img is not None:
                last_img = img
            done = False
            print("Reset.")

        time.sleep(0.01)  # keep UI responsive without busy-spinning

    cv2.destroyAllWindows()
    sm.close()
    sock.close()
    ctx.term()


if __name__ == "__main__":
    # set logging to INFO only
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
