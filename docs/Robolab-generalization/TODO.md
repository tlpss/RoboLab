# TODO — RoboLab Generalization

## Open

- **Teleop pipeline → VariationSpec**: `examples/teleop/teleop_sim.py`, `remote_agent.py`, `scripted_banana_agent.py` are not yet wired to consume a `VariationSpec` for variation-aware demo collection.
- **Scripted banana agent (pre)grasp pose is broken**: `examples/teleop/scripted_banana_agent.py` succeeds intermittently but the (pre)grasp pose isn't reliable. Root cause is a tangle of (a) targeting `base_link` instead of the TCP (see TCP-frame TODO), (b) the banana's rigid-body root being at COM rather than the geometric spine — currently compensated by a hand-tuned `GRASP_LOCAL_XY_OFFSET` (~4 cm, narrow window between working and overshoot), and (c) hover phase ending with the descent target unreachable when offsets are wrong. Cleanup steps once the TCP frame is in: drop the local-xy offset, derive grasp height from the banana's bbox geometry instead of root z, and either drive the agent off bbox centroid or add a proper spine-detection step.
- **First generalization-bench task**: scratch work in `robolab/tasks/variations-bench/` (e.g. spoon-in-container or marker-in-mug, intra-class manipulated + receptacle pools, large distractor set).
- **`VariationSpec` package**: design exists in `docs/variation_architecture.md`; implementation pending.
- **Control the robot in the TCP frame** (data collection + policy inference). Currently every consumer of EE pose — teleop deltas (`remote_agent.py`), scripted IK targets (`scripted_banana_agent.py`), recorded `ee_pose` observations, and policy action/observation spaces — uses the gripper *mounting* frame (`base_link` in `examples/teleop/agent.py:25`), which sits ~10 cm behind the fingertips and forces magic-number offsets everywhere. We want the TCP (point between fingertips) to be the unified control frame end-to-end. Subtasks:
  - Promote the existing `tcp` Xform under `base_link` in `assets/robots/franka_robotiq_2f_85_flattened.usd` to a rigid link (apply `PhysicsRigidBodyAPI` + tiny `PhysicsMassAPI`, no collider) connected to `base_link` via a `PhysicsFixedJoint`, so it appears in `robot.data.body_names`.
  - Switch `EE_BODY` in `agent.py` from `"base_link"` to `"tcp"` and re-tune the agent constants (now no `GRASP_LOCAL_XY_OFFSET` fudge for finger length needed).
  - Audit every other place that reads or commands the EE frame — recorder observation terms, action spaces, evaluation utilities, policy clients (`robolab_policy_client/`), env configs — and switch them to the TCP frame consistently. The dataset format change is breaking; flag old hdf5s as incompatible.
  - Verify on policy inference: the `pi05` and other clients should receive TCP-frame proprio and emit TCP-frame deltas; without this, demos collected in TCP frame won't match what the policy sees at deploy time.

## Done

_(move items here when finished, with a one-line note on what shipped.)_
