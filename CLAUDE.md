# Claude Context — RoboLab + RoboLab Generalization

This repo is a fork of [NVlabs/RoboLab](https://github.com/NVlabs/RoboLab), a task-based evaluation benchmark for manipulation policies built on Isaac Lab. The active work in this fork is **RoboLab Generalization** (working name) — a framework for spec-driven train/test variation, synthetic data generation, and scaling-law experiments.

## Active design docs

Read these first when working on the generalization framework:

- [docs/variation_architecture.md](docs/variation_architecture.md) — RoboLab Generalization architecture: axis taxonomy, role-based object pools, superset-scene pattern, `VariationSpec` API, train/test split convention.
- [docs/data_scaling_laws_paper_summary.md](docs/data_scaling_laws_paper_summary.md) — Lin et al. (2410.18647) summary and the gap analysis for synthetic replication.

## Repo layout

| Path | Purpose |
|---|---|
| `robolab/` | Core package (tasks, registrations, eval, world state, recorder) |
| `robolab/tasks/benchmark/` | The 120 benchmark task definitions |
| `robolab/tasks/randomize_initial_pose/` | Existing per-task pose-variation pattern (subset of proposed `PoseSpec`) |
| `robolab/registrations/droid_jointpos/` | Built-in DROID + joint-position env registrations, including `*_bg_variations.py` and `*_lighting_variations.py` |
| `assets/objects/` | USD object library (ycb, hope, hot3d, handal, vomp, fruits_veggies, objaverse, …) |
| `assets/scenes/` | USD scenes (tabletop layouts) |
| `assets/backgrounds/` | HDR/EXR dome lights |
| `examples/policy/` | Eval entry points: `run_eval.py`, `run_eval_table_variation.py`, `run_eval_background_variation.py`, `run_eval_lighting.py`, `run_eval_camera_pose_variation.py` |
| `examples/teleop/` | Teleop scripts; `teleop_sim.py` and `remote_agent.py` are in flight |
| `skills/` | Claude Code skills: `robolab-scenegen`, `robolab-taskgen` |
| `scripts/` | Validation utilities (`check_tasks_valid.py`, `check_registered_envs.py`) |
| `docs/` | User-facing docs (env registration, tasks, scenes, objects, lighting, etc.) |

## Working conventions

- **Don't add backwards-compat shims.** This is research code; rename and move freely.
- **Don't author new variation systems alongside the existing ones.** If you touch variations, route through the design in `docs/variation_architecture.md`. Do not extend the per-file `randomize_initial_pose/` pattern — that's the combinatorial blow-up the new design replaces.
- **Tabletop scope.** RoboLab Generalization is intentionally limited to tabletop manipulation. Walls/fixtures are visual context only; tables are the only physical interaction surface. Anything beyond that needs an architecture extension before code.
- **Tables are full-USD assets** — mesh + authored material in one file. Do *not* treat mesh and material as independent variation axes (UVs are mesh-specific).
- **Asset splits live in Python**, not in JSON manifests or `train/`/`test/` folders. The `VariationSpec` module is the single source of truth.
- **Variations are logged into `extra_fields`** of `summarize_run` so each episode's result row carries what was sampled.

## Key concepts (RoboLab Generalization)

- **Three variation mechanisms**, distinguished by whether the change loads new prims:
  - *Events* — randomize values on existing prims (built-in `RandomizeInitPoseUniform`, `RandomizeCameraPoseUniform`)
  - *Stage mutation* — edit USD on existing prims (material rebind, light attrs, dome texture)
  - *Registration* — anything that changes which prims are loaded (only used for the superset scene at registration time)
- **Superset scene:** one registered env per task containing the union of every USD that any spec might activate. Inactive prims are made invisible + parked + physics-disabled at reset.
- **Role-based object pools:** `manipulated`, `receptacle`, `fixture`, `distractor`. Each role has its own pool, sampling strategy, and `PoseSpec`.
- **Train/test split:** two `VariationSpec` instances referencing disjoint Python lists/ranges. The superset scene includes the union; the runtime dispatcher activates only what the active spec selects.

## Running things

- Most scripts must be run inside the project's Docker image (Isaac Lab dependency). See `docker/` and the top of any `examples/policy/run_eval*.py` for the launch pattern.
- Quick sanity check on registered envs: `python scripts/check_registered_envs.py --registration <path>`.
- Quick sanity check on a task: `python examples/demo/run_empty.py --task <TaskClassName>`.

## Ongoing work / open threads

- `examples/teleop/teleop_sim.py`, `remote_agent.py`, `scripted_banana_agent.py` — teleop pipeline. Not yet wired to consume a `VariationSpec` for variation-aware demo collection.
- `robolab/tasks/variations-bench/` — scratch directory for the first generalization-bench task (e.g. spoon-in-container or marker-in-mug, intra-class manipulated + receptacle pools, large distractor set).
- The `VariationSpec` package itself does not yet exist — the design in `docs/variation_architecture.md` precedes the implementation.

## Pointers

- Existing user docs: [docs/environment_registration.md](docs/environment_registration.md), [docs/environment_run.md](docs/environment_run.md), [docs/objects.md](docs/objects.md), [docs/scene.md](docs/scene.md), [docs/background.md](docs/background.md), [docs/lighting.md](docs/lighting.md), [docs/camera.md](docs/camera.md).
- Skills: `/robolab-scenegen` (scene `.usda` from natural language), `/robolab-taskgen` (task `.py` from natural language).
