# RoboLab Generalization — Architecture

Design notes for **RoboLab Generalization** (working name; package: `robolab_generalization`), a framework on top of RoboLab for specifying training and test variation distributions, generating superset scenes, and running synthetic generalization / scaling-law experiments.

This document covers: how variations are introduced today, where the seams in the codebase are, and the proposed spec-driven design for RoboLab Generalization.

## 1. Background: How variations work today

RoboLab supports two distinct mechanisms for varying environments. Understanding which axis belongs to which mechanism is the core architectural decision.

### 1.1 Registration-time variations

Variations are baked into separately registered Gym environments. Each variant is a distinct env name, queryable via `get_envs(task=...)` or `get_envs(tag=...)`.

- **Entry point:** `auto_discover_and_create_cfgs(...)` inside a `register_envs()` function — see [environment_registration.md](environment_registration.md).
- **Per-variant kwargs:** `background_cfg`, `lighting_cfg`, `camera_cfg`, `robot_cfg`, `actions_cfg`, `observations_cfg`, plus `add_tags` / `env_postfix` to distinguish variants.
- **Naming:** `env_name = env_prefix + TaskClassName + env_postfix`. One Task class can produce many Env names.
- **Examples in repo:**
  - `robolab/registrations/droid_jointpos/auto_env_registrations.py` — base registration
  - `robolab/registrations/droid_jointpos/auto_env_registrations_bg_variations.py` — backgrounds enumerated as variants
  - `robolab/registrations/droid_jointpos/auto_env_registrations_lighting_variations.py` — lighting enumerated as variants

### 1.2 Runtime variations

Applied **after** `create_env`, without re-registration. Two flavors:

- **Events** (`env_cfg.events` or `events=` kwarg) — randomize parameters of *existing* prims:
  - `RandomizeInitPoseUniform.from_params(objects=[...], pose_range={...})`
  - `RandomizeCameraPoseUniform.from_params(cameras=[...], pose_range={...})`
- **Stage mutation** — direct USD edits via `pxr` (see `examples/policy/run_eval_table_variation.py`):
  - Rebind materials on existing meshes (`UsdShade.MaterialBindingAPI.Bind`)
  - Toggle visibility (`UsdGeom.Imageable.MakeInvisible/MakeVisible`)
  - Edit attributes on existing lights (intensity, color, dome `texture_file`)

### 1.3 Existing pose-variation pattern

Pose randomization is already used in the codebase via a per-task event class. Example: [`robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_30cm.py`](../robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_30cm.py).

```python
@configclass
class RandomizeInitPoseUniform:
    randomize_init_pose = EventTerm(
        func=reset_pose_uniform,
        mode="reset",                                  # fires every env.reset()
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": ["banana", "bowl"],          # objects to perturb
            "reset_to_default_otherwise": True,        # everything else snaps to scene default
            "use_collision_check": True,               # rejection sampling on overlap
        },
    )

@dataclass
class BananaInBowlUniformInitPose30cmTask(Task):
    ...
    events = RandomizeInitPoseUniform
```

Properties of this pattern:

- Distribution is **hardcoded into the task file**; the magnitude (`30cm`) is in the filename.
- All listed assets share a **single pose range** — no per-role distribution.
- `mode="reset"` ties sampling to `env.reset()`, so each episode is a fresh draw.
- Collision-checked rejection sampling (`use_collision_check=True`) is already implemented — this validates the rejection-sampling approach proposed for `PoseSpec`.
- Each (task × distribution) is its own Task class / file (folder: `randomize_initial_pose/`), which is exactly the combinatorial-blow-up pattern RoboLab Generalization is designed to avoid.

This is a **strict subset of the `PoseSpec.anchor + jitter` mode in RoboLab Generalization**: anchor = scene-default pose, jitter = the `pose_range` dict, applied to a flat asset list with no role distinction. The migration path is to lift this distribution out of the Task class and into the per-role `PoseSpec`, so one Task can serve many distributions without spawning new files.

### 1.4 Built-in robustness eval scripts

| Script | Mechanism |
|---|---|
| `run_eval_lighting.py` | Registration variants, tag-queried |
| `run_eval_background_variation.py` | Registration variants, tag `background_variations` |
| `run_eval_camera_pose_variation.py` | Runtime events |
| `run_eval_table_variation.py` | Runtime stage mutation (material rebind) |

## 2. Axis taxonomy

Each variation axis maps to exactly one mechanism, determined by whether the change requires loading new USD prims:

| Axis | Mechanism | Reason |
|---|---|---|
| Object shape / mesh | Registration *or* superset-scene | Different USD assets must be present in the stage |
| # objects | Superset-scene + visibility toggle | Can be runtime if all candidates are pre-loaded |
| Distractor set | Superset-scene + visibility toggle | Same as above |
| Object material / appearance | Stage mutation | Material rebind on existing mesh |
| Table (mesh + material as one asset) | Superset-scene + visibility toggle | Pool of complete table USDs (each with authored material); one active per episode |
| Lighting intensity / color / pose | Stage mutation or events | Edit attrs on existing light prim |
| Lighting *type* (sphere → directional) | Registration | Different prim type |
| Background HDR | Stage mutation | Swap `texture_file` on dome light |
| Camera pose jitter | Events | Built-in `RandomizeCameraPoseUniform` |
| Object initial pose | Events | Built-in `RandomizeInitPoseUniform` |
| Robot / action space | Registration | Structural |

**Rule of thumb:**
- *Events* — randomize values on prims that already exist.
- *Stage mutation* — edit USD on prims that already exist.
- *Registration* — anything that changes which prims/assets are loaded.

## 3. Superset-scene pattern

To collapse "# objects" and "distractor set" from registration axes to runtime axes, register **one env per task** whose scene contains the union of every USD across every role-pool (manipulated, receptacle, fixture, distractor — both train and test). At episode reset, show + place the active subset for each role; hide + park + disable-physics on the rest.

### 3.1 Mechanics of "hiding" an object

Visibility alone is insufficient — the rigid body still collides. The full recipe:

1. `UsdGeom.Imageable(prim).MakeInvisible()` — render hidden
2. Teleport pose to an off-stage location (e.g., 100 m below ground)
3. Set kinematic / disable rigid-body dynamics so it doesn't fall forever
4. Optionally disable collision API to skip contact-sensor work

### 3.2 Implications for `contact_object_list`

Contact sensors are auto-generated at registration time from `contact_object_list` (see [environment_generation.md](environment_generation.md)). For the superset pattern, every potentially-active object must be in `contact_object_list` up front. Sensor count grows quadratically with the superset size — measure VRAM and step time at the target scale.

### 3.3 Trade-offs

- ✔ All object/distractor variations become runtime; no per-combo registration.
- ✔ Train/test asset pools are enforced by which subset is *activated*, not which env is loaded.
- ✘ Cannot introduce *novel* meshes outside the registered superset. Acceptable when train/test pools are known up front.
- ✘ VRAM + physics cost scales with superset size, even for hidden objects.

### 3.4 Scope: tabletop manipulation only

RoboLab Generalization is scoped to tabletop manipulation. The robot interacts physically with exactly one surface — the table — plus the manipulated/receptacle/distractor objects. Walls, fixtures, and room geometry are visual context only (background HDR), not interactive.

Within this scope, an "environment" decomposes into three orthogonal axes:

| Axis | Mechanism |
|---|---|
| Table (mesh + authored material as one USD) | Superset-scene + visibility (one active table per episode) |
| Background HDR | Stage mutation on dome light |
| Lighting | Stage mutation / events |

This is sufficient for the Lin et al. (2410.18647) "data scaling laws" research style: their notion of "different kitchens" reduces to (different table + different background + different lighting) for tabletop tasks.

**Out of scope** (would require additional architecture not described here):
- Tasks needing functional environment structure (sinks, drawers, articulated fixtures)
- Tasks where the robot interacts with multiple support surfaces in one episode
- Mobile-base navigation between surfaces

### 3.5 Table pool mechanics

Tables are handled with the same superset-and-toggle pattern as objects:

- **Each table USD is a complete asset** — mesh + authored material in one file. Mesh and material are *not* independent axes. UV layouts are mesh-specific, so binding a material designed for one mesh to another tends to smear, mis-tile, or seam visibly. Visual diversity comes from authoring multiple table USDs.
- **All candidate tables co-exist in the registered scene**, parked off-stage (or stacked at the same nominal pose, only one made visible at a time).
- **The *active* table determines the work-surface anchor** for object pose distributions. Each table USD carries a small metadata block: `work_surface_anchor: (x, y, z, yaw)` and optional `work_surface_extent: BBox`. Object pose anchors in `PoseSpec` are interpreted *relative to* the active table's anchor, so swapping tables of different heights/sizes doesn't invalidate object placements.
- **Friction / restitution** can vary per table USD (carried in the asset, not the spec), or be normalized at asset-prep time so all tables behave identically and only visuals/geometry vary.

The existing runtime material-rebind helper in [`examples/policy/run_eval_table_variation.py`](../examples/policy/run_eval_table_variation.py) is kept for one-off material studies on a fixed table mesh, but it is **not** part of the standard variation flow. The standard flow is: table USD swap, no material rebind.

> *Future extension (not adopted):* allow per-mesh material allowlists, so a table USD could declare which tiling/generic materials are valid for its UV layout and the dispatcher samples (mesh, material) jointly. Worth revisiting only if a use case forces it.

## 4. `VariationSpec` API

A single config object describes the variation axes for a phase (train or test). Train and test specs draw from **disjoint asset pools** at the spec level — the runtime mechanism is identical.

### 4.0 Object pools are nested by role

Tasks distinguish between several categories of objects, and the spec reflects that. Each category has its own pool, its own sampling rule, and (where relevant) its own pose distribution. The categories are not fungible — a task references the *manipulated* object by role, not by index into a flat list.

```python
@dataclass
class ObjectCategoryPool:
    """A named pool of candidate USDs for one role in the task."""
    role: str                         # e.g. "manipulated", "receptacle", "fixture", "distractor"
    pool: list[str]                   # USD asset names available for this role
    active: SamplingStrategy          # how many + which to activate per episode
    pose: PoseSpec                    # spawn pose distribution for the activated members
    materials: AxisSpec | None = None # optional per-role material override

@dataclass
class PoseSpec:
    """Pose distribution for the activated objects in a category.

    Either a fixed anchor pose plus jitter (per-object), or a region from
    which to sample independently per object. Collision/overlap is resolved
    by rejection sampling at reset time.
    """
    anchor: tuple[float, float, float] | None = None
    jitter: PoseRange | None = None        # (x, y, z, yaw) ranges around anchor
    region: BBox | None = None             # alternative: sample uniformly in a bbox
    min_separation: float = 0.0            # reject samples closer than this
```

Typical roles (a task picks the subset it needs):

| Role | Description | Active count |
|---|---|---|
| `manipulated` | The object(s) the policy must act on (Colosseum "MO") | usually 1 |
| `receptacle` | Target container/surface (bowl, rack, plate) | usually 1 |
| `fixture` | Static scene props that matter for the task (e.g. drawer) | task-defined |
| `distractor` | Non-task objects that should not be touched (Colosseum "RO") | 0–N |

```python
@dataclass
class VariationSpec:
    # Structural — defines the registered superset (union of all role pools)
    object_categories: dict[str, ObjectCategoryPool]

    # Visual / parametric axes (runtime stage mutation)
    tables: AxisSpec                        # pool of complete table USDs (mesh + material)
    lighting: AxisSpec                      # intensity / color / pose
    background_hdrs: AxisSpec               # HDR/EXR file pool

    # Camera (runtime events)
    camera_pose_jitter: PoseRange | None

    sampling: Literal["enumerate", "random", "latin_hypercube"]
    seed: int
```

`AxisSpec` and `SamplingStrategy` carry: a value list, a sampling mode (enumerate vs. random vs. fixed), and optional weights.

Object pose lives **inside** each `ObjectCategoryPool` rather than as a global axis, because the manipulated object's pose distribution is generally tighter and task-anchored, while distractor poses cover a broader region. Pinning pose to the role keeps that asymmetry first-class.

### 4.1 Runtime dispatcher

A small set of mutation helpers, keyed by axis name, applied at episode reset:

```python
class VariationApplier:
    def apply(self, env, spec: VariationSpec, episode_idx: int) -> dict:
        sample = self._draw(spec, episode_idx)
        for role, role_sample in sample.categories.items():
            self._apply_visibility(env, role, role_sample.active)
            self._apply_poses(env, role, role_sample.poses)
            self._apply_materials(env, role, role_sample.materials)
        self._apply_table(env, sample.table)
        self._apply_lighting(env, sample.lighting)
        self._apply_background(env, sample.background_hdr)
        return sample  # return what was applied so it can be logged in extra_fields
```

Returned `sample` dict feeds straight into `summarize_run(..., extra_fields=...)` so each episode result carries its variation labels.

### 4.2 Train vs. test

The split is **a property of the experiment**, expressed entirely as Python lists and ranges inside the spec module. There is no `splits.json`, no per-asset `split` field in `object_catalog.json`, and no train/test folders under `assets/`. Different experiments can split the same asset pool differently without touching the asset tree.

Two specs reference shared Python constants:

```python
# experiments/spoon_in_container/splits.py — the split lives here, in code

SPOONS_TRAIN = [
    "assets/objects/handal/spoon.usd",
    "assets/objects/handal/spoon_1.usd",
    "assets/objects/handal/ladle.usd",
    "assets/objects/handal/measuring_spoon.usd",
]
SPOONS_TEST = [
    "assets/objects/handal/spoon_2.usd",
    "assets/objects/handal/pink_spaghetti_spoon.usd",
    "assets/objects/handal/green_serving_spoon.usd",
]

CONTAINERS_TRAIN = [f"assets/objects/vomp/container_a{i:02d}/container_a{i:02d}.usd"
                   for i in range(1, 15)]
CONTAINERS_TEST  = [f"assets/objects/vomp/container_b{i:02d}/container_b{i:02d}.usd"
                   for i in range(1, 8)]

BACKGROUNDS_TRAIN = sorted(glob("assets/backgrounds/indoors/*.hdr"))[:8]
BACKGROUNDS_TEST  = sorted(glob("assets/backgrounds/indoors/*.hdr"))[8:]

LIGHT_INTENSITY_TRAIN = (200.0, 600.0)
LIGHT_INTENSITY_TEST  = (600.0, 1200.0)
```

```python
# experiments/spoon_in_container/specs.py

train_spec = VariationSpec(
    object_categories={
        "manipulated": ObjectCategoryPool(role="manipulated", pool=SPOONS_TRAIN,
            active=SamplingStrategy(count=1, mode="random"),
            pose=PoseSpec(anchor=(0.4, 0.0, 0.8),
                          jitter=PoseRange(x=(-0.05, 0.05), y=(-0.05, 0.05), yaw=(-0.3, 0.3)))),
        "receptacle":  ObjectCategoryPool(role="receptacle", pool=CONTAINERS_TRAIN,
            active=SamplingStrategy(count=1, mode="random"),
            pose=PoseSpec(anchor=(0.6, 0.2, 0.8), jitter=PoseRange(yaw=(-0.2, 0.2)))),
        "distractor":  ObjectCategoryPool(role="distractor", pool=DISTRACTORS_TRAIN,
            active=SamplingStrategy(count_range=(0, 5), mode="random"),
            pose=PoseSpec(region=BBox(x=(0.3, 0.7), y=(-0.3, 0.3), z=(0.8, 0.8)),
                          min_separation=0.08)),
    },
    background_hdrs=AxisSpec(values=BACKGROUNDS_TRAIN, sampling="random"),
    lighting=AxisSpec(intensity_range=LIGHT_INTENSITY_TRAIN, ...),
    sampling="random",
    seed=0,
)

test_spec = VariationSpec(
    object_categories={
        "manipulated": ObjectCategoryPool(role="manipulated", pool=SPOONS_TEST, ...),
        "receptacle":  ObjectCategoryPool(role="receptacle",  pool=CONTAINERS_TEST, ...),
        "distractor":  ObjectCategoryPool(role="distractor",  pool=DISTRACTORS_TEST, ...),
    },
    background_hdrs=AxisSpec(values=BACKGROUNDS_TEST, sampling="random"),
    lighting=AxisSpec(intensity_range=LIGHT_INTENSITY_TEST, ...),
    sampling="enumerate",   # reproducible coverage at eval time
    seed=42,
)
```

For training: sample per episode (cheap, broad coverage). For test: enumerate the cross-product (or a deterministic Latin-hypercube subset) for reproducibility.

#### Parametric axes

For non-USD axes (lighting intensity/color, camera pose jitter, object pose jitter), "disjoint" means **disjoint ranges** rather than enumerated lists. The convention is to give each `AxisSpec` an explicit range and split numerically — e.g. train ⊂ `[200, 600]`, test ⊂ `[600, 1200]`, or interleaved buckets.

#### Superset registration uses the union

The superset scene (§6) is registered once and must include every asset reachable from *either* spec:

```python
all_assets = train_spec.all_assets() | test_spec.all_assets()
build_superset_scene(output_path=..., object_usds=sorted(all_assets), ...)
```

#### Optional disjointness check

A small helper `assert_disjoint(train_spec, test_spec)` that raises if any USD path or range overlaps. Useful as a guardrail, not load-bearing — some experiments deliberately want partial overlap (ID/OOD comparisons).

#### Reproducibility

The split is a Python module under version control. Pin a commit, you've pinned the split. No external manifest can drift out of sync with the spec.

### 4.3 Scale envelope

Target: 2–50 values per axis.

- **Visual axes (50×50×50):** ~125k unique combos. Runtime cost is the per-reset mutation only; no rebuild. Trivial.
- **Structural axes:** absorbed by the superset scene. No registration explosion.
- **Registration count:** one env per task (per phase, if train/test scenes differ structurally).

## 5. Implementation seams

Where RoboLab Generalization plugs into existing RoboLab code:

| Concern | Current code | Change |
|---|---|---|
| Registration | `auto_discover_and_create_cfgs` in `robolab/registrations/...` | Add a "superset" registration helper that takes `⋃ category.pool for category in object_categories` and feeds the union into `contact_object_list`, tagging each prim with its role for the runtime dispatcher |
| Episode reset hook | `env.reset()` in `robolab/eval/episode.py` | Call `VariationApplier.apply(env, spec, episode_idx)` immediately after reset, before policy inference |
| Stage mutation primitives | `change_table_material` in `examples/policy/run_eval_table_variation.py` | Promote to `robolab/variations/runtime.py` as reusable helpers (`set_visibility`, `set_pose`, `set_light_attrs`, `set_dome_texture`, `set_object_material`). Table-material rebind is *not* in the standard set — tables are full-USD swaps. |
| Per-episode logging | `summarize_run(..., extra_fields=...)` | Pass the `VariationApplier` sample dict |
| Spec loading | n/a | New module `robolab/variations/spec.py` with `VariationSpec`, `AxisSpec`, samplers |

## 6. Superset-scene generation from spec

The superset scene is a **derived artifact** of the `VariationSpec`, not a hand-authored file. Adding or removing assets is a one-line edit in the spec; the scene `.usda` is regenerated.

### 6.1 Generator API

```python
def build_superset_scene(
    output_path: str,
    base_scene_path: str,           # empty/table-only scene to start from
    object_usds: list[str],         # absolute paths to the asset USDs
    parking_origin=(0.0, 0.0, -100.0),
    parking_spacing=0.5,
) -> None:
    """Compose a superset scene by referencing each object USD and parking it off-stage."""
```

Implementation sketch (USD scripting via `pxr`):

```python
from pathlib import Path
from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.CreateNew(output_path)
root = stage.DefinePrim("/World", "Xform")

base = stage.DefinePrim("/World/base", "Xform")
base.GetReferences().AddReference(base_scene_path)

for i, usd_path in enumerate(object_usds):
    prim_name = Path(usd_path).stem            # must match contact_object_list entry
    prim = stage.DefinePrim(f"/World/{prim_name}", "Xform")
    prim.GetReferences().AddReference(usd_path)
    x = parking_origin[0] + (i % 10) * parking_spacing
    y = parking_origin[1] + (i // 10) * parking_spacing
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, y, parking_origin[2]))
    UsdGeom.Imageable(prim).MakeInvisible()    # dispatcher makes visible at reset

stage.SetDefaultPrim(root)
stage.GetRootLayer().Save()
```

### 6.2 Spec → scene wiring

The asset list in the spec is the input. The scene file and `contact_object_list` are co-derived:

```python
spec = VariationSpec(
    object_categories={
        "manipulated": ObjectCategoryPool(
            pool=glob("assets/objects/handal/spoon*.usd"), ...),
        "receptacle":  ObjectCategoryPool(
            pool=glob("assets/objects/vomp/container_a*/*.usd"), ...),
        "distractor":  ObjectCategoryPool(
            pool=glob("assets/objects/ycb/*.usd"), ...),
    },
    ...,
)

all_assets = [usd for cat in spec.object_categories.values() for usd in cat.pool]

build_superset_scene(
    output_path="robolab/tasks/variations-bench/scenes/spoon_in_container.usda",
    base_scene_path="assets/scenes/base_empty.usda",
    object_usds=all_assets,
)

contact_object_list = ["table"] + [Path(p).stem for p in all_assets]
```

The Task class then loads the generated scene with `import_scene(...)` unchanged. Adding/removing objects is a spec edit + regenerate; no manual GUI work.

### 6.3 Preconditions on incoming USDs

The generator composes well-formed assets — it does not fix malformed ones. Each USD must already satisfy the requirements in [docs/objects.md](objects.md):

- `defaultPrim` named to match the file
- `RigidBodyAPI`, `MassAPI`, PhysX `convexDecomposition` collision
- Friction 2.0–5.0
- Texture paths relative to the USD

If a user supplies a list of raw OBJ/GLB files, those still need to go through the IsaacSim authoring workflow first ([docs/objects.md](objects.md) §"Creating New Objects").

### 6.4 Per-object parking pose vs. per-instance pose

Parking poses written by the generator are *initial* poses only — the runtime dispatcher overwrites them on `env.reset()` (visibility on, teleport to active spawn pose for activated assets; everything else stays parked). The generator's pose grid is just for scene-load sanity, not for episode behavior.

### 6.5 Settle step

`assets/scenes/_utils/settle_scenes.py` is **not** run on the generated scene — settling parked-off-stage objects gives nonsense rest poses. Settling, if needed, happens at episode reset *after* the dispatcher has placed the active subset.

### 6.6 Implementation seam

| Concern | Current code | Change |
|---|---|---|
| Scene authoring | Manual GUI drag-drop into `.usda` | New utility `robolab/variations/scene_builder.py::build_superset_scene` driven by `VariationSpec` |
| Catalog query | `assets/objects/_utils/generate_catalog.py` (`iter_object_files`, `load_catalog`) | Reuse to resolve "all spoons in handal" → list of USD paths, instead of glob |
| `contact_object_list` derivation | Hand-written in each Task | Auto-derive from `spec.all_assets()` so the Task and scene stay in sync |

## 7. Open questions

- **Per-env variation in `num_envs > 1`.** Each parallel env should sample independently. The mutation helpers must accept an env index (USD prim paths use `{ENV_REGEX_NS}` — variations need to apply per-instance, not globally).
- **Reproducibility.** A spec + episode index must determine the full sample deterministically (for resume + comparison across policies).
- **Physics settling after object placement.** When activating a previously-hidden object, may need a few simulation steps before the policy starts.
- **Sensor cost at supeset size 50+.** Contact sensors are pairwise; benchmark before committing to the largest pool sizes.

## 8. Related docs

- [environment_registration.md](environment_registration.md) — registration mechanics
- [environment_run.md](environment_run.md) — `create_env`, events, robustness scripts
- [environment_generation.md](environment_generation.md) — contact sensor auto-generation
- [background.md](background.md), [lighting.md](lighting.md), [camera.md](camera.md) — built-in variation configs
- [scene.md](scene.md), [objects.md](objects.md) — USD scene/object authoring
