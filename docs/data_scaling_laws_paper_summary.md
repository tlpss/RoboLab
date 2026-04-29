# Data Scaling Laws in Imitation Learning for Robotic Manipulation

Summary of Lin et al. (2024), [arxiv.org/abs/2410.18647](https://arxiv.org/abs/2410.18647), [project page](https://data-scaling-laws.github.io/), with notes on what is needed to reproduce the experiments using synthetic data via **RoboLab Generalization** (working name; see [variation_architecture.md](variation_architecture.md)).

## 1. Research question

How does an imitation-learning policy's ability to generalize to **novel environments and novel objects** scale with the diversity and quantity of the training data? Specifically, the paper studies three axes independently:

1. Number of distinct **training environments**
2. Number of distinct **training objects** (within a single semantic category)
3. Number of **demonstrations per environment / per object**

## 2. Key findings

- Generalization to **unseen environments + unseen objects** follows an approximate **power law** in both `#environments` and `#objects`.
- **Diversity beats quantity:** beyond a small per-environment / per-object demo threshold, adding more demos at the same env/object gives diminishing returns. Adding a *new* environment or object is dramatically more valuable than adding more demos at an existing one.
- **Practical recipe:** with **32 distinct environments**, **1 unique object per environment**, **~50 demonstrations each** (~1,600 demos total per task), they achieve **~90% success rate** on entirely novel environments + novel objects — collected by 4 people in one afternoon.
- Aggregate scale across all experiments: **>40,000 demos collected**, **>15,000 real-world rollouts** for evaluation.

## 3. Tasks

Five manipulation tasks (project page):

- Pour Water
- Mouse Arrangement
- Fold Towels
- Unplug Charge
- Shuffle

The "afternoon recipe" headline result is reported on two of these.

## 4. Definition of "environment"

An *environment* is a **distinct physical space** — a real-world location with its own table/work-surface and surroundings. Examples named on the project page:

- Iron Cabinet
- Wooden Podium
- Black Workbench
- Two Chairs
- Cart
- Workstation
- Meeting Room

So one environment ≈ (one specific table or work surface, one specific room/lighting, one specific spatial layout). This is critical for synthetic replication: each "environment" is one discrete tuple, **not** a continuous random sample over visual axes.

## 5. Object definition

Each environment is paired with **one unique manipulation object** (in the recipe condition). For the object-scaling axis, multiple intra-class object instances are used (e.g. many different mice for "Mouse Arrangement"). Test objects are held-out instances **of the same semantic class** as training objects — this is *intra-class* generalization, not category transfer.

## 6. Experimental design

Three independent ablation sweeps. In each, two of the three axes are held fixed and one is varied; success rate is measured on a held-out test set of (novel environment, novel object) pairs.

| Sweep axis | Held fixed | Measured |
|---|---|---|
| `#environments` | demos per env, single object class | Test success on novel env + novel object |
| `#objects` (within class) | demos per object, single environment | Test success on novel env + novel object |
| `#demos per env/object` | small env+object pool | Test success — expected to plateau |

The third sweep is the key result: it shows demos-per-unit saturates, while the first two show power-law scaling.

## 7. Data collection

- Real-world **teleoperation**.
- Each (environment, object) pair receives a fixed number of demos (≈50 in the recipe condition).
- All demos are collected by humans; no synthetic augmentation.

## 8. Train/test protocol

- **Train:** N envs × M objects, all with K demos each.
- **Test:** disjoint envs + disjoint objects (within the same class). Generalization is measured on the *cross product* — novel environment containing a novel object.
- Specific test-set sizes are not enumerated in the abstract / project page snippets I read.

---

# Replicating the experiments synthetically with RoboLab Generalization

RoboLab Generalization (see [`variation_architecture.md`](variation_architecture.md)) supports everything *except* the demo-collection side. Mapping the paper to the framework:

## What maps directly

| Paper concept | RoboLab Generalization spec field |
|---|---|
| One environment | One frozen tuple `(table_usd, background_hdr, lighting_setting)` |
| Number of training environments | Length of an enumerable list of such tuples |
| Object pool (within class) | `ObjectCategoryPool.pool` for the `manipulated` role |
| Disjoint train/test env split | Two `VariationSpec`s with disjoint env-tuple lists |
| Disjoint train/test object split | Two `VariationSpec`s with disjoint `manipulated.pool` lists |
| Per-episode coverage logging | `extra_fields` in `summarize_run` |

## What needs to be added

### 1. `EnvironmentSpec` as a frozen tuple

The current spec samples `tables`, `background_hdrs`, and `lighting` axes independently. Lin et al. requires *frozen* environments — each one a complete (table, background, lighting) tuple counted as one discrete environment. Add:

```python
@dataclass(frozen=True)
class EnvironmentSpec:
    table_usd: str
    background_hdr: str
    lighting: LightingSetting   # concrete values, not a range

@dataclass
class VariationSpec:
    environments: list[EnvironmentSpec]   # length N → "N environments"
    object_categories: dict[str, ObjectCategoryPool]
    ...
```

The dispatcher samples *one* `EnvironmentSpec` per episode (or assigns one per env_id when `num_envs > 1`). For the per-environment condition, also pin which object goes with which environment (paper's recipe pairs them 1:1).

### 2. Synthetic teleop / demo-generation pipeline

The paper collects demos via real-world teleoperation. To replicate synthetically you need *a way to generate trajectories* under the train spec. Three options, in increasing difficulty:

- **Scripted policies** per task (banana-grasp, mug-pour, etc.) — fastest to set up; limited to tasks where a hand-coded controller is feasible. Existing `examples/teleop/scripted_banana_agent.py` is the seed of this.
- **Sim-teleop** with the existing `teleop_sim.py` — a human-in-the-loop generates demos. Slow and expensive at the 1,600+ demo scale; not the right tool for power-law sweeps.
- **Existing policy rollouts** (e.g. a strong base policy) — collect successful rollouts as "demos." Risks self-distillation artifacts.

Whichever path: the demo-generation loop must consume the same `VariationSpec` so demos cover the same (env, object) cells the paper varies.

### 3. Sweep driver

The scaling-law experiment is ~30+ training runs:

- `#envs ∈ {1, 2, 4, 8, 16, 32}`
- `#objects ∈ {1, 2, 4, 8, 16}`
- `#demos ∈ {10, 25, 50, 100, 200}`

A small Python driver constructs `VariationSpec`s by truncating pools and dispatches train+eval jobs. Each job logs `(N_envs, N_objects, N_demos, success_rate)` to a results table; power-law fits are post-hoc.

### 4. Asset pool sizes

To run the full sweep on the largest axis (`32 envs`), you need at least 32 + held-out test environments — roughly 40+ distinct table USDs and 40+ distinct background HDRs. RoboLab today has:

- Tables: 1 base mesh × 4 materials in `run_eval_table_variation.py` → far short of 40
- Backgrounds: 4 built-in HDR configs + arbitrary user-supplied HDRs (e.g. Poly Haven) → easy to scale to 40+
- Objects per class: e.g. 8 spoons in `handal/`, 5 mugs across datasets → large object pools require either more authoring or a synthetic mesh generator

Authoring 40+ table USDs is the bottleneck. Could be addressed by procedural generation (parametric tables) or by relaxing the "per-table authored material" rule and treating table-mesh × table-material as the environment count.

## Concrete experimental plan template

For a single task (e.g. spoon-in-container):

1. **Author 40 table USDs** (or 8 meshes × 5 authored material variants) → split 32 train / 8 test.
2. **Curate 40 HDR backgrounds** from Poly Haven → 32 train / 8 test, paired 1:1 with tables.
3. **Curate 40 spoons** → 32 train / 8 test.
4. **Define lighting**: train range and test range disjoint (or 32 + 8 discrete settings).
5. **Build train `VariationSpec`** with `environments = [(table_i, hdr_i, light_i) for i in train_idx]` and `manipulated.pool = spoons_train`.
6. **Generate ~50 demos per (env, object)** via scripted policy → 1,600 demos.
7. **Train policy** on those demos.
8. **Evaluate** with `test_spec` whose env-tuples and object pool are disjoint.
9. **Sweep** by truncating each list to {1, 2, 4, 8, 16, 32} and re-running.

## Limitations of synthetic replication

- The paper's main contribution is showing scaling **transfers** to real-world. A pure-sim replication validates the *shape* of the power law but cannot validate the real-world claim.
- Sim-to-real gap may compress measured success rates; the absolute "90%" target may not be meaningful synthetically.
- Synthetic data is much cheaper to generate, so the relative cost of "more demos vs. more diversity" is different. The paper's *practical* conclusion ("don't bother with more demos beyond ~50/cell") may be specific to teleop economics.

## Open architectural questions

- Should `EnvironmentSpec` allow correlated lighting (e.g. an HDR ships with a recommended sun direction), or stay independent?
- Pairing convention: paper uses 1 object per env in the recipe — does our spec express "this object lives in this environment" as a constraint, or do we sample independently?
- Demo storage and variation-tagging format — currently HDF5 via the patched recorder manager; needs the sampled `VariationSpec` instance written into each episode's metadata for downstream analysis.
