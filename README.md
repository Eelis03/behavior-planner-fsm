# Behavior Planner Fsm

Finite state machine and cost-based lane change decisions on a highway traffic simulation.

[![CI](https://github.com/Eelis03/behavior-planner-fsm/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/behavior-planner-fsm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

A behaviour planner for a highway vehicle: a five state machine over keeping lane,
preparing a change left or right, and executing a change left or right, driven by a
weighted cost function and constrained by a safety gate that can refuse any manoeuvre the
cost function asks for. It runs against a microscopic traffic simulation in which the
surrounding vehicles follow the Intelligent Driver Model longitudinally and choose their
own lanes with MOBIL, so the gaps the planner reasons about are contested rather than
handed to it. The intended reader is someone building or reviewing a decision layer, who
needs to see where the boundary between ranking manoeuvres and permitting them is drawn,
and what happens at that boundary.

## Problem

A vehicle on a multi-lane road has to decide, repeatedly and quickly, whether to stay in
its lane or move to another one. The decision is not a continuous optimisation. It is a
choice among a small number of discrete manoeuvres, each of which commits the vehicle for
several seconds, and each of which is either legal from the current behaviour state or is
not. Four things are in tension:

1. Progress. A slow vehicle ahead costs speed, and a faster lane recovers it.
2. Safety. A gap that is large enough to fit in is not necessarily large enough to enter.
3. Comfort. Lateral effort has a cost, and so does changing one's mind repeatedly.
4. Lane discipline. Keeping right is a rule of the road, not a preference to be traded
   away whenever the left lane is marginally faster.

Three properties separate a usable behaviour planner from a scoring function.

*The state machine must be total and explicit.* Every combination of current state and
input has to have a defined response. A machine that silently ignores an input it does
not recognise turns a planner bug into a vehicle that inexplicably never changes lane.

*Safety must not be a cost term.* A weighted sum has no veto. If safety is a large
penalty, a large enough speed advantage outweighs it, and the situation in which that
happens is precisely the situation in which it must not. The failure needs no mistuning
to appear: it is a property of addition.

*Surrounding traffic must not be passive.* If the traffic holds its lane and its speed,
every gap the planner sees is still there when it arrives, and the planner is being
tested against a problem that does not exist.

## Approach

The traffic is microscopic. Longitudinal behaviour comes from the Intelligent Driver
Model of Treiber, Hennecke and Helbing, whose acceleration

```
a = a_max * (1 - (v / v0)^delta - (s_star / s)^2),
s_star = s0 + max(0, v * T + v * dv / (2 * sqrt(a_max * b))),
```

is collision free for a leader that does not brake harder than `b`, and whose parameters
are those of the paper's highway calibration. Lane choice for the traffic comes from
MOBIL, by Kesting, Treiber and Helbing, which accepts a change when the new follower is
not forced below `-b_safe` and when

```
a_c' - a_c + p * ((a_n' - a_n) + (a_o' - a_o)) > delta_a_th
```

holds, with `p` the politeness factor. Politeness is what makes the traffic interesting:
at the default of 0.25 a slow leader will sometimes pull aside for a faster follower and
sometimes not, so a gap is a thing that has to be caught rather than a thing that waits.

The ego vehicle is planned in three separated layers.

*The finite state machine* declares its transition table over the full product of five
states and six events. The legal and the rejected pairs are both written out, and an
import time check refuses to load the module unless the two partition the product
exactly. An illegal pair raises rather than returning the current state, so the machine
has no silent no-op.

*The cost function* scores the feasible successors on four terms, each normalised to
`[0, 1]` before weighting: progress, measured as the speed shortfall a lane can deliver;
safety, measured as the shortfall against its own desired gaps; comfort, measured as the
lateral effort of the state; and lane preference, measured as distance from the preferred
lane. A prepare state is scored on the mean of the lane it is aiming at and the lane it
is still in, which is what stops the machine from parking in a prepare state and
collecting the target lane's benefit without paying for the manoeuvre. Every weight and
every scale is a documented field of `CostConfig`.

*The safety gate* is a separate object with its own thresholds and its own constructor.
It returns a verdict, not a number, and a vetoed candidate is removed from the set before
any cost is compared. It refuses a manoeuvre that leaves the road, that moves into a
space a vehicle already occupies, that would leave less than
`minimum_gap + headway * speed` to the leader or the follower in the target lane, that
would leave less than three seconds to collision with either, or that would force the new
follower to brake harder than three metres per second squared. The last of these is
MOBIL's own safety criterion applied to the ego, which holds the ego to the standard the
traffic is held to.

Trajectory generation follows the Frenet frame formulation of Werling and colleagues,
with the lateral transition as the minimum jerk quintic and the longitudinal profile from
forward integrating the car following model. The quintic's peak rate, acceleration and
jerk are available in closed form, so the comfort of a manoeuvre is checked rather than
sampled.

`docs/design-notes.md` records the alternatives that were considered and rejected, the
conditions under which this planner gives poor results, and what a hand-tuned cost
function cannot do however carefully it is tuned.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/behavior-planner-fsm.git
cd behavior-planner-fsm
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from behavior_planner import Road, Scenario, VehicleSpec, run_scenario, scenario_metrics

scenario = Scenario(
    name="overtake",
    description="One slow vehicle ahead in the right lane.",
    road=Road(lane_count=3, length=1200.0),
    ego=VehicleSpec(lane=0, s=0.0, speed=28.0, desired_speed=31.0),
    duration=30.0,
    scripted=(VehicleSpec(lane=0, s=60.0, speed=20.0, desired_speed=20.0, holds_lane=True),),
)

trace = run_scenario(scenario)
metrics = scenario_metrics(trace)
print(metrics.collisions, metrics.lane_changes, round(metrics.mean_speed, 2))
# 0 2 26.13
print([state.value for state in trace.state_sequence])
# ['prepare_lane_change_left', 'lane_change_left', 'keep_lane',
#  'prepare_lane_change_right', 'lane_change_right', 'keep_lane']
```

The ego moves left, passes, and returns right. Substituting `KeepLaneBaseline()` for the
default policy, or a different cost model or safety gate, requires no other change: all
of them are reached through the Protocols in `behavior_planner.algorithm.base`.

Runnable examples live in `examples/`:

```bash
uv run python examples/run_suite.py
uv run python examples/run_scenario.py --scenario gap_wait --transitions-only
uv run python examples/plot_results.py
```

The first produces the tables below. The second prints the decision timeline of one
scenario. The third writes the figures into `outputs/`.

## Results

Produced by `uv run python examples/run_suite.py`, on Python 3.12.10 with numpy 2.5.1 and
matplotlib 3.11.1, on one core of an AMD64 desktop under Windows 11. Every scenario runs
for 60 seconds on a 1200 metre three lane ring at an integration step of 0.1 seconds. The
behaviour layer plans at 2 Hz, the traffic consults MOBIL at 1 Hz, and a lane change takes
3.5 seconds. The whole run takes about 2.4 seconds.

| Scenario         | Collisions | Mean speed (m/s) | Distance (m) | Lane changes | Min headway (s) | Min TTC (s) | TTC p05 (s) | TTC median (s) |
| ---------------- | ---------- | ---------------- | ------------ | ------------ | --------------- | ----------- | ----------- | -------------- |
| free_flow        | 0          | 28.43            | 1707         | 0            | inf             | inf         | inf         | inf            |
| slow_leader      | 0          | 28.34            | 1700         | 2            | 1.96            | 6.88        | 7.36        | 12.24          |
| blocked_overtake | 0          | 20.30            | 1217         | 0            | 1.87            | 6.88        | 10.69       | 211.40         |
| gap_wait         | 0          | 19.67            | 1179         | 1            | 1.82            | 8.12        | 12.86       | 259.42         |
| light_traffic    | 0          | 25.01            | 1501         | 1            | 2.75            | 30.47       | 30.59       | 38.19          |
| dense_traffic    | 0          | 20.33            | 1220         | 1            | 1.87            | 27.31       | 27.76       | 38.35          |
| slow_right_lane  | 0          | 26.27            | 1577         | 3            | 3.18            | 13.09       | 13.29       | 14.84          |

Over the seven scenarios: 0 collisions, 8 lane changes, a mean ego speed of 24.05 m/s, a
smallest time headway of 1.82 s and a smallest time to collision of 6.88 s. The safety
gate raised three distinct veto reasons during the suite: `follower_gap`,
`follower_time_to_collision` and `follower_deceleration`.

Mean ego speed against a control policy that is identical in every respect except that it
never leaves its lane:

| Scenario         | Planner speed (m/s) | Baseline speed (m/s) | Gain (percent) | Lane changes |
| ---------------- | ------------------- | -------------------- | -------------- | ------------ |
| free_flow        | 28.43               | 28.43                | +0.0           | 0            |
| slow_leader      | 28.34               | 20.30                | +39.6          | 2            |
| blocked_overtake | 20.30               | 20.30                | +0.0           | 0            |
| gap_wait         | 19.67               | 18.54                | +6.1           | 1            |
| light_traffic    | 25.01               | 24.25                | +3.1           | 1            |
| dense_traffic    | 20.33               | 20.03                | +1.5           | 1            |
| slow_right_lane  | 26.27               | 18.39                | +42.9          | 3            |

What the numbers say:

- Collisions. Zero, in every scenario, under both policies. This is the only number in
  the table with a required value, and the others are only meaningful because it holds.
- The benefit is where the opportunity is. The two scenarios with a clear lane beside a
  slow obstruction gain 39.6 and 42.9 percent, which is most of the way to the free flow
  speed of 28.43 m/s. The two scenarios with no usable gap gain nothing: `free_flow` has
  nothing to overtake and `blocked_overtake` has nowhere to go. A planner that gained
  speed in `blocked_overtake` would be taking gaps it should not have.
- Waiting has a price and pays it. In `gap_wait` the ego sits behind a slow leader while
  faster traffic streams past on its left. It prepares a change four times and the gate
  refuses each one on `follower_gap`, at 6.5, 15.0, 24.5 and 34.0 seconds, before a
  usable gap arrives at 41.0 seconds. The 6.1 percent gain is the value of one lane
  change taken 19 seconds later than the cost function first wanted it.
- Density removes the opportunity, not the discipline. `dense_traffic`, at sixteen
  vehicles per lane, yields 1.5 percent from a single lane change, and `light_traffic`,
  at eight, yields 3.1 percent from one as well. The planner does not manufacture gains
  by changing lane more often when there is nowhere better to be.
- Headway. The smallest time headway anywhere in the suite is 1.82 s, above the 1.6 s
  time gap the traffic itself targets. The fifth percentile of the time to collision
  never falls below 7.36 s. The median varies widely between scenarios because it is a
  median over the steps at which the ego was closing on anything at all, and in the
  scripted scenarios most of those steps are a slow approach from far away.

`outputs/slow_leader_timeline.png` plots the speed, the lateral offset and the behaviour
state of one run against time, which is where the two prepare states and the two change
states are visible as a sequence. `outputs/time_to_collision.png` shows the cumulative
distribution of the times to collision per scenario, and `outputs/suite_speed.png` shows
the two speed columns above as grouped bars.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/behavior_planner/model/road.py` | `Road`: ring geometry, lane centres, arc length wrapping, signed separation, lane occupancy. |
| `src/behavior_planner/model/vehicle.py` | `Vehicle`, `VehicleShape`, `LaneChange`: pose, body, and an in-progress lateral manoeuvre. |
| `src/behavior_planner/model/lateral.py` | `LateralProfile`: the minimum jerk quintic and its closed-form peak rate, acceleration and jerk. |
| `src/behavior_planner/model/states.py` | `BehaviorState` and `BehaviorEvent`: the two alphabets of the machine. |
| `src/behavior_planner/model/config.py` | Every tunable number: IDM, MOBIL, cost weights, cost scales, safety limits, planner timings. |
| `src/behavior_planner/model/traffic.py` | `TrafficSnapshot`: per-lane neighbour queries, overlap detection, headway and time to collision. |
| `src/behavior_planner/model/decision.py` | `DecisionContext`, `CostTerms`, `SafetyVerdict`, `CandidateScore`, `Decision`. |
| `src/behavior_planner/algorithm/base.py` | The five Protocols the layers are written against. |
| `src/behavior_planner/algorithm/idm.py` | The Intelligent Driver Model, including the closed-form equilibrium gap. |
| `src/behavior_planner/algorithm/mobil.py` | MOBIL: safety criterion, incentive criterion, politeness, keep-right bias. |
| `src/behavior_planner/algorithm/fsm.py` | The transition table, its totality check, and the successor and event queries. |
| `src/behavior_planner/algorithm/cost.py` | `WeightedCostModel`: the four normalised terms. |
| `src/behavior_planner/algorithm/safety.py` | `GapAndDecelerationGate`: the veto, with its own thresholds and no access to any cost. |
| `src/behavior_planner/algorithm/trajectory.py` | Binding lanes, the acceleration command, and the sampled trajectory for a manoeuvre. |
| `src/behavior_planner/algorithm/planner.py` | `FiniteStateBehaviorPlanner` and the `KeepLaneBaseline` control. |
| `src/behavior_planner/pipeline/scenarios.py` | Declarative scenarios, the seeded random fill, and the standard suite. |
| `src/behavior_planner/pipeline/simulator.py` | The synchronous update loop and the collision check. |
| `src/behavior_planner/pipeline/trace.py` | `StepRecord` and `RunTrace`, the structured record everything downstream reads. |
| `src/behavior_planner/pipeline/suite.py` | The one place where the concrete implementations are chosen. |
| `src/behavior_planner/analysis/metrics.py` | Per-scenario and per-suite metrics. |
| `src/behavior_planner/analysis/report.py` | Rendering of the two Markdown tables above. |
| `src/behavior_planner/analysis/figures.py` | The three figures. |
| `examples/` | Wiring scripts, no logic. |

Each layer depends only on the ones above it. The model layer performs no input or output
and knows nothing about planners; the algorithm layer draws nothing and writes nothing;
the pipeline layer chooses the implementations and produces the trace; the analysis layer
reads the trace and nothing else.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

333 tests run in about 9 seconds. The first tier exercises all thirty state and event
pairs of the transition table individually, checks that the car following model converges
to the closed-form equilibrium gap and to the free flow speed from either side and never
produces a negative speed, checks the quintic against its closed-form peak derivatives,
and checks that the ego overtakes a slow leader when the adjacent lane is clear and does
not when it is blocked. Two tests in `tests/test_safety.py` construct the conflict the
gate exists for: a situation in which the cost function ranks a lane change ahead of lane
keeping and the gate refuses it anyway, and a sweep showing that raising the progress
weight by three orders of magnitude does not buy the manoeuvre.

The second tier compares a fresh run of all seven scenarios against
`tests/data/reference_run.json`. What it pins is chosen from two measurements rather than
from convenience. Counts, veto classifications and the behaviour state sequence are
compared exactly, which is safe because the smallest gap between the best two admissible
candidates anywhere in the suite is of order `1e-3` in cost units, ten orders of magnitude
above floating point noise; a test asserts that margin so the exact pins stay honest.
Continuous aggregates are compared at a relative tolerance of `1e-6`, chosen because
perturbing the ego's free flow speed by a relative `1e-9` moves them by less than `1e-9`
relative and changes no discrete outcome at all. No instantaneous position or speed late
in a run is pinned in any form.

The third tier runs each script in `examples/` as a subprocess under a reduced duration,
writing figures into a temporary directory.

## References

Models:

- M. Treiber, A. Hennecke and D. Helbing, "Congested traffic states in empirical
  observations and microscopic simulations", Physical Review E 62(2), 2000, pages 1805 to
  1824. DOI [10.1103/PhysRevE.62.1805](https://doi.org/10.1103/PhysRevE.62.1805). The
  Intelligent Driver Model and the highway parameter set used for the defaults.
- A. Kesting, M. Treiber and D. Helbing, "General lane-changing model MOBIL for
  car-following models", Transportation Research Record 1999(1), 2007, pages 86 to 94.
  DOI [10.3141/1999-10](https://doi.org/10.3141/1999-10). The lane change model that
  drives the traffic, its safety criterion, its politeness factor and its asymmetric
  keep-right variant.
- M. Werling, J. Ziegler, S. Kammel and S. Thrun, "Optimal trajectory generation for
  dynamic street scenarios in a Frenet frame", IEEE International Conference on Robotics
  and Automation, 2010, pages 987 to 993.
  DOI [10.1109/ROBOT.2010.5509799](https://doi.org/10.1109/ROBOT.2010.5509799). The
  Frenet frame formulation and the quintic lateral polynomial.
- M. Treiber and A. Kesting, "Traffic Flow Dynamics: Data, Models and Simulation",
  Springer, 2013.
  DOI [10.1007/978-3-642-32460-4](https://doi.org/10.1007/978-3-642-32460-4). The
  reference treatment of both models, including the equilibrium relations used in the
  tests.
- J. Wei, J. M. Snider, T. Gu, J. M. Dolan and B. Litkouhi, "A behavioral planning
  framework for autonomous driving", IEEE Intelligent Vehicles Symposium, 2014, pages 458
  to 464. DOI [10.1109/IVS.2014.6856582](https://doi.org/10.1109/IVS.2014.6856582). The
  cost-based behaviour layer this planner follows in structure.
- J. Ziegler and C. Stiller, "Spatiotemporal state lattices for fast trajectory planning
  in dynamic on-road driving scenarios", IEEE/RSJ International Conference on Intelligent
  Robots and Systems, 2009, pages 1879 to 1884.
  DOI [10.1109/IROS.2009.5354448](https://doi.org/10.1109/IROS.2009.5354448). The lattice
  alternative to a behaviour state machine, considered and rejected in the design notes.
- P. Bender, J. Ziegler and C. Stiller, "Lanelets: efficient map representation for
  autonomous driving", IEEE Intelligent Vehicles Symposium, 2014, pages 420 to 425.
  DOI [10.1109/IVS.2014.6856487](https://doi.org/10.1109/IVS.2014.6856487). The lane
  topology representation a curved or branching road would require, noted as a limitation.

Dependencies:

- [numpy](https://numpy.org/) (BSD 3-Clause). The seeded PCG64 generator that places the
  random traffic, and the percentile and mean reductions in the analysis layer. The
  simulation loop itself uses plain Python floats, which is deliberate: it removes the
  reduction order of a linear algebra kernel from the set of things a run can depend on.
- [matplotlib](https://matplotlib.org/) (matplotlib license, a BSD-style permissive
  license). The three figures.
- [pytest](https://docs.pytest.org/) (MIT), [ruff](https://docs.astral.sh/ruff/) (MIT),
  and [mypy](https://mypy-lang.org/) (MIT). Development only: test running, linting, and
  type checking.

## License

Released under the MIT license. See [LICENSE](LICENSE).
