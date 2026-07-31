# Design notes for Behavior Planner Fsm

## Method selection

### Why a finite state machine at the behaviour layer

The manoeuvre a highway vehicle is executing is genuinely discrete. It is either holding
its lane, working towards a change, or crossing a boundary, and the difference between
those is not a matter of degree. Representing it as a state carries three things a
continuous formulation does not: the manoeuvre persists across planning cycles, the set of
things that may happen next is small and enumerable, and a transition that should not
happen can be named and refused.

The five states are the standard set for this problem, and the structure follows Wei and
colleagues (2014): a behaviour layer that enumerates a small set of discrete manoeuvres,
scores them with a weighted cost, and hands the winner to a trajectory layer. The two
prepare states are not decoration. A vehicle that wants a gap in the next lane has to
adjust its longitudinal behaviour first, and a prepare state is where that adjustment
lives: in `PREPARE_LANE_CHANGE_LEFT` the ego follows the more restrictive of its own
leader and the target lane's leader, so it drops back rather than driving past the gap it
wants. The `gap_wait` scenario is the case that makes this visible.

Two properties are enforced rather than assumed.

*Totality.* `algorithm/fsm.py` declares the legal pairs in `TRANSITIONS` and the illegal
pairs in `REJECTED`, and a check that runs at import refuses to load the module unless the
two are disjoint and their union is the full thirty element product of the state and event
alphabets. A pair that is neither declared legal nor declared illegal is an error the
moment the module loads. `tests/test_fsm.py` exercises all thirty pairs individually.

*Explicit rejection.* An illegal pair raises `IllegalTransitionError`. It does not return
the current state. The distinction matters because the silent version produces a vehicle
that quietly does nothing, which is indistinguishable from a vehicle that has correctly
decided to do nothing, and the difference is a bug that survives review.

### Why the Intelligent Driver Model and MOBIL

Both are published, calibrated against measurement, and simple enough to state in a
paragraph, which makes them checkable. The Intelligent Driver Model gives an acceleration
that is a smooth function of speed, gap and closing rate, and is collision free for a
leader that does not brake harder than the follower's comfortable deceleration. Its
equilibrium gap is closed form,

```
s_eq = (s0 + v * T) / sqrt(1 - (v / v0)^delta),
```

which is what `tests/test_idm.py` checks the integrated dynamics against, rather than
against a recorded number. The defaults are the paper's highway calibration.

MOBIL matters more than it might appear. Without a lane change model for the traffic, the
surrounding vehicles hold their lanes, every gap the ego sees when it starts a manoeuvre
is still there when it finishes, and the planner is being evaluated against a problem that
has had its difficulty removed. With MOBIL at a politeness of 0.25 the traffic competes
for the same gaps, and it also produces courtesy: a slow leader will sometimes pull aside
for a faster follower. That behaviour is correct, and it is also inconvenient for a
scenario designed to obstruct the ego, which is why the scripted obstructions carry a
`holds_lane` flag that raises their MOBIL switching threshold beyond reach. The flag is
declared on the scenario, not buried in the model, so a reader can see which vehicles are
props and which are traffic.

### Why the safety gate is a separate object rather than a fifth cost term

This is the central structural decision of the repository, and it is worth stating the
argument rather than the conclusion.

A weighted cost function computes `w1 * c1 + w2 * c2 + ...`, and every term in that sum is
negotiable by construction. If safety appears as `w_safety * c_safety`, then for any
finite `w_safety` there is a progress advantage large enough to outweigh it. That is not a
tuning error to be avoided by choosing `w_safety` carefully; it is what addition means.
The situations in which the progress advantage is largest, a slow obstruction ahead and a
fast lane beside it, are exactly the situations in which the temptation to accept a
marginal gap is greatest. A cost function trading safety against speed is not a hypothesis
about what might go wrong. It is the predictable behaviour of the arithmetic.

The gate therefore:

- is a separate class, `GapAndDecelerationGate`, whose only fields are its limits and a
  car following model, checked structurally by
  `tests/test_safety.py::test_the_gate_reads_none_of_the_cost_configuration`;
- reads its thresholds from `SafetyLimits`, which is a different dataclass from
  `CostConfig`, so the cost function's notion of a roomy gap and the gate's notion of a
  permitted gap cannot drift into being the same number by accident;
- returns a `SafetyVerdict`, which carries a boolean and a named reason, not a penalty;
- is applied in `FiniteStateBehaviorPlanner._score_all` before any cost comparison, so a
  vetoed candidate is not in the set the minimum is taken over.

`tests/test_safety.py` constructs the conflict directly. The ego is behind a slow leader,
which makes the progress term prefer the next lane; a vehicle sits alongside it, just
behind, occupying the space it would move into; and the soft safety weight is set to zero,
which is the misconfiguration the gate exists to survive. Under those conditions the cost
function ranks the lane change ahead of lane keeping, the gate refuses it, and the planner
follows the gate. A second test raises the progress weight from 5 to 5000 and confirms
that the manoeuvre is still refused, which is the property a cost term could not have.

The soft safety term in the cost function is not redundant with the gate, and the
difference is measurable. Setting the safety weight to zero and running the suite changes
which veto reasons appear: with the term the gate raises `follower_gap` and its relatives,
and without it the gate also has to raise `occupied` and `leader_gap`. The cost term makes
the planner polite; the gate makes it safe. Only one of the two is load bearing.

### Why the lateral profile is a single quintic

The manoeuvre is a transition between two lateral offsets with zero lateral velocity and
acceleration at both ends. The unique quintic satisfying those six boundary conditions is
the minimum jerk transition, and its peak rate, acceleration and jerk are

```
1.875 * D / T,    (10 / sqrt(3)) * D / T^2,    60 * D / T^3,
```

for a displacement `D` over a duration `T`. Having them in closed form means the comfort of
a manoeuvre can be asserted rather than sampled, which is what `tests/test_lateral.py` and
`tests/test_trajectory.py` do. It also makes the lateral state of a vehicle a pure function
of its manoeuvre and the elapsed time rather than an integrated quantity, so two vehicles
given the same manoeuvre follow bit-identical paths and a seeded run is reproducible.

Werling and colleagues (2010) sample a family of such polynomials over terminal offset and
duration and select by a jerk-and-deviation cost. That family is not generated here. The
behaviour layer above has already fixed the terminal lateral offset to a lane centre, and
the duration is a comfort parameter rather than a decision variable, so the sampled family
would collapse to one member. The lateral profile is nevertheless kept behind the same
closed-form object, so widening it to a family is a change to one class.

## Rejected alternatives

### A spatiotemporal state lattice instead of a behaviour state machine

Ziegler and Stiller (2009) plan directly in a lattice over position, time and speed,
producing the manoeuvre as a by-product of the search rather than as a prior decision. It
would have removed the cost weights entirely, since the lattice search optimises a single
objective over trajectories rather than ranking discrete behaviours, and it generalises to
situations a five state machine does not cover, such as threading between two vehicles
that are themselves changing lane.

Rejected on two grounds. First, cost: a lattice dense enough to represent lane changes at
usable resolution is a search over thousands of nodes per cycle, against the two or three
candidate evaluations the state machine performs, and the whole scenario suite here runs
in 2.4 seconds. Second, and more important for a repository whose subject is the safety
boundary: a lattice makes that boundary harder to see, not easier. The veto becomes a
feasibility constraint distributed over edge expansion, and the question of why a
particular manoeuvre was refused is answered by a search trace rather than by a named
reason. The state machine was chosen because its decisions are inspectable, which is the
same reason the gate returns a `VetoReason` rather than a boolean.

### A learned policy

A policy trained on this simulator would likely beat these hand-tuned weights on mean
speed, and would not need the weight window described below. It was rejected because it
makes the failure mode this repository is about worse rather than better: a learned policy
has no separable safety layer unless one is added afterwards, and if one is added
afterwards then the interesting question is the design of that layer, which is what is
built here. A learned policy behind the `BehaviorPolicy` Protocol would be a natural
extension, and the gate would apply to it unchanged.

### Rule-based lane changing without a cost function

A decision table of the form "change left if the leader is slower than X and the gap
exceeds Y" is simpler than a weighted cost and easier to certify. It was rejected because
it does not compose: each new consideration multiplies the number of rules, and the
interaction between two rules is not visible from either. The cost formulation keeps the
number of things to tune at four and makes their trade explicit, at the price of the
tuning difficulty recorded below.

### An asymmetric MOBIL for the traffic

Kesting and colleagues describe an asymmetric variant enforcing the European keep-right
rule and the prohibition on passing on the right. It is implemented only partially here:
`MobilParams.right_bias` provides the lane bias term and defaults to zero, which selects
the symmetric rule. The passing prohibition, which requires a vehicle's acceleration to be
capped by that of the vehicle in the lane to its left, is not implemented. It would have
made the traffic more realistic for a European motorway and would have interacted with the
ego's own keep-right preference in a way worth studying, but it also makes the traffic
model considerably harder to state and to test, and the ego's behaviour is what is under
examination here.

### An exact rotated-rectangle collision test

Vehicles are tested as axis-aligned boxes in the Frenet frame: two bodies overlap when
their longitudinal separation is under the mean of their lengths and their lateral
separation is under the mean of their widths. On a straight reference line this is exact
for a vehicle aligned with the road and slightly conservative for one that is yawed mid
manoeuvre, which is the safe direction. A rotated rectangle test would have been more
faithful and was rejected as unjustified: the peak lateral speed of a 3.5 second lane
change is 1.98 m/s, which at 28 m/s is a yaw of about 4 degrees, and at that angle the
axis-aligned box overestimates the swept width by a few centimetres.

### A ring road rather than an open road

Vehicles that leave the end of an open road have to be replaced at the start, and the
replacement policy then determines the density, which becomes a free parameter that the
results depend on and that nobody can check. A ring keeps the density exactly constant and
makes the scenario fully specified by its initial condition. The cost is that a 60 second
run at 28 m/s laps a 1200 metre ring more than once, so the ego meets the same traffic
again; this is stated here because it affects how the lane change counts should be read.

## Known limitations

### The cost weights sit in a narrow window, and this is a property of the method

The two behaviours the planner must exhibit constrain the weights from opposite
directions.

To overtake, leaving a lane whose leader is slow must beat staying in it. A prepare state
scores the mean of the two lanes on progress, so the gain is half the progress weight
times the inefficiency of the current lane, and it must exceed `comfort` plus
`lane_preference` times the normalised lane distance.

To come back, once the obstruction is passed, nothing favours the right lane except the
preference term, so `lane_preference` times the normalised lane distance must exceed
`comfort`.

The second inequality bounds `comfort` above by roughly half of `lane_preference`. The
first then bounds `lane_preference` above by the smallest speed advantage that ought to
provoke a manoeuvre. With the defaults, `comfort = 0.10`, `lane_preference = 0.35` and a
normalised lane distance of 0.5 on a three lane road, the return inequality has a margin
of 0.075 cost units and the overtake inequality has a margin that depends on how slow the
obstruction is. Both hold, but neither holds by much.

This is not a defect of these particular numbers. It is what a hand-tuned linear cost
function is: a single weight vector asked to produce the right ranking in every situation
the vehicle will meet, where the situations differ by more than the weights can express.
The specific things it cannot do:

- It cannot make the return-right decision depend on how long the ego has been left,
  because the cost function is memoryless. A term for that would need state, and state in
  a cost function is a second machine hiding inside the first.
- It cannot distinguish a lane change worth 2 m/s for the next ten seconds from one worth
  2 m/s for the next two, because the progress term measures an instantaneous speed
  shortfall rather than an integral. A receding horizon formulation would, at the cost of
  having to predict the traffic.
- It cannot be tuned per situation without becoming a rule table with extra steps. Every
  situation the current weights rank wrongly is an argument for another term, and every
  term widens the space in which the previous tuning has to be rechecked.

What the design does instead is contain the consequences. The weights determine which
manoeuvre is preferred and nothing else; whether that manoeuvre is permitted is a separate
question answered by a separate object with separate thresholds. A mistuned weight vector
produces a planner that overtakes too eagerly or too rarely. It does not produce a planner
that merges into an occupied gap.

### The traffic prediction is a constant speed assumption

The gate evaluates the time to collision and the imposed deceleration from the current
speeds of the neighbours, and the trajectory generator extrapolates leaders at constant
speed over its horizon. A vehicle that brakes hard during the ego's 3.5 second lane change
is therefore not anticipated. The abort path covers this partially: the gate is
re-evaluated every planning cycle while the manoeuvre runs, and a change less than 40
percent complete is given up when the gate objects. Beyond that point the ego finishes, on
the grounds that reversing across a boundary it has already crossed is worse than
completing. A planner facing genuinely adversarial traffic would need a prediction model
rather than a constant speed extrapolation, and the interface for that would sit beside
`CarFollowingModel` in `algorithm/base.py`.

### The road is straight and has no topology

`Road` is a straight ring with a fixed number of parallel lanes of equal width. The Frenet
frame is kept so that a curved reference line could be substituted without touching the
layers above, but nothing here handles a lane that appears, ends, splits or merges, and the
lane index arithmetic assumes that lane `k + 1` is always the lane to the left of lane `k`.
A real road needs a topological representation such as the lanelets of Bender and
colleagues, and the change would reach into `Road.occupied_lanes`,
`BehaviorState.lane_offset` and the feasibility filter in the planner. Curvature would
additionally invalidate the axis-aligned collision box.

### The scenario suite is small and its scripted cases are hand-built

Seven scenarios, each run once at one seed, is enough to demonstrate the behaviours and not
enough to characterise the planner. The scenarios with hand-placed obstructions are built
to make one behaviour unambiguous each, and those obstructions hold their lanes, which is
a simplification declared on the scenario rather than hidden in the model. A proper
evaluation would sweep density and seed and report distributions rather than single runs.
The metrics module and the trace record are built to support that; the suite is not.

### Collisions are counted, not prevented by construction

Zero collisions across the suite is a measured result, not a guarantee. The Intelligent
Driver Model is collision free only against a leader that does not brake harder than the
follower's comfortable deceleration, and the safety gate reduces but does not eliminate the
risk of a cut-in during the ego's manoeuvre. The claim this repository makes is the one the
metric supports: under these seven scenarios, with these parameters, no two vehicles
overlapped at any step. A guarantee would require a reachability argument over the whole
state space, which is a different piece of work.

### The behaviour layer is the only thing being tested

There is no perception, no localisation and no tracking controller. The planner sees exact
positions and speeds of every vehicle, and the trajectory it generates is followed exactly.
Every number in the Results section should be read with that in mind: the gaps the gate
enforces carry no allowance for measurement error, and a planner facing estimated states
would need those thresholds widened by an amount that depends on the estimator.
