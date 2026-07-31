"""Declarative scenario definitions and the standard suite.

A scenario is data, not code. Everything that varies between scenarios, the
road, the ego's starting pose, the scripted vehicles, the random fill and the
seed, is a field, so a scenario can be serialised, compared and reproduced
exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from behavior_planner.model.config import DriverParams, IdmParams, MobilParams
from behavior_planner.model.road import Road
from behavior_planner.model.vehicle import EGO_ID, Vehicle, VehicleShape

__all__ = ["RandomFill", "Scenario", "VehicleSpec", "build_vehicles", "standard_suite"]


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """One vehicle placed explicitly."""

    lane: int
    s: float
    speed: float
    desired_speed: float
    holds_lane: bool = False
    """Whether this vehicle refuses every lane change MOBIL offers it.

    MOBIL with a positive politeness factor makes a slow leader pull aside for a
    faster follower, which is realistic and is also exactly what a scenario
    designed to obstruct the ego must not do. A lane-holding vehicle stands in
    for a driver who does not yield, or for a heavy vehicle restricted to its
    lane, and it is what makes the scripted scenarios test the ego rather than
    the courtesy of the traffic.
    """


@dataclass(frozen=True, slots=True)
class RandomFill:
    """Traffic drawn from a seeded generator rather than placed by hand.

    Vehicles are laid out at even spacing within each lane and then displaced by
    a uniform jitter, which breaks the artificial symmetry of a perfect platoon
    without letting two vehicles start on top of each other. Desired speeds are
    normal about :attr:`speed_mean` and are clipped to a positive range.
    """

    lanes: tuple[int, ...]
    per_lane: int
    speed_mean: float
    """Mean free flow speed in lane 0, in metres per second."""

    speed_gradient: float = 0.0
    """Increase in the mean free flow speed per lane index, in metres per second.

    Motorway traffic sorts itself by speed, and a simulation in which every lane
    runs at the same mean gives a lane change planner nothing to decide: no lane
    is better than any other and the correct policy is to stay put. A positive
    gradient makes the left lanes faster, which is what puts the progress term
    and the keep-right preference in opposition.
    """

    speed_sigma: float = 2.0
    jitter: float = 8.0
    minimum_clearance: float = 30.0
    """Smallest same-lane separation a placement may produce, in metres.

    A run that starts with two vehicles a car length apart records a time
    headway of a fraction of a second before the simulation has taken a single
    step, which says nothing about the planner. A slot that would violate this
    clearance is left empty instead. Half this distance is required between
    neighbouring lanes, because two vehicles abreast with a large speed
    difference are one decision cycle away from a cut-in that no lane change
    model would refuse.
    """

    holds_lane: bool = False
    """Whether these vehicles refuse every lane change MOBIL offers them."""

    def __post_init__(self) -> None:
        """Reject a fill the placement routine cannot honour."""
        if self.per_lane < 0:
            raise ValueError(f"per_lane must not be negative, got {self.per_lane}")
        if self.speed_sigma < 0.0:
            raise ValueError(f"speed_sigma must not be negative, got {self.speed_sigma}")
        if self.jitter < 0.0:
            raise ValueError(f"jitter must not be negative, got {self.jitter}")
        if self.minimum_clearance <= 0.0:
            raise ValueError(
                f"minimum_clearance must be positive, got {self.minimum_clearance}"
            )


@dataclass(frozen=True, slots=True)
class Scenario:
    """A reproducible traffic situation."""

    name: str
    description: str
    road: Road
    ego: VehicleSpec
    duration: float = 60.0
    seed: int = 0
    scripted: tuple[VehicleSpec, ...] = ()
    fill: RandomFill | None = None
    shape: VehicleShape = field(default_factory=VehicleShape)

    def __post_init__(self) -> None:
        """Reject a scenario whose ego is not on the road."""
        if not self.road.contains_lane(self.ego.lane):
            raise ValueError(f"ego lane {self.ego.lane} is not on the road of {self.name!r}")
        if self.duration <= 0.0:
            raise ValueError(f"duration must be positive, got {self.duration}")

    def with_duration(self, duration: float) -> Scenario:
        """A copy of this scenario shortened or lengthened to ``duration``."""
        return replace(self, duration=duration)


def _driver(desired_speed: float, *, holds_lane: bool = False) -> DriverParams:
    """Driver parameters at a given free flow speed.

    A lane-holding driver is given an unreachable switching threshold, which is
    MOBIL's own way of saying that no incentive is large enough.
    """
    threshold = math.inf if holds_lane else MobilParams().changing_threshold
    return DriverParams(
        idm=IdmParams(desired_speed=desired_speed),
        mobil=MobilParams(changing_threshold=threshold),
    )


def build_vehicles(scenario: Scenario) -> tuple[Vehicle, ...]:
    """Instantiate every vehicle of ``scenario``.

    The ego is always identifier :data:`EGO_ID` and always first in the tuple.
    Placement consumes the generator in a fixed order, lane by lane and slot by
    slot, so the same seed gives the same traffic on any machine.
    """
    road = scenario.road
    vehicles = [
        Vehicle(
            vehicle_id=EGO_ID,
            s=road.wrap(scenario.ego.s),
            d=road.lane_center(scenario.ego.lane),
            speed=scenario.ego.speed,
            shape=scenario.shape,
            driver=_driver(scenario.ego.desired_speed),
        )
    ]
    next_id = EGO_ID + 1
    for spec in scenario.scripted:
        vehicles.append(
            Vehicle(
                vehicle_id=next_id,
                s=road.wrap(spec.s),
                d=road.lane_center(spec.lane),
                speed=spec.speed,
                shape=scenario.shape,
                driver=_driver(spec.desired_speed, holds_lane=spec.holds_lane),
            )
        )
        next_id += 1

    fill = scenario.fill
    if fill is not None and fill.per_lane > 0:
        rng = np.random.default_rng(scenario.seed)
        spacing = road.length / fill.per_lane
        occupied = [(vehicle.s, road.nearest_lane(vehicle.d)) for vehicle in vehicles]
        for position, lane in enumerate(fill.lanes):
            # Stagger the lanes against each other so vehicles do not start the
            # run abreast, which is neither typical of motorway traffic nor a
            # useful initial condition for a lane change planner.
            stagger = spacing * position / len(fill.lanes)
            mean = fill.speed_mean + fill.speed_gradient * lane
            for slot in range(fill.per_lane):
                offset = float(rng.uniform(-fill.jitter, fill.jitter))
                s = road.wrap(slot * spacing + stagger + offset)
                speed = float(rng.normal(mean, fill.speed_sigma))
                speed = min(max(speed, 5.0), road.speed_limit)
                if _too_close(road, occupied, s, lane, fill.minimum_clearance):
                    continue
                occupied.append((s, lane))
                vehicles.append(
                    Vehicle(
                        vehicle_id=next_id,
                        s=s,
                        d=road.lane_center(lane),
                        speed=speed,
                        shape=scenario.shape,
                        driver=_driver(speed, holds_lane=fill.holds_lane),
                    )
                )
                next_id += 1
    return tuple(vehicles)


def _too_close(
    road: Road, occupied: list[tuple[float, int]], s: float, lane: int, clearance: float
) -> bool:
    """True when placing at ``(s, lane)`` would start the run in a conflict.

    Neighbouring lanes are checked at half the clearance. Two vehicles abreast
    in adjacent lanes with a large speed difference are one MOBIL cycle away
    from a cut-in that no lane change model would have refused, and a metric
    taken from the first second of such a run measures the placement, not the
    planner.
    """
    for other_s, other_lane in occupied:
        distance = abs(other_lane - lane)
        if distance > 1:
            continue
        required = clearance if distance == 0 else 0.5 * clearance
        if abs(road.separation(other_s, s)) < required:
            return True
    return False


def standard_suite() -> tuple[Scenario, ...]:
    """The six scenarios the reported metrics are measured on.

    The first three are scripted so that the correct behaviour is unambiguous
    and can be asserted directly. The last three are seeded fills that exercise
    the planner against traffic which is itself changing lane.
    """
    three_lane = Road(lane_count=3, length=1200.0)
    return (
        Scenario(
            name="free_flow",
            description="Empty road. The ego should reach its free flow speed and stay right.",
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=20.0, desired_speed=31.0),
            duration=60.0,
            seed=1,
        ),
        Scenario(
            name="slow_leader",
            description=(
                "A slow vehicle ahead in the ego lane with the adjacent lane clear. "
                "The ego should overtake and then return to the right."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=28.0, desired_speed=31.0),
            duration=60.0,
            seed=2,
            scripted=(
                VehicleSpec(lane=0, s=60.0, speed=20.0, desired_speed=20.0, holds_lane=True),
            ),
        ),
        Scenario(
            name="blocked_overtake",
            description=(
                "The same slow leader, with the middle lane occupied alongside the ego "
                "by traffic at matching speed. The ego should hold its lane."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=28.0, desired_speed=31.0),
            duration=60.0,
            seed=3,
            scripted=(
                VehicleSpec(lane=0, s=60.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=1, s=8.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=1, s=-22.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=1, s=38.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=1, s=68.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=2, s=10.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=2, s=-20.0, speed=20.0, desired_speed=20.0, holds_lane=True),
                VehicleSpec(lane=2, s=40.0, speed=20.0, desired_speed=20.0, holds_lane=True),
            ),
        ),
        Scenario(
            name="gap_wait",
            description=(
                "A slow lane-holding leader ahead, with a stream of faster traffic in "
                "the middle lane. The ego has every reason to move left and has to "
                "wait for a gap the safety gate will accept."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=26.0, desired_speed=31.0),
            duration=60.0,
            seed=7,
            scripted=(
                VehicleSpec(lane=0, s=70.0, speed=18.0, desired_speed=18.0, holds_lane=True),
                VehicleSpec(lane=1, s=4.0, speed=28.0, desired_speed=28.0, holds_lane=True),
                VehicleSpec(lane=1, s=-56.0, speed=28.0, desired_speed=28.0, holds_lane=True),
                VehicleSpec(lane=1, s=-116.0, speed=28.0, desired_speed=28.0, holds_lane=True),
                VehicleSpec(lane=1, s=-176.0, speed=28.0, desired_speed=28.0, holds_lane=True),
                VehicleSpec(lane=1, s=-236.0, speed=28.0, desired_speed=28.0, holds_lane=True),
            ),
        ),
        Scenario(
            name="light_traffic",
            description=(
                "Eight vehicles per lane on a ring, sorted by speed with the right "
                "lane slowest. The traffic itself changes lane under MOBIL."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=22.0, desired_speed=31.0),
            duration=60.0,
            seed=4,
            fill=RandomFill(
                lanes=(0, 1, 2),
                per_lane=8,
                speed_mean=22.0,
                speed_gradient=3.0,
                speed_sigma=2.0,
            ),
        ),
        Scenario(
            name="dense_traffic",
            description=(
                "Sixteen vehicles per lane, near the density at which lane changes "
                "stop paying for themselves."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=20.0, desired_speed=31.0),
            duration=60.0,
            seed=5,
            fill=RandomFill(
                lanes=(0, 1, 2),
                per_lane=16,
                speed_mean=20.0,
                speed_gradient=2.5,
                speed_sigma=2.0,
            ),
        ),
        Scenario(
            name="slow_right_lane",
            description=(
                "A slow lane-holding platoon filling the right lane, standing in for "
                "heavy vehicles, with the two left lanes clear. The keep-right "
                "preference has to yield to the progress term."
            ),
            road=three_lane,
            ego=VehicleSpec(lane=0, s=0.0, speed=22.0, desired_speed=31.0),
            duration=60.0,
            seed=6,
            fill=RandomFill(
                lanes=(0,),
                per_lane=14,
                speed_mean=18.0,
                speed_sigma=1.5,
                holds_lane=True,
            ),
        ),
    )
