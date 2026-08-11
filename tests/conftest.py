"""Fixtures shared by the test suite."""

from __future__ import annotations

import pytest

from behavior_planner.algorithm.cost import WeightedCostModel
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.mobil import MobilLaneChangeModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.road import Road
from behavior_planner.model.vehicle import EGO_ID, Vehicle, VehicleShape


@pytest.fixture
def road() -> Road:
    """A three lane ring road."""
    return Road(lane_count=3, length=1200.0)


@pytest.fixture
def config() -> PlannerConfig:
    """The default planner configuration."""
    return PlannerConfig()


@pytest.fixture
def car_following() -> IntelligentDriverModel:
    """The default car following model."""
    return IntelligentDriverModel()


@pytest.fixture
def lane_change_model(car_following: IntelligentDriverModel) -> MobilLaneChangeModel:
    """MOBIL wired to the default car following model."""
    return MobilLaneChangeModel(car_following=car_following)


@pytest.fixture
def gate(config: PlannerConfig, car_following: IntelligentDriverModel) -> GapAndDecelerationGate:
    """The default safety gate."""
    return GapAndDecelerationGate(limits=config.safety, car_following=car_following)


@pytest.fixture
def cost(config: PlannerConfig) -> WeightedCostModel:
    """The default cost model."""
    return WeightedCostModel(config.cost)


@pytest.fixture
def planner(cost: WeightedCostModel, gate: GapAndDecelerationGate) -> FiniteStateBehaviorPlanner:
    """The default behaviour policy."""
    return FiniteStateBehaviorPlanner(cost=cost, gate=gate)


def make_vehicle(
    road: Road,
    *,
    vehicle_id: int,
    lane: int,
    s: float,
    speed: float,
    desired_speed: float | None = None,
) -> Vehicle:
    """A vehicle centred in ``lane`` at ``s``."""
    from behavior_planner.model.config import DriverParams, IdmParams

    driver = DriverParams(
        idm=IdmParams(desired_speed=desired_speed if desired_speed is not None else speed)
    )
    return Vehicle(
        vehicle_id=vehicle_id,
        s=road.wrap(s),
        d=road.lane_center(lane),
        speed=speed,
        shape=VehicleShape(),
        driver=driver,
    )


def make_ego(road: Road, *, lane: int, s: float, speed: float, desired_speed: float) -> Vehicle:
    """The ego vehicle centred in ``lane`` at ``s``."""
    return make_vehicle(
        road,
        vehicle_id=EGO_ID,
        lane=lane,
        s=s,
        speed=speed,
        desired_speed=desired_speed,
    )
