"""Parameter scheduling configurations for YAML serialization.

This module provides YAML-serializable configurations for parameter schedules
that can be converted to the actual schedule objects used by the optimizer.
"""

from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict
from abc import ABC, abstractmethod

from boltz.model.potentials.schedules import (
    PiecewiseSchedule,
    PiecewiseStepFunction,
    ExponentialInterpolation,
    ExponentialInterpolationWithBounds,
    ResolutionScaling,
)


@dataclass
class ParameterSchedule(ABC):
    """Base class for parameter schedule configurations."""

    @abstractmethod
    def to_schedule(self) -> Any:
        """Convert this config to the actual schedule object."""
        pass


@dataclass
class ConstantScheduleConfig(ParameterSchedule):
    """Configuration for a constant parameter value."""

    value: Union[float, int]

    def to_schedule(self) -> Union[float, int]:
        return self.value


@dataclass
class PiecewiseScheduleConfig(ParameterSchedule):
    """Configuration for piecewise linear schedule.

    Parameters
    ----------
    breakpoints : List[float]
        Normalized time points (0-1) where schedule changes
    values : List[Union[float, int, ParameterSchedule]]
        Values at each segment. Can be constants or nested schedules.
    """

    breakpoints: List[float]
    values: List[Union[float, int, "ParameterSchedule"]]

    def to_schedule(self) -> PiecewiseSchedule:
        converted_values = []
        for value in self.values:
            if isinstance(value, ParameterSchedule):
                converted_values.append(value.to_schedule())
            else:
                converted_values.append(value)

        return PiecewiseSchedule(self.breakpoints, converted_values)


@dataclass
class PiecewiseStepScheduleConfig(ParameterSchedule):
    """Configuration for piecewise step function schedule."""

    thresholds: List[float]
    values: List[Union[float, int]]

    def to_schedule(self) -> PiecewiseStepFunction:
        return PiecewiseStepFunction(self.thresholds, self.values)


@dataclass
class ExponentialInterpolationConfig(ParameterSchedule):
    """Configuration for exponential interpolation schedule.

    Parameters
    ----------
    start : float
        Starting value
    end : float
        Ending value
    alpha : float
        Exponential parameter controlling curve shape
    """

    start: float
    end: float
    alpha: float

    def to_schedule(self) -> ExponentialInterpolation:
        return ExponentialInterpolation(
            start=self.start, end=self.end, alpha=self.alpha
        )


@dataclass
class ExponentialInterpolationWithBoundsConfig(ParameterSchedule):
    """Configuration for bounded exponential interpolation schedule."""

    start: float
    end: float
    alpha: float
    start_t: float
    end_t: float

    def to_schedule(self) -> ExponentialInterpolationWithBounds:
        return ExponentialInterpolationWithBounds(
            start=self.start,
            end=self.end,
            alpha=self.alpha,
            start_t=self.start_t,
            end_t=self.end_t,
        )


@dataclass
class ResolutionScalingConfig(ParameterSchedule):
    """Configuration for resolution-based parameter scaling."""

    resolution_schedule: ParameterSchedule
    reference_resolution: float
    base: Union[float, ParameterSchedule]

    def to_schedule(self) -> ResolutionScaling:
        base_schedule = (
            self.base.to_schedule()
            if isinstance(self.base, ParameterSchedule)
            else self.base
        )
        return ResolutionScaling(
            self.resolution_schedule.to_schedule(),
            self.reference_resolution,
            base=base_schedule,
        )


def infer_schedule_type(config_dict: Dict[str, Any]) -> Optional[str]:
    """Infer schedule type from configuration fields.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Dictionary that may represent a schedule configuration

    Returns
    -------
    Optional[str]
        Inferred schedule type or None if not a schedule
    """
    if not isinstance(config_dict, dict):
        return None

    # Check for exponential with bounds
    if all(
        field in config_dict for field in ["start", "end", "alpha", "start_t", "end_t"]
    ):
        return "exponential_bounds"

    # Check for basic exponential
    if all(field in config_dict for field in ["start", "end", "alpha"]):
        return "exponential"

    # Check for piecewise step
    if "thresholds" in config_dict and "values" in config_dict:
        return "piecewise_step"

    # Check for piecewise
    if "breakpoints" in config_dict and "values" in config_dict:
        return "piecewise"

    # Check for resolution scaling
    if "resolution_schedule" in config_dict and "reference_resolution" in config_dict:
        return "resolution_scaling"

    return None


def parse_schedule_config(
    config: Union[Dict[str, Any], float, int, ParameterSchedule],
) -> ParameterSchedule:
    """Parse a schedule configuration from various input formats.

    Parameters
    ----------
    config : Union[Dict[str, Any], float, int, ParameterSchedule]
        Schedule configuration in various formats

    Returns
    -------
    ParameterSchedule
        Parsed schedule configuration
    """
    if isinstance(config, (float, int)):
        return ConstantScheduleConfig(config)

    if isinstance(config, ParameterSchedule):
        return config

    if not isinstance(config, dict):
        raise ValueError(f"Invalid schedule config type: {type(config)}")

    schedule_type = config.get("type")
    if schedule_type is None:
        raise ValueError("Schedule config must specify 'type' field")

    # Remove type from config dict for dataclass construction
    config_data = {k: v for k, v in config.items() if k != "type"}

    if schedule_type == "constant":
        return ConstantScheduleConfig(**config_data)
    elif schedule_type == "piecewise":
        # Parse nested values (parse dict values as schedules with type inference)
        if "values" in config_data:
            parsed_values = []
            for v in config_data["values"]:
                if isinstance(v, dict):
                    if "type" in v:
                        # Explicit type provided
                        parsed_values.append(parse_schedule_config(v))
                    else:
                        # Try to infer type from fields
                        inferred_type = infer_schedule_type(v)
                        if inferred_type:
                            # Add type and parse as schedule
                            schedule_config = dict(v)
                            schedule_config["type"] = inferred_type
                            parsed_values.append(parse_schedule_config(schedule_config))
                        else:
                            # Not a schedule, keep as is
                            parsed_values.append(v)
                else:
                    parsed_values.append(v)
            config_data["values"] = parsed_values
        return PiecewiseScheduleConfig(**config_data)
    elif schedule_type == "piecewise_step":
        return PiecewiseStepScheduleConfig(**config_data)
    elif schedule_type == "exponential":
        return ExponentialInterpolationConfig(**config_data)
    elif schedule_type == "exponential_bounds":
        return ExponentialInterpolationWithBoundsConfig(**config_data)
    elif schedule_type == "resolution_scaling":
        # Parse nested schedules
        if "resolution_schedule" in config_data:
            config_data["resolution_schedule"] = parse_schedule_config(
                config_data["resolution_schedule"]
            )
        if "base" in config_data and isinstance(config_data["base"], dict):
            config_data["base"] = parse_schedule_config(config_data["base"])
        return ResolutionScalingConfig(**config_data)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
