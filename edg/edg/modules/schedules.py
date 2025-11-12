import math
from abc import ABC
from typing import List, Dict, Union


class ParameterSchedule(ABC):
    def compute(self, t):
        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t):
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (
                math.exp(self.alpha * t) - 1
            ) / (math.exp(self.alpha) - 1)
        else:
            return self.start + (self.end - self.start) * t


class ExponentialInterpolationWithBounds(ParameterSchedule):
    def __init__(self, start, end, alpha, start_t, end_t):
        self.start = start
        self.end = end
        self.alpha = alpha
        self.start_t = start_t
        self.end_t = end_t

    def compute(self, t):
        if t < self.start_t:
            return self.start
        elif t > self.end_t:
            return self.end
        else:
            if self.alpha != 0:
                return self.start + (self.end - self.start) * (
                    math.exp(self.alpha * (t - self.start_t)) - 1
                ) / (math.exp(self.alpha * (self.end_t - self.start_t)) - 1)
            else:
                return self.start + (self.end - self.start) * (
                    (t - self.start_t) / (self.end_t - self.start_t)
                )


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]


class PiecewiseSchedule(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return (
            self.values[idx].compute(t)
            if isinstance(self.values[idx], ParameterSchedule)
            else self.values[idx]
        )


class ResolutionScaling(ParameterSchedule):
    """Resolution dependent scale, inspired by Levy et al. 2024"""

    def __init__(
        self,
        resolution_schedule,
        r0,
        base: Union[float, ParameterSchedule] = 0.005,
        invert=False,
    ):
        self.r0 = r0
        self.resolution_schedule = resolution_schedule
        self.base = base
        self.invert = invert

    def compute(self, t):
        rt = self.resolution_schedule.compute(t)
        if isinstance(self.base, ParameterSchedule):
            base_value = self.base.compute(t)
        else:
            base_value = self.base
        scaler = self.r0 / rt if self.invert else rt / self.r0
        return base_value * (scaler) ** 3


class Ramp(ParameterSchedule):
    """Series of exponential ramps, inspired by Rosetta FastRelax.

    Executes a sequence of exponential interpolations between a base value
    and specified targets, returning to the base value outside the active interval.
    """

    def __init__(
        self, base: float, start_t: float, end_t: float, ramps: List[Dict[str, float]]
    ):
        """Initialize the Ramp parameter schedule.

        Parameters
        ----------
        base : float
            Base value to return outside the active interval.
        start_t : float
            Start time of the ramp sequence.
        end_t : float
            End time of the ramp sequence.
        ramps : List[Dict[str, float]]
            List of ramp specifications, each containing:
            - 'target': Target value for this ramp (required)
            - 'alpha': Exponential parameter (optional, defaults to 0)
        """
        self.base = base
        self.start_t = start_t
        self.end_t = end_t
        self.ramps = ramps

        segment_duration = (end_t - start_t) / len(ramps)
        self.interpolators = []

        for i, ramp in enumerate(ramps):
            segment_start = start_t + i * segment_duration
            segment_end = segment_start + segment_duration

            interpolator = ExponentialInterpolationWithBounds(
                start=base,
                end=ramp["target"],
                alpha=ramp.get("alpha", 0),
                start_t=segment_start,
                end_t=segment_end,
            )
            self.interpolators.append(interpolator)

    def compute(self, t: float) -> float:
        """Compute the parameter value at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate the parameter.

        Returns
        -------
        float
            Parameter value at time t.
        """
        if t < self.start_t or t >= self.end_t:
            return self.base

        for interpolator in self.interpolators:
            if interpolator.start_t <= t < interpolator.end_t:
                return interpolator.compute(t)

        return self.interpolators[-1].end