from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any

import numpy as np
from dataclasses_json import DataClassJsonMixin


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):

    value: float | int | np.number | np.floating | np.ndarray | None
    maximize: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            if not isinstance(self.value, (float, int, np.number, np.floating)):
                raise TypeError(f"Metric value must be numeric, got {type(self.value)}")
            self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """Return True if *self* represents a better metric than *other*."""
        if self.value is None:
            return False
        if not isinstance(other, MetricValue) or other.value is None:
            return True

        if self.maximize != other.maximize:
            raise ValueError(
                f"Cannot compare metrics with different optimization directions: "
                f"{self.maximize} vs {other.maximize}"
            )

        if self.value == other.value:
            return False

        comp = self.value > other.value
        return comp if self.maximize else not comp  # type: ignore

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        return f"Metric{opt_dir}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    @property
    def value_npsafe(self):
        return self.value if self.value is not None else float("nan")


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
    