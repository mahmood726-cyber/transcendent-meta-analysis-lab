from dataclasses import dataclass, field
import numpy as np

@dataclass
class MAData:
    """Standard input for all meta-analysis methods."""
    effect: np.ndarray
    se: np.ndarray
    group: np.ndarray | None = None
    year: np.ndarray | None = None
    design: np.ndarray | None = None

    def __post_init__(self):
        assert len(self.effect) == len(self.se), "effect and se must have same length"
        assert len(self.effect) > 0, "MAData must have at least 1 study"
        assert np.all(self.se > 0), "All standard errors must be positive"

@dataclass
class MAResult:
    """Standard output from all meta-analysis methods."""
    estimate: float
    ci_lower: float
    ci_upper: float
    method: str
    details: dict = field(default_factory=dict)
