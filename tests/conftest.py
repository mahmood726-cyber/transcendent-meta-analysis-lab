import numpy as np
import pytest
from core.types import MAData

@pytest.fixture
def small_5():
    """5 studies with hand-calculable IV-weighted mean."""
    return MAData(
        effect=np.array([-0.5, -0.3, 0.0, 0.2, -0.1]),
        se=np.array([0.2, 0.3, 0.1, 0.4, 0.15]),
    )

@pytest.fixture
def bimodal_20():
    """Two well-separated groups for clustering tests."""
    rng = np.random.RandomState(42)
    effects = np.concatenate([rng.normal(-1.0, 0.1, 10), rng.normal(0.5, 0.1, 10)])
    ses = np.full(20, 0.15)
    groups = np.array(["A"] * 10 + ["B"] * 10)
    return MAData(effect=effects, se=ses, group=groups)

@pytest.fixture
def homogeneous_10():
    """10 studies from N(0, 0.3^2), tau^2=0."""
    rng = np.random.RandomState(123)
    effects = rng.normal(0.0, 0.05, 10)
    ses = np.full(10, 0.3)
    return MAData(effect=effects, se=ses)

@pytest.fixture
def single_study():
    """k=1 edge case."""
    return MAData(effect=np.array([0.5]), se=np.array([0.2]))

@pytest.fixture
def pair_studies():
    """k=2 edge case."""
    return MAData(effect=np.array([-0.3, 0.1]), se=np.array([0.2, 0.3]))

@pytest.fixture
def temporal_30():
    """30 studies with a known shift at index 15."""
    rng = np.random.RandomState(99)
    effects = np.concatenate([rng.normal(0.0, 0.1, 15), rng.normal(-0.5, 0.1, 15)])
    ses = np.full(30, 0.2)
    years = np.arange(2000, 2030)
    return MAData(effect=effects, se=ses, year=years)

@pytest.fixture
def with_outlier():
    """10 studies near 0 + 1 outlier at 5.0."""
    rng = np.random.RandomState(77)
    effects = np.concatenate([rng.normal(0.0, 0.1, 10), [5.0]])
    ses = np.concatenate([np.full(10, 0.2), [0.3]])
    return MAData(effect=effects, se=ses)

@pytest.fixture
def clustered_reviews():
    """10 reviews x 7 outcomes each."""
    rng = np.random.RandomState(55)
    effects, ses, groups = [], [], []
    for r in range(10):
        review_mean = rng.normal(0, 0.5)
        for o in range(7):
            effects.append(review_mean + rng.normal(0, 0.15))
            ses.append(rng.uniform(0.1, 0.4))
            groups.append(f"R{r:02d}")
    return MAData(effect=np.array(effects), se=np.array(ses), group=np.array(groups))
