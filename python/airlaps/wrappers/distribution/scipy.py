from typing import Union

from scipy.stats import rv_continuous, rv_discrete

from airlaps.core import Distribution, T

__all__ = ['ScipyDistribution']


class ScipyDistribution(Distribution[T]):
    """This class wraps a SciPy distribution (rv_continuous or rv_discrete from scipy.stats) as an AIRLAPS
    distribution.

    !!! warning
        Using this class requires SciPy to be installed.
    """

    def __init__(self, scipy_distribution: Union[rv_continuous, rv_discrete]) -> None:
        """Initialize ScipyDistribution.

        # Parameters
        scipy_distribution: The SciPy distribution (rv_continuous or rv_discrete from scipy.stats) to wrap.
        """
        super().__init__()
        self._scipy_distribution = scipy_distribution

    def unwrapped(self) -> Union[rv_continuous, rv_discrete]:
        """Unwrap the SciPy distribution (rv_continuous or rv_discrete from scipy.stats) and return it.

        # Returns
        The original SciPy distribution.
        """
        return self._scipy_distribution

    def sample(self) -> T:
        return self._scipy_distribution.rvs()
