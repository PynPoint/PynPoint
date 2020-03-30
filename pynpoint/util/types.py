"""
Additional custom types to make type hints easier to read.
"""

from typing import Union

import numpy as np


StaticAttribute = Union[str, float, int, np.generic]
NonStaticAttribute = Union[np.ndarray, tuple, list]
