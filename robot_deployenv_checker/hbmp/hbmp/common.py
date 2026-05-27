import numpy as np
from typing import Tuple, Union, List, Dict


DTypeVerticesFaces = Tuple[np.ndarray, np.ndarray]
DTypeListVertices = List[np.ndarray]
DTypeVertices = np.ndarray
DTypeConvex = Union[DTypeVerticesFaces, DTypeListVertices]
DTypeState = np.ndarray
DTypeFloat = np.float32
DTypeDouble = np.float64
DTypeIndex = np.uint32
DTypeBool = np.bool_
INF_FLOAT = float("inf")
