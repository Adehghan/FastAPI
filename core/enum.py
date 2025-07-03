from enum import Enum


class ModelTypeEnum(int, Enum):
    LinearRegression = 1
    MultiLinearRegression = 2
    Lasso = 3
    Ridge = 4 