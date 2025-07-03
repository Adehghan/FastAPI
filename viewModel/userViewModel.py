from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class GenderEnum(int, Enum):
    Female = 1
    Male = 2

class userNodel(BaseModel):
    fname: str
    lname: str
    gender: GenderEnum
    age: int = Field(gt = 0) # Greater Than
    national_code: Optional[str] = Field(None, min_length = 10)
    username: str
    password: str = Field(..., min_length=10)
    remark: Optional[str]

