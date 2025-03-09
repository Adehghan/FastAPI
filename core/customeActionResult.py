from typing import Any
from pydantic import BaseModel

class custom_action_result(BaseModel):
    error_code: int
    error_message: str
    data: Any
