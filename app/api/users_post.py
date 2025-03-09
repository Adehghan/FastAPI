from enum import Enum
from fastapi import APIRouter, Response, status, Query, Body
from viewModel import userViewModel
from viewModel import roleViewModel


router = APIRouter(prefix='/user', tags=["Users"])


@router.post('/new')
def new_user(fname: str, lname: str, age: int, nationalCode: str, response: Response):
    if age <= 0:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {'The age is not valid.'}
    elif len(nationalCode) != 10:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {'The national code is not valid.'}
    else:
        return {'Saved successfully'}
    
    
@router.post('/create')
def create_user(user: userViewModel.userNodel,
                role: roleViewModel.roleModel,
                response: Response,
                id: int = Query(alias='userId', deprecated=True),
                content: str = Body(any, min_length=10, max_length=20, regex='^[A-Z].*')
                ):
    if user.age > 100:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {'The age is not valid.'}
    # elif len(user.national_code) != 10:
    #     response.status_code = status.HTTP_406_NOT_ACCEPTABLE
    #     return {'The national code is not valid.'}
    else:
        user.remark="test"
        result = {
            'error_code' : status.HTTP_200_OK,
            'message' : 'Saved successfully',
            'data' : user
        }

        response.status_code = status.HTTP_200_OK
        return result
