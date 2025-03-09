from fastapi import APIRouter, Response, status
from typing import Optional

router = APIRouter(prefix='/user', tags=['Users'])


@router.get("/info/{fname}/{lname}/{age}")
def showFullName(fname, lname, age:int):
    return('Welcome %s %s %i' % (fname, lname, age))

@router.get('/infoQuery/{id}')
def showInfo(id: int, fname: str, lname: Optional['str']= None, age:int=20):
    return('Welcome %s %s %i' % (fname, lname, age))

@router.get('/getId')
def getId(id:int, respone:Response):   
    
    if id > 50:
        respone.status_code = status.HTTP_404_NOT_FOUND        
        return {'The id %i not found in the system.' % id}
    else:
        return {'The id is %i' % id}

@router.get('/getInfo')
def showInfo(fname:str, lname:str, age:Optional['int'] = 5, response: Response = status.HTTP_200_OK):
    if age <= 0:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'The age is not valid.'}
    else:
        return {'welcome %s %s' % (fname, lname)}