from fastapi import FastAPI
from typing import List
from starlette.middleware.cors import CORSMiddleware

from db import session
from model import UserTable, User

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/users")
async def read_users():
    users = session.query(UserTable).all()

    return users


@app.get("/users/{user_id}")
async def read_user(user_id: int):
    user = session.query(UserTable).filter(UserTable.id == user_id).first()
    return user


@app.post("/user")
async def create_users(name: str, email: str):
    user = UserTable()
    user.name = name
    user.email = email

    session.add(user)
    session.commit()

    return "{} created...".format(name)


@app.put("/users")
async def update_user(users: List[User]):
    for i in users:
        user = session.query(UserTable).filter(UserTable.id == i.id).first()
        user.name = i.name
        user.email = i.email
        session.commit()

    return "{} updated...".format(users[0].name)


@app.delete("/user")
async def delete_users(user_id: int):
    user = session.query(UserTable).filter(UserTable.id == user_id).delete()
    session.commit()

    return read_users
