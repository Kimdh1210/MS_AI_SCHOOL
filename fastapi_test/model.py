from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
from db import Base
from db import ENGINE

class UserTable(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)

class User(BaseModel):
    id: int
    name: str
    email: str