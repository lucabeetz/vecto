from fastapi import FastAPI
from pydantic import BaseModel
from vecto import Vecto


class Input(BaseModel):
    text: str
    metadata: dict = {}


class Query(BaseModel):
    text: str


app = FastAPI()
vecto = Vecto()


@app.post("/add")
async def add(input: Input):
    uuid = vecto.add(input.text, input.metadata)
    return {"uuid": uuid}


@app.post("/query")
async def query(query: Query):
    metadata = vecto.query(query.text)
    return {"metadata": metadata}
