import os, sys
import re

from typing import Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import BackgroundTasks

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shortuuid
import http

from databases import Database
from fastapi import Depends
from starlette.requests import Request

from collections import Counter
import difflib

from koalpaca import gen

print(f"==================== chatserver_fastapi.py ====================")

class Chat(BaseModel):
    id: str = shortuuid.uuid()
    description: Union[str, None] = None
    contents: str
    
class Message(BaseModel):
    # id: str = shortuuid.uuid()
    text: str
    sender: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="./templates/static"), name="static")
templates = Jinja2Templates(directory="./templates")

# CORS
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/conversation", status_code=200)
def conversation(chat: Chat):
    print(f"chatserver: {chat} received.")

    chat_dict = chat.dict()
    if chat.contents:
        print(f"received message: {chat.contents}\n\n")
        
        result = gen(chat.contents)
        print(f"generated message: {result}\n\n")
        
        try:
            splits = result.split("###")
            result = splits[2] if len(splits) >= 3 else result
            result = result.split(":")[-1]
            
            chat_dict.update({"contents":result})
        except Exception as e:
            print(e)
    return chat_dict
