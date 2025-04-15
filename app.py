from typing import Annotated, Callable, Coroutine
from fastapi.responses import HTMLResponse, RedirectResponse
import marimo
from fastapi import FastAPI, Form, Request, Response


# Create a marimo asgi app
server = (
    marimo.create_asgi_app()
    .with_app(path="", root="./pages/index.py")
    .with_app(path="/lesson_1", root="./pages/lesson_1.py")
    .with_app(path="/lesson_2", root="./pages/lesson_2.py")
    .with_app(path="/lesson_3", root="./pages/lesson_3.py")
)

# Create a FastAPI app
app = FastAPI()

# app.add_middleware(auth_middleware)
# app.add_route("/login", my_login_route, methods=["POST"])

app.mount("/", server.build())
