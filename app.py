import marimo
from fastapi import FastAPI

from page_registry import PAGE_SPECS


server = marimo.create_asgi_app()
for page in PAGE_SPECS:
    server = server.with_app(path=page.path, root=page.root)

app = FastAPI()
app.mount("/", server.build())
