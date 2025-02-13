import marimo
from fastapi import FastAPI

# Create a marimo asgi app
server = (
    marimo.create_asgi_app()
    .with_app(path="", root="./pages/index.py")
    .with_app(path="/dashboard", root="./pages/lesson_1.py")
    .with_app(path="/sales", root="./pages/lesson_2.py")
)

# Create a FastAPI app
app = FastAPI()

app.mount("/", server.build())

# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
