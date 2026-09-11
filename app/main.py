from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "FastAPI is running successfully"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }