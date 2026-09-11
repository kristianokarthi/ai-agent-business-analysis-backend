from fastapi import FastAPI

app = FastAPI(
    title="AI Business Analyzer API",
    version="1.0.0",
)


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