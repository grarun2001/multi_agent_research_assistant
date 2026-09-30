import uvicorn

if __name__ == "__main__":
    print("Starting Multi-Agent Research Assistant API Server on port 8000...")
    print("Swagger Documentation: http://localhost:8000/docs")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
