@echo off
title Multi-Agent Research Assistant FastAPI (Port 8000)
echo Starting FastAPI Server on port 8000...
echo Swagger UI will be available at: http://localhost:8000/docs
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
pause
