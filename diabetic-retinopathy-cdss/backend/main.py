"""
FastAPI Main Application Entrypoint for Diabetic Retinopathy CDSS.
Provides RESTful APIs, Swagger documentation (/docs), and static file mounts.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.api import router
from src.utils.config import GRADCAM_DIR, REPORTS_DIR, UPLOAD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")

app = FastAPI(
    title="Diabetic Retinopathy Clinical Decision Support System (DR-CDSS)",
    description=(
        "MCS Academic Research Project REST API combining Deep Learning (EfficientNetB3), "
        "Explainable AI (Grad-CAM), Retrieval-Augmented Generation (RAG), and "
        "Rule-Based Clinical Referral Decision Support."
    ),
    version="1.0.0",
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for images and reports
for d in [UPLOAD_DIR, GRADCAM_DIR, REPORTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static/gradcam", StaticFiles(directory=str(GRADCAM_DIR)), name="gradcam")
app.mount("/static/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Include Router
app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy Clinical Decision Support System API is running.",
        "documentation": "/docs",
        "health_check": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
