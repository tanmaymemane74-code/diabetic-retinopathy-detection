"""
FastAPI REST Routes for Diabetic Retinopathy Clinical Decision Support System.
"""

import os
from pathlib import Path
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.inference_service import DRInferenceService
from src.utils.config import UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["Diabetic Retinopathy CDSS"])
service = DRInferenceService()


class GuidanceQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3


class ReportRequest(BaseModel):
    patient_id: str
    predicted_grade: int
    confidence: float
    probabilities: list[float]
    eye: Optional[str] = "Right Eye (OD)"


@router.get("/health")
def health_check():
    """Service health and model readiness check."""
    return {
        "status": "healthy",
        "service": "Diabetic Retinopathy CDSS",
        "model_architecture": "EfficientNetB3",
        "xai_module": "Grad-CAM",
        "rag_vector_db": "FAISS",
    }


@router.post("/predict")
@router.post("/screen")
async def predict_retinopathy(
    file: UploadFile = File(...),
    patient_id: str = Form("PATIENT-ANON"),
    eye: str = Form("Right Eye (OD)"),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    save_record: Optional[bool] = Form(True),
):
    """Predict Diabetic Retinopathy severity and generate complete clinical decision support."""
    # Validate extension
    allowed = {".jpg", ".jpeg", ".png"}
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {list(allowed)}",
        )

    # Save uploaded file
    upload_path = Path(UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex[:10]}_{file.filename}"
    saved_file = upload_path / temp_filename

    with open(saved_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = service.analyze_fundus_image(
            image_path=str(saved_file),
            patient_id=patient_id,
            eye=eye,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/explain")
async def explain_fundus(
    file: UploadFile = File(...),
    pred_index: Optional[int] = Form(None),
):
    """Generate Grad-CAM activation heatmap for uploaded retinal image."""
    upload_path = Path(UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    temp_filename = f"explain_{uuid.uuid4().hex[:8]}_{file.filename}"
    saved_file = upload_path / temp_filename

    with open(saved_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        preprocessed = service.preprocessor.preprocess_single(str(saved_file))
        if preprocessed is None:
            raise ValueError("Preprocessing failed.")

        xai_res = service.gradcam.explain_prediction(
            image_path=str(saved_file),
            preprocessed_img=preprocessed,
            save_figure=True,
        )
        return xai_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {str(e)}")


@router.post("/clinical-guidance")
def get_clinical_guidance(query_data: GuidanceQuery):
    """Retrieve semantically relevant clinical guidelines (AAO, ADA)."""
    try:
        results = service.knowledge_retriever.retrieve(
            query=query_data.query, top_k=query_data.top_k or 3
        )
        return {"query": query_data.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guidance retrieval error: {str(e)}")


@router.get("/clinical-guidance/{grade}")
def get_clinical_guidance_for_grade(grade: int):
    """Retrieve semantically relevant clinical guidelines for a specific DR grade (0-4)."""
    if grade not in range(5):
        raise HTTPException(status_code=400, detail="Grade must be an integer between 0 and 4.")
    try:
        action = service.decision_engine.get_referral_recommendation(grade)
        guidelines = service.knowledge_retriever.retrieve_for_grade(grade, top_k=2)
        return {
            "grade": grade,
            "suggested_action": action,
            "guidelines": guidelines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guidance retrieval error: {str(e)}")


@router.get("/reports")
def list_screenings(limit: int = 50, offset: int = 0):
    """Retrieve historical patient screening records and analytics."""
    records = service.database.get_all_screenings(limit=limit, offset=offset)
    stats = service.database.get_screening_stats()
    return {"stats": stats, "screenings": records}


@router.get("/reports/{screening_id}")
def get_screening_record(screening_id: int):
    """Get detailed screening report record by ID."""
    rec = service.database.get_screening(screening_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Screening record not found.")
    return rec


@router.get("/reports/{screening_id}/pdf")
def download_pdf_report(screening_id: int):
    """Download the clinical screening PDF report."""
    rec = service.database.get_screening(screening_id)
    if not rec or not rec.get("report_path") or not os.path.exists(rec["report_path"]):
        raise HTTPException(status_code=404, detail="PDF report file not found on disk.")
    return FileResponse(
        path=rec["report_path"],
        media_type="application/pdf",
        filename=Path(rec["report_path"]).name,
    )
