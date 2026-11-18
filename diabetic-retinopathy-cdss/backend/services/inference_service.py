"""
Inference Service for Diabetic Retinopathy Clinical Decision Support.
Maintains cached singletons for the deep learning model, preprocessor,
quality assessor, Grad-CAM explainer, and RAG knowledge retriever.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.classification.efficientnet_model import build_efficientnet_b3
from src.clinical_decision.decision_engine import DRClinicalDecisionEngine
from src.explainability.gradcam import DRGradCAM
from src.llm.report_generator import DRReportGenerator
from src.preprocessing.preprocessor import DRImagePreprocessor
from src.preprocessing.quality_assessment import ImageQualityAssessor
from src.rag.knowledge_retriever import DRKnowledgeRetriever
from src.reporting.pdf_generator import DRPDFReportGenerator
from src.utils.config import (
    DATASET_DIR,
    DR_CLASSES,
    DR_DESCRIPTIONS,
    IMAGE_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    UPLOAD_DIR,
)
from src.utils.database import DRDatabase

logger = logging.getLogger("dr_cdss.service")


class DRInferenceService:
    """Central singleton service managing all model pipelines and clinical decision support."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        logger.info("Initializing DRInferenceService singleton...")
        self.preprocessor = DRImagePreprocessor(target_size=IMAGE_SIZE)
        self.quality_assessor = ImageQualityAssessor()
        self.decision_engine = DRClinicalDecisionEngine()
        self.knowledge_retriever = DRKnowledgeRetriever()
        self.report_generator = DRReportGenerator()
        self.pdf_generator = DRPDFReportGenerator()
        self.database = DRDatabase()

        self.model = None
        self.gradcam = None
        self._load_or_build_model()

        # Build knowledge base vector index
        try:
            self.knowledge_retriever.build_index()
        except Exception as e:
            logger.warning("Knowledge retriever index build warning: %s", e)

        self._initialized = True

    def _load_or_build_model(self):
        """Load trained weights or build initialized architecture."""
        import tensorflow as tf

        weights_candidates = [
            Path(MODEL_PATH),
            Path(MODELS_DIR) / "efficientnet_b3_best.keras",
            Path(MODELS_DIR) / "efficientnet_b3_dr.h5",
        ]

        loaded = False
        for p in weights_candidates:
            if p.exists():
                try:
                    logger.info("Loading trained model weights from: %s", p)
                    self.model = tf.keras.models.load_model(str(p), compile=False)
                    loaded = True
                    break
                except Exception as e:
                    logger.warning("Could not load weights from %s: %s", p, e)

        if not loaded:
            logger.info("No saved checkpoint found. Instantiating pretrained EfficientNetB3 architecture...")
            self.model = build_efficientnet_b3(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=5)

        self.gradcam = DRGradCAM(model=self.model)
        logger.info("Deep learning model and Grad-CAM successfully ready for inference.")

    def analyze_fundus_image(
        self,
        image_path: str,
        patient_id: str = "ANON-001",
        eye: str = "Right Eye (OD)",
    ) -> Dict[str, Any]:
        """Complete clinical pipeline: quality -> preprocess -> predict -> explain -> CDS -> RAG."""
        # 1. Quality Assessment
        quality_report = self.quality_assessor.get_quality_report(image_path)

        # 2. Preprocessing
        preprocessed = self.preprocessor.preprocess_single(image_path, apply_ben_graham=True)
        if preprocessed is None:
            raise ValueError(f"Unable to preprocess retinal fundus image: {image_path}")

        # 3. Model Prediction
        img_tensor = np.expand_dims(preprocessed, axis=0)
        raw_probs = self.model.predict(img_tensor, verbose=0)[0]
        pred_class = int(np.argmax(raw_probs))
        confidence = float(raw_probs[pred_class])
        probs_list = [round(float(p), 4) for p in raw_probs]

        # 4. Grad-CAM Explanation
        xai_res = self.gradcam.explain_prediction(
            image_path=image_path,
            preprocessed_img=preprocessed,
            save_figure=True,
        )

        # 5. Clinical Decision Support
        cds_res = self.decision_engine.evaluate(
            predicted_grade=pred_class,
            confidence=confidence,
            image_quality_report=quality_report,
            class_probabilities=probs_list,
        )

        # 6. RAG Evidence Retrieval
        retrieved_evidence = self.knowledge_retriever.retrieve_for_grade(pred_class, top_k=2)

        # 7. Narrative Report Generation
        narrative = self.report_generator.generate_report(
            patient_id=patient_id,
            predicted_grade=pred_class,
            confidence=confidence,
            probabilities=probs_list,
            retrieved_evidence=retrieved_evidence,
            image_quality_report=quality_report,
            clinical_decision=cds_res,
            eye=eye,
        )

        # 8. Compile PDF Report
        pdf_path = self.pdf_generator.generate_report(
            patient_id=patient_id,
            predicted_grade=pred_class,
            confidence=confidence,
            probabilities=probs_list,
            image_path=image_path,
            gradcam_path=xai_res.get("gradcam_image_path"),
            image_quality_report=quality_report,
            clinical_decision=cds_res,
            eye=eye,
        )

        # 9. Record in SQLite Database
        self.database.add_patient(patient_id=patient_id)
        screening_id = self.database.add_screening(
            patient_id=patient_id,
            image_path=image_path,
            eye=eye,
            predicted_grade=pred_class,
            confidence=confidence,
            probabilities={i: probs_list[i] for i in range(len(probs_list))},
            image_quality=quality_report.get("overall_quality", "acceptable"),
            gradcam_path=xai_res.get("gradcam_image_path"),
            report_path=pdf_path,
        )

        grade_label = DR_CLASSES.get(pred_class, f"Grade {pred_class}")
        return {
            "screening_id": screening_id,
            "patient_id": patient_id,
            "eye": eye,
            "predicted_grade": pred_class,
            "predicted_grade_label": grade_label,
            "predicted_class": grade_label,
            "confidence": round(confidence, 4),
            "class_probabilities": probs_list,
            "image_quality": quality_report,
            "gradcam": xai_res,
            "clinical_decision_support": cds_res,
            "retrieved_guidelines": retrieved_evidence,
            "clinical_narrative": narrative,
            "pdf_report_path": pdf_path,
        }
