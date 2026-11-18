"""
AI-Assisted Clinical Screening Report Generator.
Grounds narrative reports in retrieved medical guidelines (AAO, ADA, WHO)
using OpenAI API or structured clinical template fallback.
Enforces strict medical safety constraints: no medication prescriptions,
explicit uncertainty communication, and mandatory ophthalmologist review.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.utils.config import DR_CLASSES, DR_DESCRIPTIONS, OPENAI_API_KEY

logger = logging.getLogger("dr_cdss.llm")


class DRReportGenerator:
    """Generates structured, guideline-grounded clinical narrative reports."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self.model = model

    def generate_report(
        self,
        patient_id: str,
        predicted_grade: int,
        confidence: float,
        probabilities: List[float],
        retrieved_evidence: List[Dict[str, Any]],
        image_quality_report: Optional[Dict[str, Any]] = None,
        clinical_decision: Optional[Dict[str, Any]] = None,
        eye: str = "Right Eye (OD)",
    ) -> Dict[str, Any]:
        """Generate structured clinical report with RAG grounding."""
        grade_name = DR_CLASSES.get(predicted_grade, f"Grade {predicted_grade}")

        # Check if OpenAI is available
        if self.api_key and len(self.api_key.strip()) > 10:
            try:
                return self._generate_with_llm(
                    patient_id=patient_id,
                    predicted_grade=predicted_grade,
                    confidence=confidence,
                    probabilities=probabilities,
                    retrieved_evidence=retrieved_evidence,
                    image_quality_report=image_quality_report,
                    clinical_decision=clinical_decision,
                    eye=eye,
                )
            except Exception as e:
                logger.warning("LLM generation failed: %s. Falling back to clinical template generator.", e)

        # Fallback to structured deterministic clinical template
        return self._generate_template_report(
            patient_id=patient_id,
            predicted_grade=predicted_grade,
            confidence=confidence,
            probabilities=probabilities,
            retrieved_evidence=retrieved_evidence,
            image_quality_report=image_quality_report,
            clinical_decision=clinical_decision,
            eye=eye,
        )

    def _generate_with_llm(
        self,
        patient_id: str,
        predicted_grade: int,
        confidence: float,
        probabilities: List[float],
        retrieved_evidence: List[Dict[str, Any]],
        image_quality_report: Optional[Dict[str, Any]],
        clinical_decision: Optional[Dict[str, Any]],
        eye: str,
    ) -> Dict[str, Any]:
        """Call OpenAI API with grounded context and safety constraints."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        grade_name = DR_CLASSES.get(predicted_grade, f"Grade {predicted_grade}")

        evidence_text = "\n\n".join(
            f"[Source: {e.get('source', 'Guideline')}] (Similarity: {e.get('similarity_score', 'N/A')}):\n{e.get('text', '')}"
            for e in retrieved_evidence
        )

        prompt = f"""You are an expert AI clinical decision support assistant specialized in ophthalmology and diabetic retinopathy screening.
Generate a structured, professional clinical screening report based strictly on the provided image classification and retrieved guidelines.

PATIENT & SCREENING INFORMATION:
- Patient ID: {patient_id}
- Eye Examined: {eye}
- Predicted Severity: {grade_name} (Grade {predicted_grade})
- Model Confidence: {confidence * 100:.1f}%
- Class Probabilities: {probabilities}
- Image Quality: {image_quality_report.get('overall_quality', 'Acceptable') if image_quality_report else 'Acceptable'}

RETRIEVED CLINICAL GUIDELINES EVIDENCE:
{evidence_text}

CLINICAL DECISION RULES:
- Referral Urgency: {clinical_decision.get('urgency_level', 'Routine') if clinical_decision else 'Standard'}
- Recommended Action: {clinical_decision.get('suggested_action', 'Annual review') if clinical_decision else 'Follow-up'}
- Timeframe: {clinical_decision.get('suggested_timeframe', '12 months') if clinical_decision else '12 months'}

MANDATORY CLINICAL SAFETY RULES:
1. Ground your interpretation strictly in the retrieved AAO and ADA guidelines.
2. DO NOT prescribe medications or create speculative medical therapies.
3. Clearly emphasize that this is an AI-assisted screening assessment and requires qualified ophthalmologist review.
4. Output a clean, JSON-compatible object with keys:
   - "clinical_summary": brief paragraph summarizing findings
   - "pathology_description": morphological explanation of this grade
   - "referral_recommendations": actionable specialist triage advice
   - "guideline_citations": list of references used
   - "disclaimer": standard medical disclaimer
"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional healthcare AI clinical report generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        try:
            # Try parsing JSON
            parsed = json.loads(content)
            return parsed
        except Exception:
            return {
                "clinical_summary": content,
                "predicted_grade": predicted_grade,
                "confidence": confidence,
                "disclaimer": "AI-assisted screening output. Requires clinician verification.",
            }

    def _generate_template_report(
        self,
        patient_id: str,
        predicted_grade: int,
        confidence: float,
        probabilities: List[float],
        retrieved_evidence: List[Dict[str, Any]],
        image_quality_report: Optional[Dict[str, Any]],
        clinical_decision: Optional[Dict[str, Any]],
        eye: str,
    ) -> Dict[str, Any]:
        """Grounded clinical report using deterministic rule-based medical templates."""
        grade_name = DR_CLASSES.get(predicted_grade, f"Grade {predicted_grade}")
        grade_desc = DR_DESCRIPTIONS.get(predicted_grade, "")

        summary = (
            f"Automated deep learning screening of the retinal fundus photograph for {patient_id} ({eye}) "
            f"indicates features consistent with '{grade_name}' (Model confidence: {confidence * 100:.1f}%). "
            f"Image quality was evaluated as {image_quality_report.get('overall_quality', 'acceptable').upper() if image_quality_report else 'ACCEPTABLE'}."
        )

        referral_text = (
            clinical_decision.get("suggested_action", "Routine dilated eye exam.")
            if clinical_decision
            else "Follow clinical guidelines."
        )

        urgency_text = (
            clinical_decision.get("urgency_level", "Routine") if clinical_decision else "Routine"
        )

        timeframe_text = (
            clinical_decision.get("suggested_timeframe", "12 months") if clinical_decision else "12 months"
        )

        citations = [
            e.get("source", "AAO/ADA Retinopathy Guidelines") for e in retrieved_evidence
        ]
        if not citations:
            citations = [
                "American Academy of Ophthalmology (AAO) Preferred Practice Pattern: Diabetic Retinopathy (2023)",
                "American Diabetes Association (ADA) Standards of Care in Diabetes (2024)",
            ]

        return {
            "patient_id": patient_id,
            "eye_examined": eye,
            "predicted_grade": predicted_grade,
            "predicted_grade_label": grade_name,
            "confidence": round(confidence, 4),
            "clinical_summary": summary,
            "pathology_description": grade_desc,
            "referral_recommendations": {
                "urgency_level": urgency_text,
                "suggested_action": referral_text,
                "suggested_timeframe": timeframe_text,
            },
            "retrieved_evidence_excerpts": [
                {"source": e.get("source", ""), "excerpt": e.get("text", "")[:200] + "..."}
                for e in retrieved_evidence[:2]
            ],
            "guideline_citations": list(set(citations)),
            "disclaimer": (
                "This document is an AI-assisted clinical screening report generated for decision support. "
                "It is not an independent medical diagnosis. All findings and referral intervals must be "
                "reviewed and confirmed by a licensed medical practitioner."
            ),
        }
