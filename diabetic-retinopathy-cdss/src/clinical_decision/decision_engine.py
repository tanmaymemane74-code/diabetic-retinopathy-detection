"""
Clinical Decision Support Engine (CDSS) for Diabetic Retinopathy Screening.
Rule-based expert system grounded in American Academy of Ophthalmology (AAO)
and American Diabetes Association (ADA) clinical guidelines.
Provides transparent, auditable referral considerations and urgency stratification.
"""

from typing import Any, Dict, List, Optional
from src.utils.config import DR_CLASSES, DR_DESCRIPTIONS


class DRClinicalDecisionEngine:
    """Evaluates prediction probabilities, image quality, and model confidence

    to determine evidence-based referral urgency and recommended next actions.
    """

    def __init__(self, confidence_threshold: float = 0.65, uncertainty_threshold: float = 1.2):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate(
        self,
        predicted_grade: int,
        confidence: float,
        image_quality_report: Dict[str, Any],
        class_probabilities: Optional[List[float]] = None,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate clinical decision support evaluation.

        Args:
            predicted_grade: Integer DR grade (0 to 4)
            confidence: Float confidence score (0.0 to 1.0)
            image_quality_report: Quality report from ImageQualityAssessor
            class_probabilities: 5-element list of softmax probabilities
            patient_metadata: Optional dict with diabetes_type, duration_years, hbA1c

        Returns:
            Structured clinical decision report with referral urgency, suggested
            actions, evidence rationale, and guideline references.
        """
        is_gradable = image_quality_report.get("is_gradable", True)
        overall_quality = image_quality_report.get("overall_quality", "good")

        # -------------------------------------------------------------
        # 1. Handle Ungradable Photographs (Clinical Safety Rule)
        # -------------------------------------------------------------
        if not is_gradable or overall_quality == "ungradable":
            return {
                "screening_classification": "Ungradable Retinal Examination",
                "referable_status": "Indeterminate - Requires In-Person Examination",
                "urgency_level": "Routine / Timely Rescreen",
                "suggested_action": (
                    "Photograph is ungradable due to media opacity, poor illumination, "
                    "or severe blur. Mandatory clinical safety protocol: Do NOT assume absence "
                    "of disease. Schedule in-person dilated slit-lamp fundus examination."
                ),
                "suggested_timeframe": "Within 4 to 8 weeks",
                "evidence_rationale": "AAO PPP specifies that ungradable photographs require in-person evaluation to prevent missed sight-threatening pathology.",
                "guideline_citations": [
                    "AAO Preferred Practice Pattern: Diabetic Retinopathy (2023) - Section 3",
                    "ADA Standards of Care in Diabetes (2024) - Telemedicine & Retinal Imaging",
                ],
                "requires_urgent_specialist": False,
                "clinician_review_mandatory": True,
                "safety_alert": "UNGRADABLE_IMAGE",
            }

        # Check for model uncertainty / ambiguous predictions
        is_low_confidence = confidence < self.confidence_threshold

        # -------------------------------------------------------------
        # 2. Rule-Based Urgency and Referral Stratification
        # -------------------------------------------------------------
        grade_label = DR_CLASSES.get(predicted_grade, f"Grade {predicted_grade}")

        if predicted_grade == 0:
            # Grade 0: No DR
            screening_class = "Non-Referable DR"
            urgency = "Routine Annual Surveillance"
            action = (
                "Continue annual dilated eye examination. Reinforce systemic diabetes management: "
                "target HbA1c < 7.0%, blood pressure < 130/80 mmHg, and lipid optimization."
            )
            timeframe = "12 months"
            ref_status = "Non-Referable"
            urgent_spec = False
            guidelines = [
                "AAO PPP (2023): Annual dilated retinal exam for patients with no apparent retinopathy.",
                "ADA Standards of Care (2024): Routine annual screening; optimize glycemic and blood pressure targets.",
            ]

        elif predicted_grade == 1:
            # Grade 1: Mild NPDR
            screening_class = "Non-Referable DR (Early Microvascular Changes)"
            urgency = "Closer Surveillance"
            action = (
                "Isolated microaneurysms detected. Repeat dilated eye examination in 6 to 12 months. "
                "Notify primary care physician to review glycemic control and blood pressure management."
            )
            timeframe = "6 to 12 months"
            ref_status = "Non-Referable (Monitor)"
            urgent_spec = False
            guidelines = [
                "AAO PPP (2023): Mild NPDR requires follow-up within 6-12 months without immediate laser or injection therapy.",
                "ADA Standards of Care (2024): Re-evaluate at least annually or sooner if systemic control is sub-optimal.",
            ]

        elif predicted_grade == 2:
            # Grade 2: Moderate NPDR
            screening_class = "Referable Diabetic Retinopathy"
            urgency = "Semi-Urgent Specialist Assessment"
            action = (
                "Multiple microaneurysms, hemorrhages, and/or exudates detected. "
                "Refer to general ophthalmologist or retina specialist. Assess for concurrent diabetic macular edema (DME)."
            )
            timeframe = "Within 3 to 6 months"
            ref_status = "Referable"
            urgent_spec = False
            guidelines = [
                "AAO PPP (2023): Moderate NPDR referral indicated within 3 to 6 months; monitor for DME development.",
                "International Council of Ophthalmology (ICO) Guidelines: Referable diabetic retinopathy.",
            ]

        elif predicted_grade == 3:
            # Grade 3: Severe NPDR
            screening_class = "Referable Sight-Threatening DR"
            urgency = "Urgent Retina Specialist Evaluation"
            action = (
                "Severe nonproliferative retinopathy features detected (4-2-1 rule criteria). "
                "High risk (~50% at 1 year) of progression to proliferative DR. "
                "Urgent referral to a vitreoretinal specialist for evaluation of panretinal photocoagulation or anti-VEGF."
            )
            timeframe = "Within 2 to 4 weeks"
            ref_status = "Referable (High Risk)"
            urgent_spec = True
            guidelines = [
                "AAO PPP (2023): Severe NPDR requires prompt referral within 2-4 weeks due to imminent proliferative risk.",
                "ETDRS Report No. 12: High-risk characteristics warranting specialist intervention.",
            ]

        elif predicted_grade == 4:
            # Grade 4: Proliferative DR
            screening_class = "Sight-Threatening Proliferative DR (PDR)"
            urgency = "Immediate / Emergency Specialist Assessment"
            action = (
                "Neovascularization or preretinal hemorrhage suspected. Emergency specialist assessment indicated. "
                "Immediate treatment consideration (panretinal laser photocoagulation, anti-VEGF therapy, or vitrectomy) "
                "to prevent irreversible vision loss."
            )
            timeframe = "Within 24 to 48 hours (maximum 1 week)"
            ref_status = "Referable (Sight-Threatening)"
            urgent_spec = True
            guidelines = [
                "AAO PPP (2023): Proliferative DR requires immediate/urgent evaluation within 24-48 hours to 1 week.",
                "ADA Standards of Care (2024): Prompt referral to ophthalmologist knowledgeable in diabetic retinopathy management.",
            ]

        else:
            raise ValueError(f"Invalid predicted grade: {predicted_grade}")

        # Alert flag for low confidence
        safety_alert = None
        if is_low_confidence:
            safety_alert = f"LOW_CONFIDENCE ({confidence * 100:.1f}%)"
            action = (
                f"[CAUTION: Model confidence is low ({confidence * 100:.1f}%)]. "
                + action
                + " Clinician verification is required before acting on this automated assessment."
            )

        return {
            "predicted_grade": predicted_grade,
            "predicted_grade_label": grade_label,
            "confidence": round(confidence, 4),
            "screening_classification": screening_class,
            "referable_status": ref_status,
            "urgency_level": urgency,
            "suggested_action": action,
            "suggested_timeframe": timeframe,
            "evidence_rationale": DR_DESCRIPTIONS.get(predicted_grade, ""),
            "guideline_citations": guidelines,
            "requires_urgent_specialist": urgent_spec,
            "clinician_review_mandatory": True,
            "safety_alert": safety_alert,
        }

    def get_referral_recommendation(self, grade: int) -> Dict[str, Any]:
        """Convenience method returning referral recommendation for a given grade (0-4)."""
        res = self.evaluate(
            predicted_grade=grade,
            confidence=1.0,
            image_quality_report={"is_gradable": True, "overall_quality": "good"},
        )
        return {
            "grade": grade,
            "grade_label": res["predicted_grade_label"],
            "urgency": res["urgency_level"],
            "action": res["suggested_action"],
            "timeframe": res["suggested_timeframe"],
            "guidelines": res["guideline_citations"],
        }
