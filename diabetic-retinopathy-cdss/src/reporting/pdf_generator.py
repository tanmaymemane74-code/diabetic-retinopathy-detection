"""
PDF Screening Report Generator for Diabetic Retinopathy Clinical Decision Support.
Uses ReportLab to compile publication-grade, audit-ready clinical screening summaries
with original fundus images, Grad-CAM heatmaps, probability tables, and clinical guidelines.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.config import DR_CLASSES, DR_DESCRIPTIONS, REPORTS_DIR

logger = logging.getLogger("dr_cdss.pdf")


class DRPDFReportGenerator:
    """Generates standardized, professional clinical screening PDF reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or REPORTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define healthcare-themed typography and colors."""
        self.styles.add(
            ParagraphStyle(
                name="ClinicalHeader",
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#1A365D"),  # Deep Navy Blue
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ClinicalSubheader",
                fontName="Helvetica",
                fontSize=11,
                leading=14,
                textColor=colors.HexColor("#4A5568"),  # Slate Grey
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                fontName="Helvetica-Bold",
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#2B6CB0"),  # Medical Blue
                spaceBefore=8,
                spaceAfter=4,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ClinicalBody",
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#2D3748"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ClinicalBold",
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#1A202C"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="UrgencyAlert",
                fontName="Helvetica-Bold",
                fontSize=10,
                leading=13,
                textColor=colors.HexColor("#C53030"),  # Crimson
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="DisclaimerText",
                fontName="Helvetica-Oblique",
                fontSize=7.5,
                leading=10,
                textColor=colors.HexColor("#718096"),
            )
        )

    def generate_report(
        self,
        patient_id: str,
        predicted_grade: int,
        confidence: float,
        probabilities: List[float],
        image_path: str,
        gradcam_path: Optional[str] = None,
        image_quality_report: Optional[Dict[str, Any]] = None,
        clinical_decision: Optional[Dict[str, Any]] = None,
        eye: str = "Right Eye (OD)",
        screening_date: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> str:
        """Compile and generate the PDF report.

        Returns:
            Absolute path to generated PDF file.
        """
        if screening_date is None:
            screening_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if output_filename is None:
            safe_pid = patient_id.replace(" ", "_").replace("/", "-")
            output_filename = f"DR_Screening_Report_{safe_pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        story = []
        grade_label = DR_CLASSES.get(predicted_grade, f"Grade {predicted_grade}")

        # -------------------------------------------------------------
        # 1. Header & Title Banner
        # -------------------------------------------------------------
        title_p = Paragraph("DIABETIC RETINOPATHY CLINICAL DECISION SUPPORT SYSTEM", self.styles["ClinicalHeader"])
        subtitle_p = Paragraph("AI-Assisted Preliminary Retinal Screening & Specialist Referral Report", self.styles["ClinicalSubheader"])
        story.append(title_p)
        story.append(subtitle_p)
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

        # -------------------------------------------------------------
        # 2. Patient & Screening Demographics Table
        # -------------------------------------------------------------
        demo_data = [
            [
                Paragraph("<b>Patient Identifier:</b>", self.styles["ClinicalBody"]),
                Paragraph(patient_id, self.styles["ClinicalBold"]),
                Paragraph("<b>Screening Date:</b>", self.styles["ClinicalBody"]),
                Paragraph(screening_date, self.styles["ClinicalBold"]),
            ],
            [
                Paragraph("<b>Eye Examined:</b>", self.styles["ClinicalBody"]),
                Paragraph(eye, self.styles["ClinicalBold"]),
                Paragraph("<b>Image Quality:</b>", self.styles["ClinicalBody"]),
                Paragraph(
                    image_quality_report.get("overall_quality", "Acceptable").upper() if image_quality_report else "ACCEPTABLE",
                    self.styles["ClinicalBold"],
                ),
            ],
            [
                Paragraph("<b>AI Model:</b>", self.styles["ClinicalBody"]),
                Paragraph("EfficientNetB3-XAI (ImageNet)", self.styles["ClinicalBold"]),
                Paragraph("<b>Screening Status:</b>", self.styles["ClinicalBody"]),
                Paragraph("Preliminary Screening (Requires Review)", self.styles["ClinicalBold"]),
            ],
        ]
        demo_table = Table(demo_data, colWidths=[110, 155, 110, 155])
        demo_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(demo_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # 3. AI Prediction & Probability Distribution
        # -------------------------------------------------------------
        story.append(Paragraph("AI SCREENING ASSESSMENT", self.styles["SectionHeading"]))

        # Urgency color based on grade
        urgency_bg = "#EBF8FF"  # Blue for 0
        urgency_border = "#3182CE"
        if predicted_grade == 1:
            urgency_bg = "#FEFCBF"  # Light yellow
            urgency_border = "#D69E2E"
        elif predicted_grade >= 2:
            urgency_bg = "#FED7D7"  # Red
            urgency_border = "#E53E3E"

        pred_banner = [
            [
                Paragraph(f"<b>Predicted Severity:</b> {grade_label.upper()}", self.styles["ClinicalHeader"]),
                Paragraph(f"<b>Confidence:</b> {confidence * 100:.1f}%", self.styles["ClinicalHeader"]),
            ]
        ]
        pred_table = Table(pred_banner, colWidths=[350, 180])
        pred_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(urgency_bg)),
                ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(urgency_border)),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ])
        )
        story.append(pred_table)
        story.append(Spacer(1, 8))

        # Probability Breakdown Table
        prob_header = ["Grade", "Category Description", "Model Probability", "Bar Indicator"]
        prob_rows = [prob_header]
        for g_idx, g_name in DR_CLASSES.items():
            prob_val = probabilities[g_idx] if g_idx < len(probabilities) else 0.0
            pct_str = f"{prob_val * 100:.1f}%"
            bar_len = int(prob_val * 25)
            bar_str = "█" * bar_len
            prob_rows.append([f"Grade {g_idx}", g_name, pct_str, bar_str])

        prob_table = Table(prob_rows, colWidths=[65, 230, 105, 130])
        prob_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("BACKGROUND", (0, predicted_grade + 1), (-1, predicted_grade + 1), colors.HexColor("#EBF8FF")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(prob_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # 4. Retinal Images (Original & Grad-CAM)
        # -------------------------------------------------------------
        story.append(Paragraph("EXPLAINABLE AI (GRAD-CAM) VISUAL EVIDENCE", self.styles["SectionHeading"]))

        image_cells = []
        if os.path.exists(image_path):
            img_obj = Image(image_path, width=220, height=220)
            image_cells.append([img_obj, Paragraph("<b>Original Retinal Photograph</b>", self.styles["ClinicalBody"])])
        else:
            image_cells.append([Paragraph("[Original Image Unavailable]", self.styles["ClinicalBody"]), ""])

        if gradcam_path and os.path.exists(gradcam_path):
            gcam_obj = Image(gradcam_path, width=280, height=180)
            img_table_data = [[
                Image(image_path, width=220, height=180) if os.path.exists(image_path) else Paragraph("N/A", self.styles["ClinicalBody"]),
                gcam_obj,
            ], [
                Paragraph("<b>Figure A:</b> Standardized Retinal Fundus", self.styles["ClinicalBody"]),
                Paragraph("<b>Figure B:</b> Grad-CAM Attention Heatmap", self.styles["ClinicalBody"]),
            ]]
            img_table = Table(img_table_data, colWidths=[240, 290])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(img_table)
        story.append(Spacer(1, 8))

        # -------------------------------------------------------------
        # 5. Clinical Decision Support Recommendations
        # -------------------------------------------------------------
        story.append(Paragraph("CLINICAL DECISION SUPPORT & REFERRAL CONSIDERATIONS", self.styles["SectionHeading"]))
        if clinical_decision:
            action_text = clinical_decision.get("suggested_action", "")
            timeframe_text = clinical_decision.get("suggested_timeframe", "")
            urgency_text = clinical_decision.get("urgency_level", "")
            citations = clinical_decision.get("guideline_citations", [])

            cds_data = [
                [Paragraph("<b>Referral Urgency:</b>", self.styles["ClinicalBold"]), Paragraph(urgency_text, self.styles["UrgencyAlert"])],
                [Paragraph("<b>Recommended Action:</b>", self.styles["ClinicalBold"]), Paragraph(action_text, self.styles["ClinicalBody"])],
                [Paragraph("<b>Suggested Timeframe:</b>", self.styles["ClinicalBold"]), Paragraph(timeframe_text, self.styles["ClinicalBold"])],
                [Paragraph("<b>Guideline Citations:</b>", self.styles["ClinicalBold"]), Paragraph("<br/>".join(f"• {c}" for c in citations), self.styles["ClinicalBody"])],
            ]
            cds_table = Table(cds_data, colWidths=[130, 400])
            cds_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(cds_table)

        story.append(Spacer(1, 12))

        # -------------------------------------------------------------
        # 6. Clinician Sign-off & Medical Legal Disclaimer
        # -------------------------------------------------------------
        sign_data = [
            [
                Paragraph("<b>Reviewing Clinician Name:</b> ___________________________", self.styles["ClinicalBody"]),
                Paragraph("<b>Clinician Signature:</b> ___________________________", self.styles["ClinicalBody"]),
                Paragraph("<b>Date:</b> _________", self.styles["ClinicalBody"]),
            ]
        ]
        sign_table = Table(sign_data, colWidths=[200, 230, 100])
        sign_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 6)]))
        story.append(sign_table)
        story.append(Spacer(1, 10))

        disclaimer_text = (
            "<b>IMPORTANT CLINICAL NOTICE & DISCLAIMER:</b> This report is generated by an Artificial Intelligence "
            "assisted decision support prototype designed for research and preliminary triage. It does NOT constitute "
            "a formal ophthalmological diagnosis, medical prescription, or independent treatment directive. All visual "
            "evaluations, severity ratings, and referral urgency classifications must be reviewed, validated, and signed "
            "off by a licensed ophthalmologist, optometrist, or qualified medical specialist."
        )
        story.append(Paragraph(disclaimer_text, self.styles["DisclaimerText"]))

        # Build document
        doc.build(story)
        logger.info("Clinical PDF report successfully generated at: %s", pdf_path)
        return str(pdf_path)
