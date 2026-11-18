"""
Explainability (XAI) module for Diabetic Retinopathy Detection.
Provides Grad-CAM class activation mapping, heatmaps, and overlay visualizers.
"""

from .gradcam import DRGradCAM

__all__ = ["DRGradCAM"]
