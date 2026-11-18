"""
Classification module for Diabetic Retinopathy Detection.
Contains architectures (EfficientNetB3, ResNet50, DenseNet121),
custom loss functions (Focal Loss, Class-Weighted Crossentropy),
evaluation metrics (QWK, Sensitivity, Specificity), and the ModelTrainer pipeline.
"""

from .efficientnet_model import build_efficientnet_b3
from .resnet_model import build_resnet50
from .densenet_model import build_densenet121
from .loss_functions import get_loss_function, FocalLoss, ClassWeightedCrossEntropy
from .metrics import calculate_clinical_metrics, compute_quadratic_weighted_kappa
from .trainer import DRModelTrainer

__all__ = [
    "build_efficientnet_b3",
    "build_resnet50",
    "build_densenet121",
    "get_loss_function",
    "FocalLoss",
    "ClassWeightedCrossEntropy",
    "calculate_clinical_metrics",
    "compute_quadratic_weighted_kappa",
    "DRModelTrainer",
]
