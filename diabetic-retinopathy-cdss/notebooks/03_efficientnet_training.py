"""
03 - EfficientNetB3 Training and Fine-Tuning Script.
Trains the primary EfficientNetB3 architecture on Diabetic Retinopathy data.
Uses two-phase transfer learning:
  Phase 1: Feature extraction (frozen backbone) with Adam lr=1e-3
  Phase 2: Fine-tuning (top 50 backbone layers unfrozen) with Adam lr=1e-5
Evaluates with Focal Loss and outputs training curves & best weights checkpoint.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.trainer import DRModelTrainer
from src.utils.config import (
    BATCH_SIZE,
    CSV_PATH,
    DATASET_DIR,
    DR_CLASSES,
    IMAGE_SIZE,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.utils.database import DRDatabase
from src.utils.dataset_loader import DRDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")
logger = logging.getLogger("dr_cdss.train_efficientnet")


def main():
    print("=" * 65)
    print("  PHASE 3: EFFICIENTNET-B3 PRIMARY MODEL TRAINING")
    print("=" * 65)

    # 1. Load dataset & prepare splits
    loader = DRDatasetLoader(data_dir=DATASET_DIR / "raw" / "train_images", csv_path=CSV_PATH)
    train_df, val_df, test_df = loader.create_splits(test_size=0.15, val_size=0.15, random_state=42)
    class_weights = loader.get_class_weights(train_df)

    print(f"\nDataset Splits:")
    print(f"  • Training   : {len(train_df)} images")
    print(f"  • Validation : {len(val_df)} images")
    print(f"  • Test       : {len(test_df)} images")
    print(f"  • Class weights: {class_weights}")

    # 2. Initialize Trainer
    trainer = DRModelTrainer(
        model_name="efficientnet_b3",
        input_size=IMAGE_SIZE,
        num_classes=5,
        loss_type="focal",
        batch_size=min(BATCH_SIZE, len(train_df)),
    )

    checkpoint_path = Path(MODELS_DIR) / "efficientnet_b3_best.keras"

    # 3. Execute 2-phase training
    history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs_phase1=2,  # Quick demo/verification epochs
        epochs_phase2=2,
        lr_phase1=1e-3,
        lr_phase2=1e-5,
        class_weights=class_weights,
        save_path=str(checkpoint_path),
    )

    # 4. Evaluate on Test Set
    metrics = trainer.evaluate(test_df=test_df)

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS (EFFICIENTNET-B3)")
    print("=" * 65)
    print(f"  Accuracy                 : {metrics['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa : {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Macro F1-Score           : {metrics['f1_macro']:.4f}")
    print(f"  Referable DR Sensitivity : {metrics['referable_dr_screening']['sensitivity']:.4f}")
    print(f"  Referable DR Specificity : {metrics['referable_dr_screening']['specificity']:.4f}")
    print(f"  Referable DR ROC-AUC     : {metrics['referable_dr_screening']['roc_auc']:.4f}")
    print("=" * 65)

    # 5. Save model metadata to database
    db = DRDatabase()
    db.add_model_metadata(
        model_name="EfficientNetB3",
        architecture="EfficientNetB3 + GlobalAvgPool + Dense(256) + Dropout(0.4)",
        accuracy=metrics["accuracy"],
        f1_score=metrics["f1_macro"],
        epochs=4,
        notes=f"QWK={metrics['quadratic_weighted_kappa']:.4f}, Referable DR Sens={metrics['referable_dr_screening']['sensitivity']:.4f}",
    )
    print("Saved model performance to database (dr_cdss.db).")


if __name__ == "__main__":
    main()
