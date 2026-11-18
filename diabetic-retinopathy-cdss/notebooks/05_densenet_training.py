"""
05 - DenseNet121 Comparative Model Training Script.
Trains the DenseNet121 architecture on Diabetic Retinopathy data.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.trainer import DRModelTrainer
from src.utils.config import (
    BATCH_SIZE,
    CSV_PATH,
    DATASET_DIR,
    IMAGE_SIZE,
    MODELS_DIR,
)
from src.utils.database import DRDatabase
from src.utils.dataset_loader import DRDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")


def main():
    print("=" * 65)
    print("  PHASE 4: DENSENET-121 COMPARATIVE MODEL TRAINING")
    print("=" * 65)

    loader = DRDatasetLoader(data_dir=DATASET_DIR / "raw" / "train_images", csv_path=CSV_PATH)
    train_df, val_df, test_df = loader.create_splits(test_size=0.15, val_size=0.15, random_state=42)
    class_weights = loader.get_class_weights(train_df)

    trainer = DRModelTrainer(
        model_name="densenet121",
        input_size=IMAGE_SIZE,
        num_classes=5,
        loss_type="focal",
        batch_size=min(BATCH_SIZE, len(train_df)),
    )

    checkpoint_path = Path(MODELS_DIR) / "densenet121_best.keras"

    history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs_phase1=2,
        epochs_phase2=0,
        lr_phase1=1e-3,
        class_weights=class_weights,
        save_path=str(checkpoint_path),
    )

    metrics = trainer.evaluate(test_df=test_df)

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS (DENSENET-121)")
    print("=" * 65)
    print(f"  Accuracy                 : {metrics['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa : {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Macro F1-Score           : {metrics['f1_macro']:.4f}")
    print(f"  Referable DR Sensitivity : {metrics['referable_dr_screening']['sensitivity']:.4f}")
    print(f"  Referable DR Specificity : {metrics['referable_dr_screening']['specificity']:.4f}")
    print("=" * 65)

    db = DRDatabase()
    db.add_model_metadata(
        model_name="DenseNet121",
        architecture="DenseNet121 + GlobalAvgPool + Dense(256) + Dropout(0.4)",
        accuracy=metrics["accuracy"],
        f1_score=metrics["f1_macro"],
        epochs=2,
        notes=f"QWK={metrics['quadratic_weighted_kappa']:.4f}, Referable DR Sens={metrics['referable_dr_screening']['sensitivity']:.4f}",
    )


if __name__ == "__main__":
    main()
