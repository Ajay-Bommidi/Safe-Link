import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Logging Setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("evaluation.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ✅ Paths & Configs
DATA_DIR = "../data/final"
MODEL_DIR = "../models/"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model_optimized.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ✅ Load Data Efficiently
def load_data():
    """Loads and prepares test data."""
    logger.info("📂 Loading evaluation data...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "fina.csv"), usecols=lambda col: col != "url")
        X = df.drop(columns=["label"]).values.astype(np.float32)
        y = df["label"].values.astype(np.int8)
        logger.info(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        raise

# ✅ Feature Scaling
def scale_data(X):
    """Applies Standard Scaling using saved scaler."""
    logger.info("📏 Scaling data...")
    try:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
        logger.info(f"✅ Data Scaled. Shape: {X_scaled.shape}")
        return X_scaled
    except Exception as e:
        logger.error(f"❌ Scaling failed: {str(e)}")
        raise

# ✅ Load Model (Fixed)
def load_model():
    """Loads the trained model correctly."""
    logger.info("📥 Loading model...")
    try:
        model = joblib.load(MODEL_PATH)  # No tuple unpacking
        threshold = 0.5  # Default threshold
        logger.info("✅ Model Loaded Successfully.")
        return model, threshold
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise

# ✅ Evaluate Model
def evaluate_model(model, threshold, X_test, y_test):
    """Evaluates model performance."""
    logger.info("📊 Evaluating Model...")
    try:
        # Get predicted probabilities
        proba = model.predict_proba(X_test)[:, 1]
        
        # Apply threshold for classification
        y_pred = (proba >= threshold).astype(int)
        
        # Compute Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, proba)
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
        plt.close()
        
        logger.info(f"📈 Metrics: {metrics}")
        logger.info(f"📄 Classification Report:\n{report}")
        return metrics
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise

# ✅ Main Execution
def main():
    """Runs the evaluation pipeline."""
    try:
        start_time = time.time()

        # Load test data & model
        X, y = load_data()
        X_scaled = scale_data(X)
        model, threshold = load_model()

        # Evaluate Model
        metrics = evaluate_model(model, threshold, X_scaled, y)
        
        logger.info(f"✅ Evaluation Completed in {time.time() - start_time:.2f}s")
        return metrics
    except Exception as e:
        logger.error(f"❌ Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
