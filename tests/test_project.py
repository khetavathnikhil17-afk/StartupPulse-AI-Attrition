# ══════════════════════════════════════════════════════════════════════════════
# StartupPulse AI — Tests
# ══════════════════════════════════════════════════════════════════════════════

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def processed_dir():
    return ROOT / "data" / "processed"


@pytest.fixture
def model_dir():
    return ROOT / "models" / "startuppulse_v1"


@pytest.fixture
def results_dir():
    return ROOT / "reports" / "results"


@pytest.fixture
def train_df(processed_dir):
    return pd.read_csv(processed_dir / "train.csv")


@pytest.fixture
def val_df(processed_dir):
    return pd.read_csv(processed_dir / "validation.csv")


@pytest.fixture
def test_df(processed_dir):
    return pd.read_csv(processed_dir / "test.csv")


@pytest.fixture
def trained_model(model_dir):
    from tensorflow import keras
    return keras.models.load_model(model_dir / "attrition_model.keras")


# ---------------------------------------------------------------------------
# Data Tests
# ---------------------------------------------------------------------------
class TestData:
    def test_train_exists(self, processed_dir):
        assert (processed_dir / "train.csv").exists()

    def test_val_exists(self, processed_dir):
        assert (processed_dir / "validation.csv").exists()

    def test_test_exists(self, processed_dir):
        assert (processed_dir / "test.csv").exists()

    def test_no_nulls(self, train_df, val_df, test_df):
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            assert df.isnull().sum().sum() == 0, f"Nulls found in {name}"

    def test_attrition_column_exists(self, train_df, val_df, test_df):
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            assert "Attrition" in df.columns, f"Attrition missing in {name}"

    def test_binary_labels(self, train_df, val_df, test_df):
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            unique = set(df["Attrition"].unique())
            assert unique <= {0, 1, 0.0, 1.0}, f"Non-binary labels in {name}: {unique}"

    def test_feature_count(self, train_df):
        assert train_df.shape[1] == 31, f"Expected 31 columns, got {train_df.shape[1]}"

    def test_split_sizes(self, train_df, val_df, test_df):
        assert len(train_df) > len(val_df)
        assert len(train_df) > len(test_df)
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_positive_class_ratio(self, train_df):
        ratio = train_df["Attrition"].mean()
        assert 0.05 < ratio < 0.5, f"Unexpected positive ratio: {ratio}"


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------
class TestModel:
    def test_model_file_exists(self, model_dir):
        assert (model_dir / "attrition_model.keras").exists()

    def test_scaler_exists(self, model_dir):
        assert (model_dir / "scaler.pkl").exists()

    def test_label_encoders_exists(self, model_dir):
        assert (model_dir / "label_encoders.pkl").exists()

    def test_model_loads(self, trained_model):
        assert trained_model is not None

    def test_model_input_shape(self, trained_model):
        assert trained_model.input_shape == (None, 30)

    def test_model_output_shape(self, trained_model):
        assert trained_model.output_shape == (None, 1)

    def test_model_predict_shape(self, trained_model):
        X = np.random.randn(5, 30).astype(np.float32)
        pred = trained_model.predict(X, verbose=0)
        assert pred.shape == (5, 1)

    def test_model_output_range(self, trained_model):
        X = np.random.randn(10, 30).astype(np.float32)
        pred = trained_model.predict(X, verbose=0).flatten()
        assert all(0 <= p <= 1 for p in pred), "Predictions out of [0,1] range"


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_metrics_file_exists(self, results_dir):
        assert (results_dir / "metrics.json").exists()

    def test_metrics_valid(self, results_dir):
        with open(results_dir / "metrics.json") as f:
            metrics = json.load(f)

        required_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
        assert required_keys <= set(metrics.keys()), f"Missing keys: {required_keys - set(metrics.keys())}"

        for key in required_keys:
            assert 0 <= metrics[key] <= 1, f"{key}={metrics[key]} out of [0,1]"

    def test_metrics_accuracy_above_random(self, results_dir):
        with open(results_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["accuracy"] > 0.5, "Accuracy below random baseline"

    def test_metrics_roc_auc_above_random(self, results_dir):
        with open(results_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["roc_auc"] > 0.5, "ROC-AUC below random baseline"

    def test_optimal_threshold_exists(self, results_dir):
        with open(results_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "optimal_threshold" in metrics
        assert 0 < metrics["optimal_threshold"] < 1


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_import(self):
        from src.data.preprocessing import run_pipeline
        assert callable(run_pipeline)

    def test_train_has_no_object_dtypes(self, train_df):
        # After preprocessing, all columns should be numeric
        for col in train_df.columns:
            assert train_df[col].dtype in [np.float64, np.int64, np.float32, np.int32], \
                f"Column {col} has dtype {train_df[col].dtype}"


# ---------------------------------------------------------------------------
# Model Training Tests
# ---------------------------------------------------------------------------
class TestTraining:
    def test_import_build_model(self):
        from src.model.train import build_model
        assert callable(build_model)

    def test_build_model_returns_compiled(self):
        from src.model.train import build_model
        model = build_model(30)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_train_v2_import(self):
        from src.model.train_v2 import apply_smote, find_optimal_threshold
        assert callable(apply_smote)
        assert callable(find_optimal_threshold)

    def test_smote_increases_minority(self):
        from src.model.train_v2 import apply_smote
        X = np.random.randn(100, 30).astype(np.float32)
        y = np.array([0] * 90 + [1] * 10, dtype=np.float32)
        X_res, y_res = apply_smote(X, y, sampling_strategy=0.8)
        assert len(y_res) > len(y)
        assert int(y_res.sum()) > 10


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------
class TestPrediction:
    def test_predict_module_importable(self):
        from src.model.predict import predict_attrition
        assert callable(predict_attrition)


# ---------------------------------------------------------------------------
# EDA Tests
# ---------------------------------------------------------------------------
class TestEDA:
    def test_eda_module_importable(self):
        from src.visualization.eda import run_eda
        assert callable(run_eda)


# ---------------------------------------------------------------------------
# SHAP Tests
# ---------------------------------------------------------------------------
class TestSHAP:
    def test_shap_module_importable(self):
        from src.explainability.shap_explainer import compute_shap_values
        assert callable(compute_shap_values)


# ---------------------------------------------------------------------------
# Dashboard Tests
# ---------------------------------------------------------------------------
class TestDashboard:
    def test_app_file_exists(self):
        assert (ROOT / "dashboard" / "app.py").exists()

    def test_pages_exist(self):
        pages_dir = ROOT / "dashboard" / "_pages"
        assert pages_dir.exists()
        expected = ["home.py", "predict.py", "analytics.py", "explainability.py", "reports.py", "about.py"]
        for page in expected:
            assert (pages_dir / page).exists(), f"Missing page: {page}"

    def test_components_exist(self):
        assert (ROOT / "dashboard" / "components" / "reusable_widgets.py").exists()

    def test_streamlit_config_exists(self):
        assert (ROOT / ".streamlit" / "config.toml").exists()
