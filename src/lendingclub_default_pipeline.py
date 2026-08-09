from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import shutil
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier

    HAS_EBM = True
except Exception:
    ExplainableBoostingClassifier = None
    HAS_EBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    HAS_TORCH = False


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
            self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        self.lower_bounds_ = np.where(np.isfinite(self.lower_bounds_), self.lower_bounds_, 0.0)
        self.upper_bounds_ = np.where(np.isfinite(self.upper_bounds_), self.upper_bounds_, 0.0)
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def log1p_nonnegative(x):
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


if HAS_TORCH:

    class _FTTransformerNet(nn.Module):
        def __init__(
            self,
            n_features: int,
            d_token: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.feature_weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_token * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Sequential(
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, 1),
            )

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
            tokens = tokens + self.feature_bias.unsqueeze(0)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(self.norm(encoded[:, 0])).squeeze(-1)


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """A compact FT-Transformer-style classifier for preprocessed tabular features."""

    def __init__(
        self,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        max_epochs: int = 12,
        patience: int = 3,
        validation_fraction: float = 0.15,
        max_train_rows: int = 0,
        calibrate: bool = True,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.max_train_rows = max_train_rows
        self.calibrate = calibrate
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, x, y, sample_weight=None, validation_data=None):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for FTTransformerClassifier. "
                "Install torch or use --skip-ft-transformer."
            )
        rng = np.random.RandomState(self.random_state)
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        weights_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)

        if self.max_train_rows and len(y_arr) > self.max_train_rows:
            sample_idx, _ = train_test_split(
                np.arange(len(y_arr)),
                train_size=self.max_train_rows,
                random_state=self.random_state,
                stratify=y_arr,
            )
            x_arr = x_arr[sample_idx]
            y_arr = y_arr[sample_idx]
            if weights_arr is not None:
                weights_arr = weights_arr[sample_idx]

        idx = np.arange(len(y_arr))
        if validation_data is not None:
            train_idx = idx
            valid_x_arr = np.asarray(validation_data[0], dtype=np.float32)
            valid_y = np.asarray(validation_data[1], dtype=np.float32)
            self.validation_source_ = "external_validation_set"
        elif 0 < self.validation_fraction < 0.5 and len(np.unique(y_arr)) == 2:
            train_idx, valid_idx = train_test_split(
                idx,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_arr,
            )
            valid_x_arr = x_arr[valid_idx]
            valid_y = y_arr[valid_idx]
            self.validation_source_ = "internal_training_split"
        else:
            rng.shuffle(idx)
            split = max(1, int(len(idx) * 0.85))
            train_idx, valid_idx = idx[:split], idx[split:]
            valid_x_arr = x_arr[valid_idx]
            valid_y = y_arr[valid_idx]
            self.validation_source_ = "internal_training_split"

        n_heads = min(self.n_heads, self.d_token)
        while self.d_token % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.classes_ = np.array([0, 1])
        if self.device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(self.device)
        self.fit_train_rows_ = int(len(train_idx))
        self.training_device_ = str(self.device_)
        self.training_device_name_ = (
            torch.cuda.get_device_name(self.device_)
            if self.device_.type == "cuda"
            else "CPU"
        )
        self.model_ = _FTTransformerNet(
            n_features=x_arr.shape[1],
            d_token=self.d_token,
            n_heads=n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device_)

        train_x = torch.tensor(x_arr[train_idx], dtype=torch.float32)
        train_y = torch.tensor(y_arr[train_idx], dtype=torch.float32)
        if weights_arr is None:
            train_w = torch.ones_like(train_y)
        else:
            train_w = torch.tensor(weights_arr[train_idx], dtype=torch.float32)
        valid_x = torch.tensor(valid_x_arr, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(train_x, train_y, train_w),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device_.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_auc = -np.inf
        best_state = None
        stale_epochs = 0
        self.training_history_ = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_losses = []
            for batch_x, batch_y, batch_w in loader:
                batch_x = batch_x.to(self.device_, non_blocking=True)
                batch_y = batch_y.to(self.device_, non_blocking=True)
                batch_w = batch_w.to(self.device_, non_blocking=True)
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = (criterion(logits, batch_y) * batch_w).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            valid_scores = self._predict_scores_array(valid_x.numpy())
            try:
                valid_auc = roc_auc_score(valid_y, valid_scores)
            except Exception:
                valid_auc = -float(np.mean(epoch_losses))
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)),
                    "valid_auc": float(valid_auc),
                }
            )
            if self.verbose:
                print(
                    f"FT-Transformer epoch {epoch + 1}: "
                    f"loss={np.mean(epoch_losses):.4f}, valid_auc={valid_auc:.4f}",
                    flush=True,
                )
            if valid_auc > best_auc + 1e-5:
                best_auc = valid_auc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.epochs_trained_ = int(len(self.training_history_))
        self.best_validation_auc_ = float(best_auc)
        self.calibrator_ = None
        self.calibration_method_ = "none"
        if self.calibrate and len(valid_y) > 0 and len(np.unique(valid_y)) == 2:
            valid_logits = self._predict_logits_array(valid_x.numpy())
            calibrator = SklearnLogisticRegression(max_iter=1_000, solver="lbfgs")
            calibrator.fit(valid_logits.reshape(-1, 1), valid_y.astype(int))
            if float(calibrator.coef_[0, 0]) > 0:
                self.calibrator_ = calibrator
                self.calibration_method_ = "validation_sigmoid_on_logits"
        return self

    def _predict_logits_array(self, x):
        x_arr = np.asarray(x, dtype=np.float32)
        self.model_.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(x_arr), self.batch_size):
                batch = torch.tensor(
                    x_arr[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device_,
                )
                logits.append(self.model_(batch).cpu().numpy())
        return np.concatenate(logits)

    def _predict_scores_array(self, x):
        logits = self._predict_logits_array(x)
        if getattr(self, "calibrator_", None) is not None:
            return self.calibrator_.predict_proba(logits.reshape(-1, 1))[:, 1]
        return 1 / (1 + np.exp(-logits))

    def predict_proba(self, x):
        scores = self._predict_scores_array(x)
        return np.column_stack([1 - scores, scores])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


if HAS_TORCH:

    class _TorchMLPNet(nn.Module):
        def __init__(
            self,
            n_features: int,
            hidden_layer_sizes: tuple[int, ...],
            dropout: float,
            batch_norm: bool,
        ):
            super().__init__()
            layers = []
            in_features = n_features
            for hidden_size in hidden_layer_sizes:
                layers.append(nn.Linear(in_features, hidden_size))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_features = hidden_size
            layers.append(nn.Linear(in_features, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """BatchNorm MLP classifier trained with Adam and BCEWithLogitsLoss."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.10,
        batch_norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 30,
        patience: int = 5,
        validation_fraction: float = 0.15,
        max_train_rows: int = 0,
        calibrate: bool = True,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.max_train_rows = max_train_rows
        self.calibrate = calibrate
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, x, y, sample_weight=None, validation_data=None):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TorchMLPClassifier. "
                "Install torch or use the sklearn MLP fallback."
            )
        rng = np.random.RandomState(self.random_state)
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        weights_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)

        if self.max_train_rows and len(y_arr) > self.max_train_rows:
            sample_idx, _ = train_test_split(
                np.arange(len(y_arr)),
                train_size=self.max_train_rows,
                random_state=self.random_state,
                stratify=y_arr,
            )
            x_arr = x_arr[sample_idx]
            y_arr = y_arr[sample_idx]
            if weights_arr is not None:
                weights_arr = weights_arr[sample_idx]

        idx = np.arange(len(y_arr))
        if validation_data is not None:
            train_idx = idx
            valid_x_arr = np.asarray(validation_data[0], dtype=np.float32)
            valid_y = np.asarray(validation_data[1], dtype=np.float32)
            self.validation_source_ = "external_validation_set"
        elif 0 < self.validation_fraction < 0.5 and len(np.unique(y_arr)) == 2:
            train_idx, valid_idx = train_test_split(
                idx,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_arr,
            )
            valid_x_arr = x_arr[valid_idx]
            valid_y = y_arr[valid_idx]
            self.validation_source_ = "internal_training_split"
        else:
            rng.shuffle(idx)
            split = max(1, int(len(idx) * 0.85))
            train_idx, valid_idx = idx[:split], idx[split:]
            valid_x_arr = x_arr[valid_idx]
            valid_y = y_arr[valid_idx]
            self.validation_source_ = "internal_training_split"

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.classes_ = np.array([0, 1])
        if self.device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(self.device)
        self.fit_train_rows_ = int(len(train_idx))
        self.training_device_ = str(self.device_)
        self.training_device_name_ = (
            torch.cuda.get_device_name(self.device_)
            if self.device_.type == "cuda"
            else "CPU"
        )
        self.model_ = _TorchMLPNet(
            n_features=x_arr.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device_)

        train_x = torch.tensor(x_arr[train_idx], dtype=torch.float32)
        train_y = torch.tensor(y_arr[train_idx], dtype=torch.float32)
        if weights_arr is None:
            train_w = torch.ones_like(train_y)
        else:
            train_w = torch.tensor(weights_arr[train_idx], dtype=torch.float32)
        valid_x = torch.tensor(valid_x_arr, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(train_x, train_y, train_w),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device_.type == "cuda",
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_auc = -np.inf
        best_state = None
        stale_epochs = 0
        self.training_history_ = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_losses = []
            for batch_x, batch_y, batch_w in loader:
                batch_x = batch_x.to(self.device_, non_blocking=True)
                batch_y = batch_y.to(self.device_, non_blocking=True)
                batch_w = batch_w.to(self.device_, non_blocking=True)
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = (criterion(logits, batch_y) * batch_w).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            valid_scores = self._predict_scores_array(valid_x.numpy())
            try:
                valid_auc = roc_auc_score(valid_y, valid_scores)
            except Exception:
                valid_auc = -float(np.mean(epoch_losses))
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)),
                    "valid_auc": float(valid_auc),
                }
            )
            if self.verbose:
                print(
                    f"BatchNorm MLP epoch {epoch + 1}: "
                    f"loss={np.mean(epoch_losses):.4f}, valid_auc={valid_auc:.4f}",
                    flush=True,
                )
            if valid_auc > best_auc + 1e-5:
                best_auc = valid_auc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.epochs_trained_ = int(len(self.training_history_))
        self.best_validation_auc_ = float(best_auc)
        self.calibrator_ = None
        self.calibration_method_ = "none"
        if self.calibrate and len(valid_y) > 0 and len(np.unique(valid_y)) == 2:
            valid_logits = self._predict_logits_array(valid_x.numpy())
            calibrator = SklearnLogisticRegression(max_iter=1_000, solver="lbfgs")
            calibrator.fit(valid_logits.reshape(-1, 1), valid_y.astype(int))
            if float(calibrator.coef_[0, 0]) > 0:
                self.calibrator_ = calibrator
                self.calibration_method_ = "validation_sigmoid_on_logits"
        return self

    def _predict_logits_array(self, x):
        x_arr = np.asarray(x, dtype=np.float32)
        self.model_.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(x_arr), self.batch_size):
                batch = torch.tensor(
                    x_arr[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device_,
                )
                logits.append(self.model_(batch).cpu().numpy())
        return np.concatenate(logits)

    def _predict_scores_array(self, x):
        logits = self._predict_logits_array(x)
        if getattr(self, "calibrator_", None) is not None:
            return self.calibrator_.predict_proba(logits.reshape(-1, 1))[:, 1]
        return 1 / (1 + np.exp(-logits))

    def predict_proba(self, x):
        scores = self._predict_scores_array(x)
        return np.column_stack([1 - scores, scores])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


POSITIVE_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
NEGATIVE_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

RAW_COLUMNS = [
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mort_acc",
    "pub_rec_bankruptcies",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "disbursement_method",
]

BORROWER_FEATURES = [
    "emp_length_num",
    "emp_length_missing",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "addr_state",
]

RISK_ASSESSMENT_FEATURES = [
    "grade",
]

FINANCIAL_FEATURES = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_score",
    "credit_history_months",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "initial_list_status",
    "disbursement_method",
]

RQ2_GROUPING_VERSION = "v2_fico_and_credit_history_in_financial_credit"
RQ2_FEATURE_SET_LABELS = {
    "borrower_only": "Borrower-only",
    "financial_only": "Financial-only",
    "grade_only": "Grade-only",
    "borrower_financial": "Borrower + Financial",
    "combined": "Combined",
}
PLOT_MODEL_LABELS = {
    "FT-Transformer": "Modified FT-Transformer-style",
}

NUMERIC_DERIVED = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "emp_length_num",
    "emp_length_missing",
    "fico_score",
    "credit_history_months",
]

COUNT_FEATURES = {
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "acc_open_past_24mths",
}

GLM_RIDGE_ALPHA = 1e-4

EBM_ORIGINAL_SCALE_FEATURES = [
    "int_rate",
    "dti",
    "loan_amnt",
    "annual_inc",
    "fico_score",
]

EBM_ORIGINAL_AXIS_LABELS = {
    "int_rate": "Interest rate (%)",
    "dti": "Debt-to-income ratio (%)",
    "loan_amnt": "Loan amount (USD)",
    "annual_inc": "Annual income (USD)",
    "fico_score": "FICO score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean LendingClub accepted loans, run EDA, and compare default prediction models."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the local LendingClub accepted-loan archive.zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "lendingclub_default_analysis",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=500_000,
        help=(
            "Modelling sample after filtering completed loans. Sampling is proportional "
            "within default-by-issue-year strata. Use 0 for all rows."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["random", "temporal", "both"],
        default="both",
        help="Run the main random split, the temporal robustness split, or both.",
    )
    parser.add_argument(
        "--term-filter",
        choices=["all", "36", "60"],
        default="all",
        help="Optional sensitivity filter for 36- or 60-month loans.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.16,
        help="Share of the full modelling sample reserved for validation.",
    )
    parser.add_argument(
        "--max-statmodels-rows",
        type=int,
        default=0,
        help=(
            "Optional cap for predictive logit/probit estimation. "
            "Use 0 (the default) to train predictions on the complete training set."
        ),
    )
    parser.add_argument(
        "--glm-inference-rows",
        type=int,
        default=0,
        help=(
            "Optional training-only GLM uncertainty analysis. Use 0 (the default) "
            "to disable it; reported coefficients and odds ratios are taken from "
            "the full-training predictive model."
        ),
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Skip MLP if a very quick run is needed.",
    )
    parser.add_argument(
        "--skip-ft-transformer",
        action="store_true",
        help="Skip FT-Transformer if PyTorch is unavailable or a quick run is needed.",
    )
    parser.add_argument(
        "--ft-max-train-rows",
        type=int,
        default=0,
        help=(
            "Optional training-row cap shared by MLP and FT-Transformer. "
            "Use 0 (the default) for the complete training set."
        ),
    )
    parser.add_argument(
        "--ft-epochs",
        type=int,
        default=12,
        help="Maximum FT-Transformer epochs with early stopping.",
    )
    parser.add_argument(
        "--ft-batch-size",
        type=int,
        default=512,
        help="FT-Transformer mini-batch size.",
    )
    parser.add_argument(
        "--skip-tree-tuning",
        action="store_true",
        help="Skip validation grid search for boosted/tree models and neural parameter candidates.",
    )
    parser.add_argument(
        "--skip-lightgbm",
        action="store_true",
        help="Skip LightGBM even if the package is available.",
    )
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost even if the package is available.",
    )
    parser.add_argument(
        "--skip-ebm",
        action="store_true",
        help="Skip Explainable Boosting Machine even if interpret-core is available.",
    )
    parser.add_argument(
        "--tuning-max-rows",
        type=int,
        default=20_000,
        help="Maximum training rows used inside the XGBoost/RF validation grid search.",
    )
    parser.add_argument(
        "--shap-sample-rows",
        type=int,
        default=5_000,
        help="Random final-test observations used for best-tree SHAP analysis (3,000-5,000 recommended).",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def requested_model_names(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    rq1_models = [
        "Logistic Regression",
        "Probit Regression",
        "Linear SVM",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Explainable Boosting Machine",
        "MLP Neural Network",
        "FT-Transformer",
    ]
    exclusions = {
        "LightGBM": args.skip_lightgbm,
        "CatBoost": args.skip_catboost,
        "Explainable Boosting Machine": args.skip_ebm,
        "MLP Neural Network": args.skip_mlp,
        "FT-Transformer": args.skip_ft_transformer,
    }
    rq1_models = [model for model in rq1_models if not exclusions.get(model, False)]
    rq2_models = [
        model
        for model in [
            "Logistic Regression",
            "LightGBM",
            "Explainable Boosting Machine",
        ]
        if not exclusions.get(model, False)
    ]
    return rq1_models, rq2_models


def validate_requested_dependencies(args: argparse.Namespace) -> None:
    missing = []
    if not HAS_XGBOOST:
        missing.append("xgboost")
    if not args.skip_lightgbm and not HAS_LIGHTGBM:
        missing.append("lightgbm")
    if not args.skip_catboost and not HAS_CATBOOST:
        missing.append("catboost")
    if not args.skip_ebm and not HAS_EBM:
        missing.append("interpret")
    if not args.skip_ft_transformer and not HAS_TORCH:
        missing.append("torch")
    if installed_version("shap") == "not installed":
        missing.append("shap")
    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Required packages are unavailable for the requested run: "
            f"{packages}. Create an isolated environment and install requirements.txt "
            "before running the experiment."
        )


def accepted_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    gz_matches = [name for name in names if name.lower().endswith("accepted_2007_to_2018q4.csv.gz")]
    if gz_matches:
        return gz_matches[0]
    csv_matches = [name for name in names if name.lower().endswith("accepted_2007_to_2018q4.csv")]
    if csv_matches:
        return csv_matches[0]
    raise FileNotFoundError("Could not find the accepted LendingClub CSV inside the archive.")


def read_header(zip_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.open(member)
        if member.lower().endswith(".gz"):
            raw = gzip.GzipFile(fileobj=raw)
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        header = pd.read_csv(text, nrows=0).columns.tolist()
    return header


def loan_status_target(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip()
    positive = clean.str.contains(r"(charged\s*off|default)", case=False, regex=True, na=False)
    negative = clean.str.contains(r"fully\s*paid", case=False, regex=True, na=False)
    target = pd.Series(np.nan, index=series.index, dtype="float")
    target.loc[negative] = 0.0
    target.loc[positive] = 1.0
    return target


def iter_accepted_chunks(
    zip_path: Path,
    member: str,
    usecols: list[str],
    chunksize: int,
):
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.open(member)
        if member.lower().endswith(".gz"):
            raw = gzip.GzipFile(fileobj=raw)
        yield from pd.read_csv(
            raw,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="warn",
        )


def load_completed_loans(
    archive: Path,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    member = accepted_member(archive)
    header = read_header(archive, member)
    usecols = [col for col in RAW_COLUMNS if col in header]
    if "loan_status" not in usecols:
        raise ValueError("loan_status is required but was not found in the accepted data.")

    status_counts: Counter[str] = Counter()
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    retained_rows = 0
    bad_line_warnings = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        for idx, chunk in enumerate(iter_accepted_chunks(archive, member, usecols, chunksize), start=1):
            total_rows += len(chunk)
            status_counts.update(chunk["loan_status"].astype("string").fillna("Missing").tolist())
            target = loan_status_target(chunk["loan_status"])
            keep = target.notna()
            filtered = chunk.loc[keep].copy()
            retained_rows += len(filtered)
            if len(filtered):
                chunks.append(filtered)
            print(
                f"Loaded chunk {idx:02d}: total rows={total_rows:,}, completed rows={retained_rows:,}",
                flush=True,
            )
        bad_line_warnings = sum("Skipping line" in str(w.message) for w in caught)

    if not chunks:
        raise ValueError("No completed loans were found after filtering loan_status.")

    df = pd.concat(chunks, ignore_index=True)
    duplicate_rows_dropped = 0
    if "id" in df.columns:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
        duplicate_rows_dropped = before_dedup - len(df)
    metadata = {
        "archive": str(archive),
        "zip_member": member,
        "raw_columns_used": usecols,
        "total_rows_read": total_rows,
        "completed_rows_retained": retained_rows,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "bad_lines_skipped": bad_line_warnings,
        "bad_lines_handling": "warn",
    }
    return df, pd.Series(status_counts).sort_values(ascending=False), metadata


def parse_emp_length(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.replace({"n/a": pd.NA, "nan": pd.NA, "none": pd.NA, "": pd.NA})
    out = cleaned.str.extract(r"(\d+)")[0].astype("float")
    out = out.where(~cleaned.str.contains("< 1", na=False), 0.0)
    out = out.where(~cleaned.str.contains("10\\+", na=False), 10.0)
    return out


def parse_month_year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string").str.strip(), format="%b-%Y", errors="coerce")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cleaning_report: dict[str, int] = {}
    out["loan_status"] = out["loan_status"].astype("string").str.strip()
    out["default"] = loan_status_target(out["loan_status"])

    out["issue_date"] = parse_month_year(out.get("issue_d", pd.Series(index=out.index)))
    out["earliest_cr_line_date"] = parse_month_year(
        out.get("earliest_cr_line", pd.Series(index=out.index))
    )

    if "term" in out:
        out["term_months"] = (
            out["term"].astype("string").str.extract(r"(\d+)")[0].astype("float")
        )
    if "emp_length" in out:
        out["emp_length_num"] = parse_emp_length(out["emp_length"])
        out["emp_length_missing"] = out["emp_length_num"].isna().astype(int)
        cleaning_report["emp_length_missing_rows"] = int(out["emp_length_missing"].sum())

    for col in ["int_rate", "revol_util"]:
        if col in out:
            out[col] = (
                out[col]
                .astype("string")
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({"": pd.NA})
            )

    numeric_raw = [
        col
        for col in NUMERIC_DERIVED
        + ["fico_range_low", "fico_range_high"]
        if col in out.columns
    ]
    for col in numeric_raw:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "annual_inc" in out:
        annual_negative = out["annual_inc"] < 0
        cleaning_report["annual_inc_negative_to_nan"] = int(annual_negative.sum())
        out.loc[annual_negative, "annual_inc"] = np.nan
        cleaning_report["annual_inc_gt_1m_rows"] = int((out["annual_inc"] > 1_000_000).sum())

    if "dti" in out:
        dti_sentinel = (out["dti"] < 0) | (out["dti"] > 100)
        cleaning_report["dti_sentinel_to_nan"] = int(dti_sentinel.sum())
        out.loc[dti_sentinel, "dti"] = np.nan

    if "revol_util" in out:
        revol_negative = out["revol_util"] < 0
        revol_over_100 = out["revol_util"] > 100
        cleaning_report["revol_util_negative_to_nan"] = int(revol_negative.sum())
        cleaning_report["revol_util_over_100_clipped"] = int(revol_over_100.sum())
        out.loc[revol_negative, "revol_util"] = np.nan
        out.loc[revol_over_100, "revol_util"] = 100.0

    for col in COUNT_FEATURES:
        if col in out:
            negative_count = out[col] < 0
            cleaning_report[f"{col}_negative_to_nan"] = int(negative_count.sum())
            out.loc[negative_count, col] = np.nan

    if {"fico_range_low", "fico_range_high"}.issubset(out.columns):
        out["fico_score"] = out[["fico_range_low", "fico_range_high"]].mean(axis=1)

    if {"issue_date", "earliest_cr_line_date"}.issubset(out.columns):
        days = (out["issue_date"] - out["earliest_cr_line_date"]).dt.days
        out["credit_history_months"] = days / 30.4375
        negative_history = out["credit_history_months"] < 0
        cleaning_report["credit_history_negative_to_nan"] = int(negative_history.sum())
        out.loc[negative_history, "credit_history_months"] = np.nan

    categorical = [
        col
        for col in BORROWER_FEATURES + RISK_ASSESSMENT_FEATURES + FINANCIAL_FEATURES
        if col in out.columns
    ]
    categorical = [col for col in categorical if col not in NUMERIC_DERIVED]
    for col in categorical:
        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.attrs["cleaning_report"] = cleaning_report
    return out


def stratified_sample(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    """Draw an exact-size proportional sample within default-by-issue-year strata."""
    if n_rows <= 0 or len(df) <= n_rows:
        return df.copy().reset_index(drop=True)

    working = df.copy()
    issue_year = working["issue_date"].dt.year if "issue_date" in working else pd.Series(np.nan, index=working.index)
    working["_sample_issue_year"] = issue_year.fillna(-1).astype(int)
    strata = ["default", "_sample_issue_year"]
    sizes = working.groupby(strata, dropna=False).size().rename("population_rows").reset_index()
    exact_quota = sizes["population_rows"] * (n_rows / len(working))
    sizes["sample_rows"] = np.floor(exact_quota).astype(int)
    sizes["remainder"] = exact_quota - sizes["sample_rows"]

    rows_left = int(n_rows - sizes["sample_rows"].sum())
    if rows_left > 0:
        eligible = sizes.loc[sizes["sample_rows"] < sizes["population_rows"]].sort_values(
            ["remainder", "population_rows"], ascending=[False, False]
        )
        for idx in eligible.index[:rows_left]:
            sizes.loc[idx, "sample_rows"] += 1

    quota_by_stratum = sizes.set_index(strata)["sample_rows"].astype(int).to_dict()
    pieces = []
    for key, stratum in working.groupby(strata, dropna=False, sort=True):
        quota = int(quota_by_stratum.get(key, 0))
        if quota > 0:
            pieces.append(stratum.sample(n=quota, random_state=random_state))
    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sample.drop(columns=["_sample_issue_year"])


def save_sample_representativeness(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, object]:
    """Compare the modelling sample with the full completed-loan population."""
    tables = paths["tables"]
    figures = paths["figures"]
    categorical_map = {
        "grade": "grade",
        "term": "term_months",
        "purpose": "purpose",
    }
    categorical_rows = []
    for label, col in categorical_map.items():
        if col not in population or col not in sample:
            continue
        pop = population[col].astype("string").fillna("Missing").value_counts(dropna=False)
        sam = sample[col].astype("string").fillna("Missing").value_counts(dropna=False)
        levels = pop.index.union(sam.index)
        for level in levels:
            pop_count = int(pop.get(level, 0))
            sample_count = int(sam.get(level, 0))
            pop_share = pop_count / max(len(population), 1)
            sample_share = sample_count / max(len(sample), 1)
            categorical_rows.append(
                {
                    "variable": label,
                    "category": str(level),
                    "population_count": pop_count,
                    "sample_count": sample_count,
                    "population_share": pop_share,
                    "sample_share": sample_share,
                    "absolute_share_difference": abs(sample_share - pop_share),
                }
            )
    categorical_table = pd.DataFrame(categorical_rows)
    categorical_table.to_csv(
        tables / "sample_vs_population_categorical_distributions.csv", index=False
    )

    numerical_rows = []
    numerical_features = ["loan_amnt", "int_rate", "fico_score", "dti", "annual_inc"]
    for col in numerical_features:
        if col not in population or col not in sample:
            continue
        pop_values = pd.to_numeric(population[col], errors="coerce").dropna().to_numpy()
        sample_values = pd.to_numeric(sample[col], errors="coerce").dropna().to_numpy()
        if len(pop_values) == 0 or len(sample_values) == 0:
            continue
        pop_mean = float(np.mean(pop_values))
        sample_mean = float(np.mean(sample_values))
        pooled_scale = float(np.std(pop_values, ddof=1))
        numerical_rows.append(
            {
                "variable": col,
                "population_nonmissing": len(pop_values),
                "sample_nonmissing": len(sample_values),
                "population_mean": pop_mean,
                "sample_mean": sample_mean,
                "population_std": pooled_scale,
                "sample_std": float(np.std(sample_values, ddof=1)),
                "population_median": float(np.median(pop_values)),
                "sample_median": float(np.median(sample_values)),
                "population_p01": float(np.quantile(pop_values, 0.01)),
                "sample_p01": float(np.quantile(sample_values, 0.01)),
                "population_p99": float(np.quantile(pop_values, 0.99)),
                "sample_p99": float(np.quantile(sample_values, 0.99)),
                "standardized_mean_difference": (
                    (sample_mean - pop_mean) / pooled_scale if pooled_scale > 0 else np.nan
                ),
                "ks_statistic": float(stats.ks_2samp(pop_values, sample_values).statistic),
            }
        )
    numerical_table = pd.DataFrame(numerical_rows)
    numerical_table.to_csv(
        tables / "sample_vs_population_numerical_distributions.csv", index=False
    )

    pop_strata = population.assign(
        issue_year=population["issue_date"].dt.year.fillna(-1).astype(int)
    ).groupby(["default", "issue_year"]).size().rename("population_count")
    sample_strata = sample.assign(
        issue_year=sample["issue_date"].dt.year.fillna(-1).astype(int)
    ).groupby(["default", "issue_year"]).size().rename("sample_count")
    strata_table = pd.concat([pop_strata, sample_strata], axis=1).fillna(0).reset_index()
    strata_table["population_share"] = strata_table["population_count"] / len(population)
    strata_table["sample_share"] = strata_table["sample_count"] / len(sample)
    strata_table["absolute_share_difference"] = (
        strata_table["sample_share"] - strata_table["population_share"]
    ).abs()
    strata_table.to_csv(tables / "sampling_strata_default_issue_year.csv", index=False)

    sns.set_theme(style="whitegrid")
    if not categorical_table.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (label, group) in zip(axes, categorical_table.groupby("variable", sort=False)):
            plot_group = group.sort_values("population_share", ascending=False).head(12)
            plot_df = plot_group.melt(
                id_vars="category",
                value_vars=["population_share", "sample_share"],
                var_name="dataset",
                value_name="share",
            )
            sns.barplot(data=plot_df, x="category", y="share", hue="dataset", ax=ax)
            ax.set_title(label)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle("500,000-Loan Sample vs Completed-Loan Population")
        fig.tight_layout()
        fig.savefig(figures / "sample_representativeness_categorical.png", dpi=160)
        plt.close(fig)

    available_numeric = [col for col in numerical_features if col in population and col in sample]
    if available_numeric:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        for ax, col in zip(axes, available_numeric):
            pop_values = pd.to_numeric(population[col], errors="coerce").dropna()
            sample_values = pd.to_numeric(sample[col], errors="coerce").dropna()
            lo, hi = pop_values.quantile([0.01, 0.99])
            pop_plot = pop_values.loc[pop_values.between(lo, hi)].sample(
                n=min(30_000, int(pop_values.between(lo, hi).sum())), random_state=42
            )
            sample_plot = sample_values.loc[sample_values.between(lo, hi)].sample(
                n=min(30_000, int(sample_values.between(lo, hi).sum())), random_state=42
            )
            sns.ecdfplot(pop_plot, ax=ax, label="Population", color="#4C78A8")
            sns.ecdfplot(sample_plot, ax=ax, label="500k sample", color="#E45756", linestyle="--")
            ax.set_title(col)
            ax.legend()
        for ax in axes[len(available_numeric):]:
            ax.axis("off")
        fig.suptitle("Numerical Distribution Representativeness (P1-P99)")
        fig.tight_layout()
        fig.savefig(figures / "sample_representativeness_numerical.png", dpi=160)
        plt.close(fig)

    return {
        "maximum_categorical_share_difference": (
            float(categorical_table["absolute_share_difference"].max())
            if not categorical_table.empty
            else None
        ),
        "maximum_absolute_standardized_mean_difference": (
            float(numerical_table["standardized_mean_difference"].abs().max())
            if not numerical_table.empty
            else None
        ),
        "maximum_numerical_ks_statistic": (
            float(numerical_table["ks_statistic"].max())
            if not numerical_table.empty
            else None
        ),
        "maximum_stratum_share_difference": float(strata_table["absolute_share_difference"].max()),
    }


def available_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    features = []
    for col in candidates:
        if col not in df.columns:
            continue
        if df[col].notna().sum() == 0:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        features.append(col)
    return features


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [col for col in features if col in NUMERIC_DERIVED]
    categorical = [col for col in features if col not in numeric]
    return numeric, categorical


def make_preprocessor(
    df: pd.DataFrame,
    features: list[str],
    *,
    glm: bool = False,
) -> ColumnTransformer:
    numeric, categorical = split_feature_types(df, features)
    count_numeric = [col for col in numeric if col in COUNT_FEATURES]
    continuous_numeric = [col for col in numeric if col not in COUNT_FEATURES]
    continuous_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", QuantileClipper(0.01, 0.99)),
            ("scaler", StandardScaler()),
        ]
    )
    count_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", QuantileClipper(0.01, 0.99)),
            (
                "log1p",
                FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one"),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore" if glm else "infrequent_if_exist",
                    min_frequency=None if glm else 50,
                    drop="first" if glm else None,
                    sparse_output=False,
                ),
            ),
        ]
    )
    transformers = []
    if continuous_numeric:
        transformers.append(("num", continuous_pipe, continuous_numeric))
    if count_numeric:
        transformers.append(("count", count_pipe, count_numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return preprocessor.get_feature_names_out().tolist()


def train_validation_test_frames(
    df: pd.DataFrame,
    features: list[str],
    split: str,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, object],
]:
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")
    model_df = df[features + ["default", "issue_date"]].copy()
    model_df = model_df.dropna(subset=["default"])
    model_df["_row_id"] = model_df.index
    issue_date_missing = int(model_df["issue_date"].isna().sum())

    if split == "temporal" and model_df["issue_date"].notna().sum() > 0:
        temporal_df = model_df.dropna(subset=["issue_date"]).copy()
        ordered = temporal_df.sort_values(["issue_date", "_row_id"]).reset_index(drop=True)
        train_share = 1 - validation_size - test_size
        month_counts = ordered.groupby("issue_date", sort=True).size()
        cumulative_counts = month_counts.cumsum()
        train_target = len(ordered) * train_share
        validation_target = len(ordered) * (train_share + validation_size)
        train_cutoff = (cumulative_counts - train_target).abs().idxmin()
        validation_candidates = cumulative_counts.loc[
            cumulative_counts.index > train_cutoff
        ]
        if validation_candidates.empty:
            raise ValueError("Temporal split has no month available for validation.")
        validation_cutoff = (
            validation_candidates - validation_target
        ).abs().idxmin()
        train = ordered.loc[ordered["issue_date"] <= train_cutoff].copy()
        validation = ordered.loc[
            (ordered["issue_date"] > train_cutoff)
            & (ordered["issue_date"] <= validation_cutoff)
        ].copy()
        test = ordered.loc[ordered["issue_date"] > validation_cutoff].copy()
        if all(part["default"].nunique() >= 2 for part in [train, validation, test]):
            meta = {
                "split": "temporal",
                "train_share_requested": round(train_share, 10),
                "validation_share_requested": validation_size,
                "test_share_requested": test_size,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_share_actual": len(train) / len(ordered),
                "validation_share_actual": len(validation) / len(ordered),
                "test_share_actual": len(test) / len(ordered),
                "whole_month_boundaries_enforced": True,
                "train_cutoff_month": str(pd.Timestamp(train_cutoff).date()),
                "validation_cutoff_month": str(
                    pd.Timestamp(validation_cutoff).date()
                ),
                "train_date_min": str(train["issue_date"].min().date()),
                "train_date_max": str(train["issue_date"].max().date()),
                "validation_date_min": str(validation["issue_date"].min().date()),
                "validation_date_max": str(validation["issue_date"].max().date()),
                "test_date_min": str(test["issue_date"].min().date()),
                "test_date_max": str(test["issue_date"].max().date()),
                "train_default_rate": float(train["default"].mean()),
                "validation_default_rate": float(validation["default"].mean()),
                "test_default_rate": float(test["default"].mean()),
                "issue_date_missing_dropped": issue_date_missing,
                "issue_date_missing_rate": issue_date_missing / max(len(model_df), 1),
                "test_set_usage": "final_evaluation_only",
            }
            return (
                train[features],
                validation[features],
                test[features],
                train["default"].astype(int),
                validation["default"].astype(int),
                test["default"].astype(int),
                meta,
            )
        raise ValueError("Temporal train/validation/test split does not contain both target classes.")

    train_validation, test = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df["default"],
    )
    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_validation["default"],
    )
    meta = {
        "split": "random",
        "train_share_requested": round(1 - validation_size - test_size, 10),
        "validation_share_requested": validation_size,
        "test_share_requested": test_size,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_default_rate": float(train["default"].mean()),
        "validation_default_rate": float(validation["default"].mean()),
        "test_default_rate": float(test["default"].mean()),
        "train_date_min": str(train["issue_date"].min().date()),
        "train_date_max": str(train["issue_date"].max().date()),
        "validation_date_min": str(validation["issue_date"].min().date()),
        "validation_date_max": str(validation["issue_date"].max().date()),
        "test_date_min": str(test["issue_date"].min().date()),
        "test_date_max": str(test["issue_date"].max().date()),
        "issue_date_missing_rows": issue_date_missing,
        "issue_date_missing_rate": issue_date_missing / max(len(model_df), 1),
        "test_set_usage": "final_evaluation_only",
    }
    return (
        train[features],
        validation[features],
        test[features],
        train["default"].astype(int),
        validation["default"].astype(int),
        test["default"].astype(int),
        meta,
    )


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def ridge_glm_inference(
    model,
    params: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=float)
    try:
        information = -np.asarray(model.hessian(params), dtype=float)
        penalty = np.eye(len(params)) * alpha
        penalty[0, 0] = 0.0
        covariance = np.linalg.pinv(information + penalty)
        std_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        z_value = params / np.where(std_error > 0, std_error, np.nan)
        p_value = 2 * stats.norm.sf(np.abs(z_value))
        conf_int = np.column_stack(
            [
                params - 1.96 * std_error,
                params + 1.96 * std_error,
            ]
        )
    except Exception:
        std_error = np.full_like(params, np.nan, dtype=float)
        z_value = np.full_like(params, np.nan, dtype=float)
        p_value = np.full_like(params, np.nan, dtype=float)
        conf_int = np.column_stack([std_error, std_error])
    return std_error, z_value, p_value, conf_int


def metric_row(
    model_name: str,
    feature_set: str,
    y_validation: pd.Series,
    validation_scores: np.ndarray,
    y_test: pd.Series,
    test_scores: np.ndarray,
    raw_validation_scores: np.ndarray | None = None,
    raw_test_scores: np.ndarray | None = None,
    calibration_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_validation_scores = (
        np.asarray(validation_scores, dtype=float)
        if raw_validation_scores is None
        else np.asarray(raw_validation_scores, dtype=float)
    )
    raw_test_scores = (
        np.asarray(test_scores, dtype=float)
        if raw_test_scores is None
        else np.asarray(raw_test_scores, dtype=float)
    )
    calibration_metadata = calibration_metadata or {}
    threshold = best_f1_threshold(y_validation, validation_scores)
    pred = (test_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_auc = roc_auc_score(y_test, test_scores)
    return {
        "feature_set": feature_set,
        "model": model_name,
        "threshold": threshold,
        "threshold_source": "validation_best_f1",
        "validation_roc_auc": roc_auc_score(y_validation, validation_scores),
        "validation_avg_precision": average_precision_score(y_validation, validation_scores),
        "validation_brier_score": brier_score_loss(
            y_validation, np.clip(validation_scores, 0, 1)
        ),
        "raw_validation_brier_score": brier_score_loss(
            y_validation, np.clip(raw_validation_scores, 0, 1)
        ),
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "ks_statistic": float(np.max(tpr - fpr)),
        "avg_precision": average_precision_score(y_test, test_scores),
        "brier_score": brier_score_loss(y_test, np.clip(test_scores, 0, 1)),
        "raw_brier_score": brier_score_loss(y_test, np.clip(raw_test_scores, 0, 1)),
        "raw_probability_mean": float(np.mean(raw_test_scores)),
        "calibrated_probability_mean": float(np.mean(test_scores)),
        "probability_calibration": calibration_metadata.get(
            "probability_calibration", "none"
        ),
        "calibration_slope": calibration_metadata.get("calibration_slope", np.nan),
        "calibration_intercept": calibration_metadata.get(
            "calibration_intercept", np.nan
        ),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def validation_sigmoid_calibration(
    y_validation: pd.Series,
    raw_validation_scores: np.ndarray,
    raw_test_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit one common Platt-style map on validation predictions only."""
    eps = 1e-7
    raw_validation_scores = np.clip(
        np.asarray(raw_validation_scores, dtype=float), eps, 1 - eps
    )
    raw_test_scores = np.clip(np.asarray(raw_test_scores, dtype=float), eps, 1 - eps)
    validation_log_odds = np.log(raw_validation_scores / (1 - raw_validation_scores))
    test_log_odds = np.log(raw_test_scores / (1 - raw_test_scores))
    calibrator = SklearnLogisticRegression(
        C=1e6,
        penalty="l2",
        solver="lbfgs",
        max_iter=1_000,
    )
    calibrator.fit(validation_log_odds.reshape(-1, 1), np.asarray(y_validation, dtype=int))
    slope = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    if slope <= 0:
        raise ValueError(
            "Validation sigmoid calibration produced a non-positive slope, which would "
            "reverse the model ranking. Inspect the raw predictions before proceeding."
        )
    validation_scores = calibrator.predict_proba(validation_log_odds.reshape(-1, 1))[:, 1]
    test_scores = calibrator.predict_proba(test_log_odds.reshape(-1, 1))[:, 1]
    metadata = {
        "probability_calibration": "validation_sigmoid_on_raw_log_odds",
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    return validation_scores, test_scores, metadata


def recover_raw_scores(calibrated_scores: np.ndarray, metric: dict[str, object]) -> np.ndarray:
    """Recover raw probabilities for auditable appendix outputs."""
    slope = float(metric["calibration_slope"])
    intercept = float(metric["calibration_intercept"])
    eps = 1e-7
    calibrated_scores = np.clip(np.asarray(calibrated_scores, dtype=float), eps, 1 - eps)
    calibrated_log_odds = np.log(calibrated_scores / (1 - calibrated_scores))
    raw_log_odds = (calibrated_log_odds - intercept) / slope
    raw_log_odds = np.clip(raw_log_odds, -30, 30)
    return 1 / (1 + np.exp(-raw_log_odds))


def fit_glm(
    name: str,
    link_name: str,
    feature_set: str,
    features: list[str],
    train_x: pd.DataFrame,
    validation_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    validation_y: pd.Series,
    test_y: pd.Series,
    max_rows: int,
    inference_rows: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, np.ndarray]:
    preprocessor = make_preprocessor(train_x, features, glm=True)
    train_mat = preprocessor.fit_transform(train_x)
    validation_mat = preprocessor.transform(validation_x)
    test_mat = preprocessor.transform(test_x)
    names = feature_names(preprocessor)

    if max_rows > 0 and len(train_y) > max_rows:
        sample_idx, _ = train_test_split(
            np.arange(len(train_y)),
            train_size=max_rows,
            random_state=random_state,
            stratify=train_y,
        )
        fit_x = train_mat[sample_idx]
        fit_y = train_y.iloc[sample_idx]
    else:
        fit_x = train_mat
        fit_y = train_y

    link = sm.families.links.Logit() if link_name == "logit" else sm.families.links.Probit()
    if link_name == "logit":
        prediction_model = SklearnLogisticRegression(
            penalty="l2",
            C=1.0 / GLM_RIDGE_ALPHA,
            solver="lbfgs",
            max_iter=1_000,
            random_state=random_state,
        )
        prediction_model.fit(fit_x, fit_y)
        raw_validation_scores = prediction_model.predict_proba(validation_mat)[:, 1]
        raw_test_scores = prediction_model.predict_proba(test_mat)[:, 1]
        prediction_params = np.concatenate(
            [prediction_model.intercept_.ravel(), prediction_model.coef_.ravel()]
        )
        prediction_estimator = "sklearn_logistic_regression_full_training"
    else:
        fit_x_const = sm.add_constant(fit_x, has_constant="add")
        validation_const = sm.add_constant(validation_mat, has_constant="add")
        test_const = sm.add_constant(test_mat, has_constant="add")
        prediction_model = sm.GLM(
            fit_y,
            fit_x_const,
            family=sm.families.Binomial(link=link),
        )
        try:
            prediction_result = prediction_model.fit_regularized(
                alpha=GLM_RIDGE_ALPHA,
                L1_wt=0.0,
                maxiter=200,
            )
        except Exception:
            prediction_result = prediction_model.fit(maxiter=100, disp=0)
        raw_validation_scores = np.asarray(prediction_result.predict(validation_const))
        raw_test_scores = np.asarray(prediction_result.predict(test_const))
        prediction_params = np.asarray(prediction_result.params)
        prediction_estimator = "statsmodels_regularized_probit_full_training"
    validation_scores, test_scores, calibration_metadata = validation_sigmoid_calibration(
        validation_y,
        raw_validation_scores,
        raw_test_scores,
    )
    metrics = metric_row(
        name,
        feature_set,
        validation_y,
        validation_scores,
        test_y,
        test_scores,
        raw_validation_scores,
        raw_test_scores,
        calibration_metadata,
    )
    metrics.update(
        {
            "training_rows": int(len(fit_y)),
            "inference_rows": int(min(inference_rows, len(train_y)))
            if inference_rows > 0
            else 0,
            "training_device": "CPU",
            "prediction_estimator": prediction_estimator,
            "coefficient_source": "full_training_fitted_prediction_model",
            "coefficient_training_rows": int(len(fit_y)),
            "supplementary_uncertainty_rows": int(min(inference_rows, len(train_y)))
            if inference_rows > 0
            else 0,
        }
    )

    feature_labels = ["const"] + names
    if len(prediction_params) != len(feature_labels):
        raise ValueError(
            f"Coefficient alignment failed for {name}: "
            f"{len(prediction_params)} parameters for {len(feature_labels)} labels."
        )
    coef = pd.DataFrame(
        {
            "feature": feature_labels,
            "coefficient": prediction_params,
            "abs_coefficient": np.abs(prediction_params),
            "coefficient_source": "full_training_fitted_prediction_model",
            "coefficient_training_rows": int(len(fit_y)),
            "prediction_estimator": prediction_estimator,
        }
    )
    if link_name == "logit":
        coef["odds_ratio"] = np.exp(coef["coefficient"].clip(-20, 20))
    coef = coef.sort_values("abs_coefficient", ascending=False)

    if inference_rows <= 0:
        return metrics, coef, pd.DataFrame(), test_scores

    if len(train_y) > inference_rows:
        inference_idx, _ = train_test_split(
            np.arange(len(train_y)),
            train_size=inference_rows,
            random_state=random_state,
            stratify=train_y,
        )
        inference_x = train_mat[inference_idx]
        inference_y = train_y.iloc[inference_idx]
    else:
        inference_x = train_mat
        inference_y = train_y
    inference_x_const = sm.add_constant(inference_x, has_constant="add")
    inference_model = sm.GLM(
        inference_y,
        inference_x_const,
        family=sm.families.Binomial(link=link),
    )
    try:
        inference_result = inference_model.fit_regularized(
            alpha=GLM_RIDGE_ALPHA,
            L1_wt=0.0,
            maxiter=200,
        )
    except Exception:
        inference_result = inference_model.fit(maxiter=100, disp=0)
    inference_params = np.asarray(inference_result.params)
    std_error, z_value, p_value, conf_int = ridge_glm_inference(
        inference_model,
        inference_params,
        GLM_RIDGE_ALPHA,
    )
    uncertainty = pd.DataFrame(
        {
            "feature": feature_labels,
            "coefficient": inference_params,
            "std_error": std_error,
            "z_value": z_value,
            "p_value": p_value,
            "ci_lower": conf_int[:, 0],
            "ci_upper": conf_int[:, 1],
            "abs_coefficient": np.abs(inference_params),
            "inference_method": "training_only_stratified_subsample_ridge_hessian",
            "inference_rows": int(len(inference_y)),
            "prediction_training_rows": int(len(fit_y)),
            "analysis_role": "supplementary_uncertainty_check_only",
            "main_interpretation_source": False,
        }
    )
    if link_name == "logit":
        uncertainty["odds_ratio"] = np.exp(
            uncertainty["coefficient"].clip(-20, 20)
        )
        uncertainty["odds_ratio_ci_lower"] = np.exp(
            uncertainty["ci_lower"].clip(-20, 20)
        )
        uncertainty["odds_ratio_ci_upper"] = np.exp(
            uncertainty["ci_upper"].clip(-20, 20)
        )
    uncertainty = uncertainty.sort_values("abs_coefficient", ascending=False)
    return metrics, coef, uncertainty, test_scores


def probability_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1 / (1 + np.exp(-scores))


def fit_sklearn_model(
    name: str,
    estimator,
    feature_set: str,
    features: list[str],
    train_x: pd.DataFrame,
    validation_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    validation_y: pd.Series,
    test_y: pd.Series,
    sample_weight: bool = False,
) -> tuple[dict[str, object], Pipeline, np.ndarray]:
    preprocessor = make_preprocessor(train_x, features)
    train_mat = preprocessor.fit_transform(train_x)
    validation_mat = preprocessor.transform(validation_x)
    model = clone(estimator)
    fit_kwargs: dict[str, object] = {}
    if sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
        fit_kwargs["sample_weight"] = weights
    if isinstance(model, (FTTransformerClassifier, TorchMLPClassifier)):
        fit_kwargs["validation_data"] = (validation_mat, validation_y.to_numpy())
    model.fit(train_mat, train_y, **fit_kwargs)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    raw_validation_scores = probability_scores(pipe, validation_x)
    raw_test_scores = probability_scores(pipe, test_x)
    validation_scores, test_scores, calibration_metadata = validation_sigmoid_calibration(
        validation_y,
        raw_validation_scores,
        raw_test_scores,
    )
    metrics = metric_row(
        name,
        feature_set,
        validation_y,
        validation_scores,
        test_y,
        test_scores,
        raw_validation_scores,
        raw_test_scores,
        calibration_metadata,
    )
    metrics.update(
        {
            "training_rows": int(getattr(model, "fit_train_rows_", len(train_y))),
            "inference_rows": np.nan,
            "training_device": str(getattr(model, "training_device_name_", "CPU")),
            "epochs_trained": getattr(model, "epochs_trained_", np.nan),
            "best_validation_auc_internal": getattr(
                model, "best_validation_auc_", np.nan
            ),
            "raw_probability_output": getattr(model, "calibration_method_", "native"),
        }
    )
    return metrics, pipe, test_scores


def sample_for_tuning(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(train_y) <= max_rows:
        return train_x, train_y
    _, sample_x, _, sample_y = train_test_split(
        train_x,
        train_y,
        test_size=max_rows,
        random_state=random_state,
        stratify=train_y,
    )
    return sample_x, sample_y


def tune_estimator_on_validation(
    model_name: str,
    base_estimator,
    param_grid: list[dict[str, object]],
    feature_set: str,
    features: list[str],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    random_state: int,
    max_rows: int,
) -> tuple[object, dict[str, object], pd.DataFrame]:
    tuning_x, tuning_y = sample_for_tuning(train_x, train_y, max_rows, random_state)
    preprocessor = make_preprocessor(tuning_x, features)
    tuning_mat = preprocessor.fit_transform(tuning_x)
    validation_mat = preprocessor.transform(validation_x)

    rows = []
    best_auc = -np.inf
    best_params: dict[str, object] = {}
    best_estimator = clone(base_estimator)

    for params in param_grid:
        estimator = clone(base_estimator).set_params(**params)
        fit_kwargs: dict[str, object] = {}
        if isinstance(estimator, (FTTransformerClassifier, TorchMLPClassifier)):
            fit_kwargs["validation_data"] = (validation_mat, validation_y.to_numpy())
        estimator.fit(tuning_mat, tuning_y, **fit_kwargs)
        if hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(validation_mat)[:, 1]
        else:
            decision = estimator.decision_function(validation_mat)
            scores = 1 / (1 + np.exp(-decision))
        auc = roc_auc_score(validation_y, scores)
        ap = average_precision_score(validation_y, scores)
        row = {
            "feature_set": feature_set,
            "model": model_name,
            "validation_roc_auc": auc,
            "validation_avg_precision": ap,
            "tuning_train_rows": len(tuning_y),
            "validation_rows": len(validation_y),
            "validation_source": "fixed_pipeline_validation_set",
        }
        row.update(params)
        rows.append(row)
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_estimator = estimator

    return best_estimator, best_params, pd.DataFrame(rows)


def save_eda(
    df: pd.DataFrame,
    status_counts: pd.Series,
    paths: dict[str, Path],
    borrower_features: list[str],
    financial_features: list[str],
):
    tables = paths["tables"]
    figures = paths["figures"]
    status_counts.rename_axis("loan_status").reset_index(name="count").to_csv(
        tables / "loan_status_counts_all_accepted.csv", index=False
    )

    target_counts = (
        df["default"].value_counts().rename(index={0: "non_default", 1: "default"})
    )
    target_counts.rename_axis("target").reset_index(name="count").to_csv(
        tables / "target_counts_model_sample.csv", index=False
    )

    features = borrower_features + financial_features
    missing = (
        df[features]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing.to_csv(tables / "missingness_model_sample.csv", index=False)

    numeric = [col for col in features if col in NUMERIC_DERIVED]
    if numeric:
        df[numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.to_csv(
            tables / "numeric_summary_model_sample.csv"
        )

    for col in ["grade", "purpose", "home_ownership"]:
        if col in df:
            summary = (
                df.groupby(col, dropna=False)["default"]
                .agg(default_rate="mean", count="size")
                .sort_values("default_rate", ascending=False)
            )
            summary.to_csv(tables / f"default_rate_by_{col}.csv")

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Default Target Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Loans")
    fig.tight_layout()
    fig.savefig(figures / "target_distribution.png", dpi=160)
    plt.close(fig)

    if "grade" in df:
        grade = (
            df.groupby("grade")["default"]
            .agg(default_rate="mean", count="size")
            .reindex(list("ABCDEFG"))
            .dropna(how="all")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=grade, x="grade", y="default_rate", ax=ax, color="#F58518")
        ax.set_title("Default Rate by Lending Grade")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Default rate")
        fig.tight_layout()
        fig.savefig(figures / "default_rate_by_grade.png", dpi=160)
        plt.close(fig)

    if "purpose" in df:
        purpose = (
            df.groupby("purpose")["default"]
            .agg(default_rate="mean", count="size")
            .query("count >= 100")
            .sort_values("default_rate", ascending=False)
            .head(12)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=purpose, y="purpose", x="default_rate", ax=ax, color="#72B7B2")
        ax.set_title("Default Rate by Purpose")
        ax.set_xlabel("Default rate")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(figures / "default_rate_by_purpose.png", dpi=160)
        plt.close(fig)

    corr_cols = [col for col in numeric if df[col].notna().mean() > 0.5]
    corr_cols = corr_cols[:25] + ["default"]
    if len(corr_cols) > 2:
        corr = df[corr_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(figures / "correlation_heatmap.png", dpi=160)
        plt.close(fig)

    plot_cols = [col for col in ["loan_amnt", "int_rate", "annual_inc", "dti"] if col in df]
    if plot_cols:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes = axes.ravel()
        for ax, col in zip(axes, plot_cols):
            plot_data = df[[col, "default"]].dropna().copy()
            if col in {"annual_inc", "dti"}:
                upper = plot_data[col].quantile(0.99)
                plot_data[col] = plot_data[col].clip(upper=upper)
            sns.boxplot(data=plot_data, x="default", y=col, ax=ax, color="#B279A2")
            ax.set_xlabel("Default")
        for ax in axes[len(plot_cols) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures / "key_numeric_distributions_by_default.png", dpi=160)
        plt.close(fig)


def save_curve_plots(
    y_test: pd.Series,
    score_map: dict[str, np.ndarray],
    paths: dict[str, Path],
    raw_score_map: dict[str, np.ndarray] | None = None,
):
    if not score_map:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in score_map.items():
        display_name = PLOT_MODEL_LABELS.get(name, name)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{display_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC Curves: Combined Features")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "roc_curves_combined.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in score_map.items():
        display_name = PLOT_MODEL_LABELS.get(name, name)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{display_name} (AP={ap:.3f})")
    ax.set_title("Precision-Recall Curves: Combined Features")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "precision_recall_curves_combined.png", dpi=160)
    plt.close(fig)

    save_probability_reliability_plot(
        y_test,
        score_map,
        paths,
        filename="calibration_curves_combined.png",
        table_filename="calibration_bins_combined.csv",
        title="Validation-Calibrated Reliability: Combined Features",
    )
    if raw_score_map:
        save_probability_reliability_plot(
            y_test,
            raw_score_map,
            paths,
            filename="calibration_curves_combined_raw.png",
            table_filename="calibration_bins_combined_raw.csv",
            title="Raw Probability Reliability: Combined Features",
        )

    all_deciles = []
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_rate = float(np.mean(y_test))
    for name, scores in score_map.items():
        display_name = PLOT_MODEL_LABELS.get(name, name)
        decile_table = make_lift_gain_table(y_test, scores, display_name)
        all_deciles.append(decile_table)
        ax.plot(
            decile_table["decile"],
            decile_table["cumulative_gain"],
            marker="o",
            label=display_name,
        )
    ax.plot([1, 10], [0.1, 1.0], linestyle="--", color="grey", linewidth=1, label="Random")
    ax.set_title(f"Cumulative Gain Curves: Combined Features (base rate={baseline_rate:.3f})")
    ax.set_xlabel("Score decile, highest risk first")
    ax.set_ylabel("Cumulative share of defaults captured")
    ax.set_ylim(0, 1.03)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "gain_curves_combined.png", dpi=160)
    plt.close(fig)

    if all_deciles:
        pd.concat(all_deciles, ignore_index=True).to_csv(
            paths["tables"] / "lift_gain_deciles_combined.csv",
            index=False,
        )


def make_calibration_bin_table(
    y_true: pd.Series,
    scores: np.ndarray,
    model_name: str,
    n_bins: int = 15,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "default": np.asarray(y_true, dtype=int),
            "score": np.clip(np.asarray(scores, dtype=float), 0, 1),
        }
    )
    table["bin"] = (
        pd.qcut(table["score"].rank(method="first"), q=n_bins, labels=False) + 1
    )
    grouped = (
        table.groupby("bin", as_index=False)
        .agg(
            observations=("default", "size"),
            defaults=("default", "sum"),
            mean_predicted_probability=("score", "mean"),
            observed_default_rate=("default", "mean"),
            minimum_predicted_probability=("score", "min"),
            maximum_predicted_probability=("score", "max"),
        )
        .sort_values("mean_predicted_probability")
    )
    z = 1.96
    n = grouped["observations"].astype(float)
    p = grouped["observed_default_rate"].astype(float)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    grouped["observed_rate_ci_lower"] = np.clip(centre - half_width, 0, 1)
    grouped["observed_rate_ci_upper"] = np.clip(centre + half_width, 0, 1)
    grouped.insert(0, "model", model_name)
    return grouped


def save_probability_reliability_plot(
    y_test: pd.Series,
    score_map: dict[str, np.ndarray],
    paths: dict[str, Path],
    filename: str,
    table_filename: str,
    title: str,
) -> None:
    all_bins: list[pd.DataFrame] = []
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for name, scores in score_map.items():
        display_name = PLOT_MODEL_LABELS.get(name, name)
        bins = make_calibration_bin_table(y_test, scores, display_name)
        all_bins.append(bins)
        ax.plot(
            bins["mean_predicted_probability"],
            bins["observed_default_rate"],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            label=display_name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(paths["figures"] / filename, dpi=180)
    plt.close(fig)
    pd.concat(all_bins, ignore_index=True).to_csv(
        paths["tables"] / table_filename,
        index=False,
    )


def make_lift_gain_table(
    y_true: pd.Series,
    scores: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    table = pd.DataFrame({"default": np.asarray(y_true), "score": np.asarray(scores)})
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table["decile"] = pd.qcut(
        np.arange(len(table)),
        q=n_bins,
        labels=np.arange(1, n_bins + 1),
    ).astype(int)
    grouped = (
        table.groupby("decile", as_index=False)
        .agg(
            loans=("default", "size"),
            defaults=("default", "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            avg_score=("score", "mean"),
        )
        .sort_values("decile")
    )
    total_defaults = grouped["defaults"].sum()
    base_rate = table["default"].mean()
    grouped["model"] = model_name
    grouped["default_rate"] = grouped["defaults"] / grouped["loans"]
    grouped["lift"] = grouped["default_rate"] / base_rate if base_rate > 0 else np.nan
    grouped["cumulative_defaults"] = grouped["defaults"].cumsum()
    grouped["cumulative_gain"] = grouped["cumulative_defaults"] / max(total_defaults, 1)
    grouped["cumulative_loans"] = grouped["loans"].cumsum()
    grouped["cumulative_population"] = grouped["cumulative_loans"] / len(table)
    return grouped[
        [
            "model",
            "decile",
            "loans",
            "defaults",
            "default_rate",
            "lift",
            "cumulative_gain",
            "cumulative_population",
            "min_score",
            "max_score",
            "avg_score",
        ]
    ]


def save_filtered_coefficients(
    coef: pd.DataFrame,
    paths: dict[str, Path],
    link: str,
    feature_set: str,
):
    filtered = coef[
        ~coef["feature"].str.contains("addr_state_", regex=False)
        & ~coef["feature"].str.contains("infrequent_sklearn", regex=False)
        & (coef["feature"] != "const")
    ].copy()
    if link == "logit" and "odds_ratio" not in filtered.columns:
        filtered["odds_ratio"] = np.exp(filtered["coefficient"].clip(-20, 20))
    filtered.head(40).to_csv(
        paths["tables"] / f"{link}_key_coefficients_{feature_set}.csv",
        index=False,
    )


def save_xgb_importance(pipe: Pipeline, test_x: pd.DataFrame, paths: dict[str, Path]):
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        booster = pipe.named_steps["model"]
        importances = booster.feature_importances_
        table = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(40)
        )
        table.to_csv(paths["tables"] / "xgboost_top_feature_importance_combined.csv", index=False)
    except Exception as exc:
        (paths["tables"] / "xgboost_importance_error.txt").write_text(str(exc), encoding="utf-8")


def save_rf_importance(pipe: Pipeline, paths: dict[str, Path]):
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        forest = pipe.named_steps["model"]
        importances = forest.feature_importances_
        table = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(40)
        )
        table.to_csv(
            paths["tables"] / "random_forest_top_feature_importance_combined.csv",
            index=False,
        )
    except Exception as exc:
        (paths["tables"] / "random_forest_importance_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def save_lgbm_importance(pipe: Pipeline, paths: dict[str, Path]):
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        model = pipe.named_steps["model"]
        importances = model.feature_importances_
        table = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(40)
        )
        table.to_csv(
            paths["tables"] / "lightgbm_top_feature_importance_combined.csv",
            index=False,
        )
    except Exception as exc:
        (paths["tables"] / "lightgbm_importance_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def save_catboost_importance(pipe: Pipeline, paths: dict[str, Path]):
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        model = pipe.named_steps["model"]
        importances = model.get_feature_importance()
        table = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(40)
        )
        table.to_csv(
            paths["tables"] / "catboost_top_feature_importance_combined.csv",
            index=False,
        )
    except Exception as exc:
        (paths["tables"] / "catboost_importance_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def save_best_tree_shap(
    model_name: str,
    pipe: Pipeline,
    test_x: pd.DataFrame,
    paths: dict[str, Path],
    sample_rows: int,
    random_state: int,
):
    try:
        import shap

        n_rows = min(max(int(sample_rows), 1), len(test_x))
        sample_x = test_x.sample(n=n_rows, random_state=random_state)
        transformed = pipe.named_steps["preprocess"].transform(sample_x)
        names = np.asarray(
            pipe.named_steps["preprocess"].get_feature_names_out(), dtype=object
        )
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        raw_values = explainer.shap_values(transformed)
        if isinstance(raw_values, list):
            raw_values = raw_values[-1]
        shap_values = np.asarray(
            raw_values.values if hasattr(raw_values, "values") else raw_values
        )
        if shap_values.ndim == 3:
            shap_values = (
                shap_values[:, :, 1]
                if shap_values.shape[-1] == 2
                else shap_values[:, :, 0]
            )
        if shap_values.shape != transformed.shape:
            raise ValueError(
                f"Unexpected SHAP shape {shap_values.shape}; expected {transformed.shape}."
            )

        top_table = (
            pd.DataFrame(
                {
                    "feature": names,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                    "mean_shap": shap_values.mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .head(40)
        )
        top_table.insert(0, "model", model_name)
        top_table.to_csv(
            paths["tables"] / "best_tree_shap_top_features_combined.csv", index=False
        )

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            transformed,
            feature_names=names,
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Summary: {model_name} (Combined Features)")
        plt.tight_layout()
        plt.savefig(paths["figures"] / "best_tree_shap_summary_combined.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            transformed,
            feature_names=names,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title(f"Mean Absolute SHAP: {model_name} (Combined Features)")
        plt.tight_layout()
        plt.savefig(paths["figures"] / "best_tree_shap_bar_combined.png", dpi=180)
        plt.close()

        (paths["tables"] / "best_tree_shap_metadata.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "sample_rows": n_rows,
                    "sample_source": "final_test_set",
                    "sampling": "random_without_replacement",
                    "random_state": random_state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        (paths["tables"] / "best_tree_shap_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def ebm_shape_coordinates(data: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(data["scores"], dtype=float)
    raw_names = np.asarray(data["names"], dtype=object)
    if len(raw_names) == len(scores) + 1:
        try:
            edges = raw_names.astype(float)
            x_values = (edges[:-1] + edges[1:]) / 2
        except (TypeError, ValueError):
            x_values = np.arange(len(scores), dtype=float)
    else:
        try:
            x_values = raw_names[: len(scores)].astype(float)
        except (TypeError, ValueError):
            x_values = np.arange(len(scores), dtype=float)
    lower = np.asarray(data.get("lower_bounds", np.full(len(scores), np.nan)), dtype=float)
    upper = np.asarray(data.get("upper_bounds", np.full(len(scores), np.nan)), dtype=float)
    return x_values, scores, lower, upper


def inverse_ebm_feature_axis(
    preprocessor: ColumnTransformer,
    feature_name: str,
    x_values: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    """Map EBM bin locations back through scaler/log1p to cleaned source units."""
    values = np.asarray(x_values, dtype=float)
    for transformer_name, transformer, source_columns in preprocessor.transformers_:
        if transformer_name not in {"num", "count"} or not isinstance(transformer, Pipeline):
            continue
        columns = list(source_columns)
        if feature_name not in columns:
            continue
        feature_index = columns.index(feature_name)
        scaler = transformer.named_steps["scaler"]
        original = values * float(scaler.scale_[feature_index]) + float(
            scaler.mean_[feature_index]
        )
        if transformer_name == "count":
            original = np.expm1(original)
        clipper = transformer.named_steps.get("winsor")
        if clipper is not None:
            lower = float(clipper.lower_bounds_[feature_index])
            upper = float(clipper.upper_bounds_[feature_index])
            original = np.clip(original, lower, upper)
        note = (
            "Original cleaned units after median imputation and p1-p99 winsorisation; "
            "EBM was fitted on the corresponding standardised values."
        )
        if transformer_name == "count":
            note = (
                "Original count scale after reversing standardisation and log1p; "
                "training values were median-imputed and p1-p99 winsorised."
            )
        return original, True, note
    return values, False, "Preprocessed model-input scale; no reliable inverse mapping available."


def format_ebm_original_axis(ax: plt.Axes, feature_name: str) -> None:
    if feature_name in {"loan_amnt", "annual_inc"}:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    elif feature_name == "fico_score" or feature_name in COUNT_FEATURES:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    elif feature_name in {"int_rate", "dti", "revol_util", "bc_util"}:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))


def save_ebm_explanations(pipe: Pipeline, paths: dict[str, Path], top_shapes: int = 6):
    try:
        preprocessor = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
        term_labels = []
        term_types = []
        for term in model.term_features_:
            label = " & ".join(str(feature_names[idx]) for idx in term)
            term_labels.append(label)
            term_types.append("main_effect" if len(term) == 1 else "interaction")
        importance = np.asarray(model.term_importances(), dtype=float)
        importance_table = pd.DataFrame(
            {
                "term": term_labels,
                "term_type": term_types,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        importance_table.to_csv(
            paths["tables"] / "ebm_global_feature_importance_combined.csv", index=False
        )

        top_importance = importance_table.head(20).sort_values("importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_importance["term"], top_importance["importance"], color="#4C78A8")
        ax.set_title("EBM Global Feature Importance: Combined Features")
        ax.set_xlabel("Mean absolute contribution")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "ebm_global_feature_importance_combined.png", dpi=180)
        plt.close(fig)

        main_effect_indices = [
            idx
            for idx in np.argsort(importance)[::-1]
            if len(model.term_features_[idx]) == 1
        ][:top_shapes]
        explanation = model.explain_global(name="EBM combined features")
        shape_rows = []
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = np.asarray(axes).ravel()
        for ax, term_idx in zip(axes, main_effect_indices):
            data = explanation.data(int(term_idx))
            label = term_labels[term_idx]
            x_preprocessed, scores, lower, upper = ebm_shape_coordinates(data)
            x_values, original_scale, transform_note = inverse_ebm_feature_axis(
                preprocessor, label, x_preprocessed
            )
            ax.plot(x_values, scores, color="#E45756", linewidth=2)
            if len(lower) == len(scores) and np.isfinite(lower).any():
                ax.fill_between(x_values, lower, upper, color="#E45756", alpha=0.18)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel(
                EBM_ORIGINAL_AXIS_LABELS.get(
                    label, f"{label} (original-scale bin midpoint)"
                )
                if original_scale
                else "Preprocessed feature value"
            )
            ax.set_ylabel("EBM contribution")
            if original_scale:
                format_ebm_original_axis(ax, label)
            for preprocessed, position, score, lo, hi in zip(
                x_preprocessed, x_values, scores, lower, upper
            ):
                shape_rows.append(
                    {
                        "term": label,
                        "x_value": position,
                        "x_preprocessed": preprocessed,
                        "axis_scale": "original_cleaned_units" if original_scale else "preprocessed",
                        "score": score,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "transformation_note": transform_note,
                    }
                )
        for ax in axes[len(main_effect_indices) :]:
            ax.axis("off")
        fig.suptitle("EBM Shape Functions for Key Variables")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "ebm_shape_functions_combined.png", dpi=180)
        plt.close(fig)
        pd.DataFrame(shape_rows).to_csv(
            paths["tables"] / "ebm_shape_function_points_combined.csv", index=False
        )

        main_effect_by_label = {
            term_labels[idx]: idx
            for idx, term in enumerate(model.term_features_)
            if len(term) == 1
        }
        requested_features = [
            feature
            for feature in EBM_ORIGINAL_SCALE_FEATURES
            if feature in main_effect_by_label
        ]
        original_rows = []
        original_fig, original_axes = plt.subplots(2, 3, figsize=(17, 9))
        original_axes = np.asarray(original_axes).ravel()
        for ax, feature in zip(original_axes, requested_features):
            term_idx = main_effect_by_label[feature]
            data = explanation.data(int(term_idx))
            x_preprocessed, scores, lower, upper = ebm_shape_coordinates(data)
            x_original, converted, transform_note = inverse_ebm_feature_axis(
                preprocessor, feature, x_preprocessed
            )
            if not converted:
                raise ValueError(f"Could not recover original EBM axis for {feature}.")
            ax.plot(x_original, scores, color="#2F6B9A", linewidth=2.2)
            if len(lower) == len(scores) and np.isfinite(lower).any():
                ax.fill_between(x_original, lower, upper, color="#2F6B9A", alpha=0.18)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(feature, fontsize=11)
            ax.set_xlabel(EBM_ORIGINAL_AXIS_LABELS[feature])
            ax.set_ylabel("EBM contribution to default log-odds")
            format_ebm_original_axis(ax, feature)
            for x_model, x_source, score, lo, hi in zip(
                x_preprocessed, x_original, scores, lower, upper
            ):
                original_rows.append(
                    {
                        "feature": feature,
                        "original_feature_value": x_source,
                        "preprocessed_feature_value": x_model,
                        "ebm_contribution_log_odds": score,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "transformation_note": transform_note,
                    }
                )
        for ax in original_axes[len(requested_features) :]:
            ax.axis("off")
        original_fig.suptitle(
            "EBM Shape Functions on Original Feature Scales\n"
            "Points are EBM bin midpoints; axes reflect cleaned p1-p99 winsorised training ranges",
            fontsize=15,
        )
        original_fig.tight_layout()
        original_fig.savefig(
            paths["figures"] / "ebm_shape_functions_original_scale_combined.png",
            dpi=180,
        )
        plt.close(original_fig)
        pd.DataFrame(original_rows).to_csv(
            paths["tables"] / "ebm_shape_function_points_original_scale_combined.csv",
            index=False,
        )
    except Exception as exc:
        (paths["tables"] / "ebm_explanation_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def save_ft_transformer_token_norm(pipe: Pipeline, paths: dict[str, Path]):
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        model = pipe.named_steps["model"].model_
        with torch.no_grad():
            token_norm = (
                model.feature_weight.detach().cpu().pow(2).sum(dim=1).sqrt().numpy()
            )
        table = (
            pd.DataFrame({"feature": names, "token_embedding_norm": token_norm})
            .sort_values("token_embedding_norm", ascending=False)
            .head(40)
        )
        table.to_csv(
            paths["tables"] / "ft_transformer_token_norm_combined.csv",
            index=False,
        )
    except Exception as exc:
        (paths["tables"] / "ft_transformer_token_norm_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def save_neural_training_history(
    pipe: Pipeline,
    model_slug: str,
    paths: dict[str, Path],
) -> None:
    model = pipe.named_steps["model"]
    history = pd.DataFrame(getattr(model, "training_history_", []))
    if not history.empty:
        history.insert(0, "model", model_slug)
        history["training_rows"] = int(getattr(model, "fit_train_rows_", 0))
        history["training_device"] = str(
            getattr(model, "training_device_name_", "CPU")
        )
        history["selected_epoch"] = history["valid_auc"].eq(
            history["valid_auc"].max()
        )
        history.to_csv(
            paths["tables"] / f"{model_slug}_training_history_combined.csv",
            index=False,
        )


def run_models(
    df: pd.DataFrame,
    paths: dict[str, Path],
    split: str,
    validation_size: float,
    test_size: float,
    random_state: int,
    run_id: str,
    max_statmodels_rows: int,
    glm_inference_rows: int,
    skip_mlp: bool,
    skip_ft_transformer: bool,
    ft_max_train_rows: int,
    ft_epochs: int,
    ft_batch_size: int,
    tune_tree_models: bool,
    tuning_max_rows: int,
    skip_lightgbm: bool,
    skip_catboost: bool,
    skip_ebm: bool,
    shap_sample_rows: int,
) -> pd.DataFrame:
    borrower = available_features(df, BORROWER_FEATURES)
    risk_assessment = available_features(df, RISK_ASSESSMENT_FEATURES)
    financial = available_features(df, FINANCIAL_FEATURES)
    borrower_financial = borrower + [col for col in financial if col not in borrower]
    feature_sets = {
        "borrower_only": borrower,
        "financial_only": financial,
        "grade_only": risk_assessment,
        "borrower_financial": borrower_financial,
        "combined": borrower_financial
        + [col for col in risk_assessment if col not in borrower_financial],
    }

    split_features = feature_sets["combined"]
    (
        train_x_all,
        validation_x_all,
        test_x_all,
        train_y,
        validation_y,
        test_y,
        split_meta,
    ) = train_validation_test_frames(
        df,
        split_features,
        split,
        validation_size,
        test_size,
        random_state,
    )
    split_meta["run_id"] = run_id
    (paths["tables"] / "split_metadata.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    partition_rows = []
    for partition_name, target in (
        ("train", train_y),
        ("validation", validation_y),
        ("test", test_y),
    ):
        defaults = int(target.sum())
        rows = int(len(target))
        partition_rows.append(
            {
                "run_id": run_id,
                "split": split,
                "partition": partition_name,
                "rows": rows,
                "defaults": defaults,
                "non_defaults": rows - defaults,
                "default_prevalence": defaults / max(rows, 1),
                "date_min": split_meta.get(f"{partition_name}_date_min", ""),
                "date_max": split_meta.get(f"{partition_name}_date_max", ""),
            }
        )
    pd.DataFrame(partition_rows).to_csv(
        paths["tables"] / "partition_composition.csv", index=False
    )

    metrics: list[dict[str, object]] = []
    tuning_reports: list[pd.DataFrame] = []
    combined_scores: dict[str, np.ndarray] = {}
    combined_raw_scores: dict[str, np.ndarray] = {}

    def register_combined_scores(
        model_name: str,
        calibrated_scores: np.ndarray,
        metric: dict[str, object],
    ) -> None:
        combined_scores[model_name] = np.asarray(calibrated_scores, dtype=float)
        combined_raw_scores[model_name] = recover_raw_scores(calibrated_scores, metric)

    combined_y_test = test_y
    combined_xgb_pipe = None
    combined_rf_pipe = None
    combined_mlp_pipe = None
    combined_ft_pipe = None
    combined_lgbm_pipe = None
    combined_catboost_pipe = None
    combined_ebm_pipe = None

    for feature_set, features in feature_sets.items():
        print(f"\n=== Feature set: {feature_set} ({len(features)} features) ===", flush=True)
        train_x = train_x_all[features]
        validation_x = validation_x_all[features]
        test_x = test_x_all[features]

        glm_models = [("Logistic Regression", "logit")]
        if feature_set == "combined":
            glm_models.append(("Probit Regression", "probit"))
        for model_name, link in glm_models:
            print(f"Fitting {model_name} / {feature_set}", flush=True)
            row, coef, uncertainty, scores = fit_glm(
                model_name,
                link,
                feature_set,
                features,
                train_x,
                validation_x,
                test_x,
                train_y,
                validation_y,
                test_y,
                max_statmodels_rows,
                glm_inference_rows,
                random_state,
            )
            metrics.append(row)
            coef.to_csv(
                paths["tables"]
                / f"{link}_coefficients_{feature_set}.csv",
                index=False,
            )
            if not uncertainty.empty:
                uncertainty.to_csv(
                    paths["tables"]
                    / f"{link}_supplementary_uncertainty_{feature_set}.csv",
                    index=False,
                )
            save_filtered_coefficients(coef, paths, link, feature_set)
            if feature_set == "combined":
                register_combined_scores(model_name, scores, row)

        if HAS_XGBOOST and feature_set == "combined":
            neg = int((train_y == 0).sum())
            pos = int((train_y == 1).sum())
            scale_pos_weight = neg / max(pos, 1)
            xgb = XGBClassifier(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            )
            if tune_tree_models:
                print(f"Tuning XGBoost / {feature_set}", flush=True)
                xgb_grid = [
                    {
                        "n_estimators": 120,
                        "max_depth": 3,
                        "learning_rate": 0.05,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                    },
                    {
                        "n_estimators": 160,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                    },
                    {
                        "n_estimators": 220,
                        "max_depth": 3,
                        "learning_rate": 0.03,
                        "subsample": 0.90,
                        "colsample_bytree": 0.90,
                    },
                    {
                        "n_estimators": 180,
                        "max_depth": 5,
                        "learning_rate": 0.04,
                        "subsample": 0.80,
                        "colsample_bytree": 0.80,
                    },
                ]
                xgb, best_params, tuning_table = tune_estimator_on_validation(
                    "XGBoost",
                    xgb,
                    xgb_grid,
                    feature_set,
                    features,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    random_state,
                    tuning_max_rows,
                )
                tuning_table["selected"] = tuning_table.apply(
                    lambda row: all(row.get(k) == v for k, v in best_params.items()),
                    axis=1,
                )
                tuning_reports.append(tuning_table)
            print(f"Fitting XGBoost / {feature_set}", flush=True)
            row, pipe, scores = fit_sklearn_model(
                "XGBoost",
                xgb,
                feature_set,
                features,
                train_x,
                validation_x,
                test_x,
                train_y,
                validation_y,
                test_y,
                sample_weight=False,
            )
            metrics.append(row)
            if feature_set == "combined":
                register_combined_scores("XGBoost", scores, row)
                combined_xgb_pipe = pipe
        elif feature_set == "combined":
            print("Skipping XGBoost because the xgboost package is not available.", flush=True)

        if HAS_LIGHTGBM and not skip_lightgbm:
            try:
                lgbm = LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=50,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
                if tune_tree_models:
                    print(f"Tuning LightGBM / {feature_set}", flush=True)
                    lgbm_grid = [
                        {
                            "n_estimators": 300,
                            "learning_rate": 0.05,
                            "num_leaves": 31,
                            "min_child_samples": 50,
                            "subsample": 0.85,
                            "colsample_bytree": 0.85,
                        },
                        {
                            "n_estimators": 500,
                            "learning_rate": 0.03,
                            "num_leaves": 31,
                            "min_child_samples": 80,
                            "subsample": 0.90,
                            "colsample_bytree": 0.90,
                        },
                        {
                            "n_estimators": 350,
                            "learning_rate": 0.05,
                            "num_leaves": 63,
                            "min_child_samples": 100,
                            "subsample": 0.80,
                            "colsample_bytree": 0.80,
                        },
                    ]
                    lgbm, best_params, tuning_table = tune_estimator_on_validation(
                        "LightGBM",
                        lgbm,
                        lgbm_grid,
                        feature_set,
                        features,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        random_state,
                        tuning_max_rows,
                    )
                    tuning_table["selected"] = tuning_table.apply(
                        lambda row: all(row.get(k) == v for k, v in best_params.items()),
                        axis=1,
                    )
                    tuning_reports.append(tuning_table)
                print(f"Fitting LightGBM / {feature_set}", flush=True)
                row, pipe, scores = fit_sklearn_model(
                    "LightGBM",
                    lgbm,
                    feature_set,
                    features,
                    train_x,
                    validation_x,
                    test_x,
                    train_y,
                    validation_y,
                    test_y,
                    sample_weight=False,
                )
                metrics.append(row)
                if feature_set == "combined":
                    register_combined_scores("LightGBM", scores, row)
                    combined_lgbm_pipe = pipe
            except Exception as exc:
                (paths["tables"] / f"lightgbm_error_{feature_set}.txt").write_text(
                    str(exc), encoding="utf-8"
                )
        elif not skip_lightgbm:
            print("Skipping LightGBM because the lightgbm package is not available.", flush=True)

        if HAS_CATBOOST and not skip_catboost and feature_set == "combined":
            try:
                catboost = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    iterations=350,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=random_state,
                    auto_class_weights="Balanced",
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=-1,
                )
                if tune_tree_models:
                    print(f"Tuning CatBoost / {feature_set}", flush=True)
                    catboost_grid = [
                        {
                            "iterations": 300,
                            "learning_rate": 0.05,
                            "depth": 4,
                            "l2_leaf_reg": 3.0,
                        },
                        {
                            "iterations": 350,
                            "learning_rate": 0.05,
                            "depth": 6,
                            "l2_leaf_reg": 3.0,
                        },
                        {
                            "iterations": 500,
                            "learning_rate": 0.03,
                            "depth": 6,
                            "l2_leaf_reg": 5.0,
                        },
                    ]
                    catboost, best_params, tuning_table = tune_estimator_on_validation(
                        "CatBoost",
                        catboost,
                        catboost_grid,
                        feature_set,
                        features,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        random_state,
                        tuning_max_rows,
                    )
                    tuning_table["selected"] = tuning_table.apply(
                        lambda row: all(row.get(k) == v for k, v in best_params.items()),
                        axis=1,
                    )
                    tuning_reports.append(tuning_table)
                print(f"Fitting CatBoost / {feature_set}", flush=True)
                row, pipe, scores = fit_sklearn_model(
                    "CatBoost",
                    catboost,
                    feature_set,
                    features,
                    train_x,
                    validation_x,
                    test_x,
                    train_y,
                    validation_y,
                    test_y,
                    sample_weight=False,
                )
                metrics.append(row)
                if feature_set == "combined":
                    register_combined_scores("CatBoost", scores, row)
                    combined_catboost_pipe = pipe
            except Exception as exc:
                (paths["tables"] / f"catboost_error_{feature_set}.txt").write_text(
                    str(exc), encoding="utf-8"
                )
        elif not skip_catboost and feature_set == "combined":
            print("Skipping CatBoost because the catboost package is not available.", flush=True)

        if HAS_EBM and not skip_ebm:
            try:
                ebm = ExplainableBoostingClassifier(
                    interactions=10,
                    max_bins=256,
                    learning_rate=0.015,
                    max_rounds=2_000,
                    outer_bags=4,
                    early_stopping_rounds=75,
                    validation_size=0.15,
                    n_jobs=1,
                    random_state=random_state,
                )
                if tune_tree_models:
                    print(f"Tuning EBM / {feature_set}", flush=True)
                    ebm_grid = [
                        {
                            "interactions": 0,
                            "learning_rate": 0.015,
                            "max_rounds": 1_500,
                            "outer_bags": 4,
                        },
                        {
                            "interactions": 10,
                            "learning_rate": 0.015,
                            "max_rounds": 2_000,
                            "outer_bags": 4,
                        },
                        {
                            "interactions": 10,
                            "learning_rate": 0.01,
                            "max_rounds": 2_500,
                            "outer_bags": 4,
                        },
                    ]
                    ebm, best_params, tuning_table = tune_estimator_on_validation(
                        "Explainable Boosting Machine",
                        ebm,
                        ebm_grid,
                        feature_set,
                        features,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        random_state,
                        min(tuning_max_rows, 12_000),
                    )
                    tuning_table["selected"] = tuning_table.apply(
                        lambda row: all(row.get(k) == v for k, v in best_params.items()),
                        axis=1,
                    )
                    tuning_reports.append(tuning_table)
                print(f"Fitting EBM / {feature_set}", flush=True)
                row, pipe, scores = fit_sklearn_model(
                    "Explainable Boosting Machine",
                    ebm,
                    feature_set,
                    features,
                    train_x,
                    validation_x,
                    test_x,
                    train_y,
                    validation_y,
                    test_y,
                    sample_weight=False,
                )
                metrics.append(row)
                if feature_set == "combined":
                    register_combined_scores("Explainable Boosting Machine", scores, row)
                    combined_ebm_pipe = pipe
            except Exception as exc:
                (paths["tables"] / f"ebm_error_{feature_set}.txt").write_text(
                    str(exc), encoding="utf-8"
                )
        elif not skip_ebm:
            print("Skipping EBM because interpret-core is not available.", flush=True)

        if feature_set != "combined":
            continue

        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        if tune_tree_models:
            print(f"Tuning Random Forest / {feature_set}", flush=True)
            rf_grid = [
                {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_leaf": 20,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 250,
                    "max_depth": 14,
                    "min_samples_leaf": 30,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_leaf": 50,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 250,
                    "max_depth": 18,
                    "min_samples_leaf": 20,
                    "max_features": "log2",
                },
            ]
            rf, best_params, tuning_table = tune_estimator_on_validation(
                "Random Forest",
                rf,
                rf_grid,
                feature_set,
                features,
                train_x,
                train_y,
                validation_x,
                validation_y,
                random_state,
                tuning_max_rows,
            )
            tuning_table["selected"] = tuning_table.apply(
                lambda row: all(
                    (pd.isna(row.get(k)) and v is None) or row.get(k) == v
                    for k, v in best_params.items()
                ),
                axis=1,
            )
            tuning_reports.append(tuning_table)
        print(f"Fitting Random Forest / {feature_set}", flush=True)
        row, pipe, scores = fit_sklearn_model(
            "Random Forest",
            rf,
            feature_set,
            features,
            train_x,
            validation_x,
            test_x,
            train_y,
            validation_y,
            test_y,
            sample_weight=False,
        )
        metrics.append(row)
        if feature_set == "combined":
            register_combined_scores("Random Forest", scores, row)
            combined_rf_pipe = pipe

        svm = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=5_000,
        )
        print(f"Fitting Linear SVM / {feature_set}", flush=True)
        row, pipe, scores = fit_sklearn_model(
            "Linear SVM",
            svm,
            feature_set,
            features,
            train_x,
            validation_x,
            test_x,
            train_y,
            validation_y,
            test_y,
            sample_weight=False,
        )
        metrics.append(row)
        if feature_set == "combined":
            register_combined_scores("Linear SVM", scores, row)

        if not skip_mlp:
            if HAS_TORCH:
                mlp = TorchMLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    dropout=0.10,
                    batch_norm=True,
                    lr=1e-3,
                    weight_decay=1e-4,
                    batch_size=512,
                    max_epochs=30,
                    patience=5,
                    validation_fraction=0.15,
                    max_train_rows=ft_max_train_rows,
                    calibrate=False,
                    random_state=random_state,
                )
                if tune_tree_models:
                    print(f"Tuning BatchNorm MLP / {feature_set}", flush=True)
                    mlp_grid = [
                        {
                            "hidden_layer_sizes": (128, 64),
                            "dropout": 0.10,
                            "lr": 1e-3,
                            "weight_decay": 1e-4,
                            "batch_size": 512,
                        },
                        {
                            "hidden_layer_sizes": (128, 64),
                            "dropout": 0.15,
                            "lr": 5e-4,
                            "weight_decay": 1e-4,
                            "batch_size": 256,
                        },
                    ]
                    mlp, best_params, tuning_table = tune_estimator_on_validation(
                        "MLP Neural Network",
                        mlp,
                        mlp_grid,
                        feature_set,
                        features,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        random_state,
                        min(tuning_max_rows, 12_000),
                    )
                    tuning_table["selected"] = tuning_table.apply(
                        lambda row: all(row.get(k) == v for k, v in best_params.items()),
                        axis=1,
                    )
                    tuning_reports.append(tuning_table)
            else:
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=60,
                    random_state=random_state,
                )
            print(f"Fitting MLP / {feature_set}", flush=True)
            row, pipe, scores = fit_sklearn_model(
                "MLP Neural Network",
                mlp,
                feature_set,
                features,
                train_x,
                validation_x,
                test_x,
                train_y,
                validation_y,
                test_y,
                sample_weight=False,
            )
            metrics.append(row)
            if feature_set == "combined":
                register_combined_scores("MLP Neural Network", scores, row)
                combined_mlp_pipe = pipe

        if not skip_ft_transformer:
            if HAS_TORCH:
                ft = FTTransformerClassifier(
                    d_token=32,
                    n_heads=4,
                    n_layers=2,
                    dropout=0.10,
                    lr=1e-3,
                    weight_decay=1e-4,
                    batch_size=ft_batch_size,
                    max_epochs=ft_epochs,
                    patience=3,
                    validation_fraction=0.15,
                    max_train_rows=ft_max_train_rows,
                    calibrate=False,
                    random_state=random_state,
                )
                if tune_tree_models:
                    print(f"Tuning FT-Transformer / {feature_set}", flush=True)
                    ft_grid = [
                        {
                            "d_token": 32,
                            "n_heads": 4,
                            "n_layers": 2,
                            "dropout": 0.10,
                            "lr": 1e-3,
                            "weight_decay": 1e-4,
                            "batch_size": 512,
                        },
                        {
                            "d_token": 32,
                            "n_heads": 4,
                            "n_layers": 2,
                            "dropout": 0.15,
                            "lr": 5e-4,
                            "weight_decay": 1e-4,
                            "batch_size": 256,
                        },
                    ]
                    ft, best_params, tuning_table = tune_estimator_on_validation(
                        "FT-Transformer",
                        ft,
                        ft_grid,
                        feature_set,
                        features,
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        random_state,
                        min(tuning_max_rows, 12_000),
                    )
                    tuning_table["selected"] = tuning_table.apply(
                        lambda row: all(row.get(k) == v for k, v in best_params.items()),
                        axis=1,
                    )
                    tuning_reports.append(tuning_table)
                print(f"Fitting FT-Transformer / {feature_set}", flush=True)
                row, pipe, scores = fit_sklearn_model(
                    "FT-Transformer",
                    ft,
                    feature_set,
                    features,
                    train_x,
                    validation_x,
                    test_x,
                    train_y,
                    validation_y,
                    test_y,
                    sample_weight=False,
                )
                metrics.append(row)
                if feature_set == "combined":
                    register_combined_scores("FT-Transformer", scores, row)
                    combined_ft_pipe = pipe
            else:
                print(
                    "Skipping FT-Transformer because PyTorch is not available.",
                    flush=True,
                )

    result = pd.DataFrame(metrics)
    result.insert(0, "run_id", run_id)
    result.insert(1, "split", split)
    result = result.sort_values(["feature_set", "roc_auc"], ascending=[True, False])
    result.to_csv(paths["tables"] / "model_metrics.csv", index=False)
    rq1 = result.loc[result["feature_set"] == "combined"].copy()
    rq1.to_csv(paths["tables"] / "rq1_combined_model_metrics.csv", index=False)
    rq2_models = {
        "Logistic Regression",
        "LightGBM",
        "Explainable Boosting Machine",
    }
    rq2 = result.loc[result["model"].isin(rq2_models)].copy()
    rq2["feature_group_label"] = rq2["feature_set"].map(RQ2_FEATURE_SET_LABELS)
    rq2.to_csv(paths["tables"] / "rq2_feature_group_comparison.csv", index=False)
    if not rq2.empty:
        feature_order = [
            RQ2_FEATURE_SET_LABELS["borrower_only"],
            RQ2_FEATURE_SET_LABELS["financial_only"],
            RQ2_FEATURE_SET_LABELS["grade_only"],
            RQ2_FEATURE_SET_LABELS["borrower_financial"],
            RQ2_FEATURE_SET_LABELS["combined"],
        ]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.pointplot(
            data=rq2,
            x="feature_group_label",
            y="roc_auc",
            hue="model",
            order=feature_order,
            markers=["o", "s", "D"],
            ax=ax,
        )
        ax.set_title("Predictive Performance by Feature Group")
        ax.set_xlabel("")
        ax.set_ylabel("Test ROC AUC")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Model", loc="best")
        fig.tight_layout()
        fig.savefig(paths["figures"] / "rq2_feature_group_auc_comparison.png", dpi=180)
        plt.close(fig)
    if tuning_reports:
        tuning_all = pd.concat(tuning_reports, ignore_index=True)
        tuning_all.insert(0, "run_id", run_id)
        tuning_all.insert(1, "split", split)
        tuning_all["selected"] = False
        selected_idx = tuning_all.groupby(
            ["model", "feature_set"], sort=False
        )["validation_roc_auc"].idxmax()
        tuning_all.loc[selected_idx, "selected"] = True
        tuning_all.to_csv(
            paths["tables"] / "tree_model_tuning_results.csv", index=False
        )
        non_parameter_columns = {
            "run_id",
            "split",
            "feature_set",
            "model",
            "validation_roc_auc",
            "validation_avg_precision",
            "tuning_train_rows",
            "validation_rows",
            "validation_source",
            "selected",
        }
        parameter_columns = [
            column
            for column in tuning_all.columns
            if column not in non_parameter_columns
        ]
        selected_rows = tuning_all.loc[selected_idx].copy()
        selected_rows["selected_configuration"] = selected_rows.apply(
            lambda row: json.dumps(
                {
                    column: row[column]
                    for column in parameter_columns
                    if pd.notna(row[column])
                },
                sort_keys=True,
                default=str,
            ),
            axis=1,
        )
        selected_rows[
            [
                "run_id",
                "split",
                "model",
                "feature_set",
                "selected_configuration",
                "validation_roc_auc",
                "validation_avg_precision",
                "tuning_train_rows",
                "validation_rows",
                "validation_source",
            ]
        ].to_csv(
            paths["tables"] / "selected_model_configurations.csv", index=False
        )
    save_curve_plots(
        combined_y_test,
        combined_scores,
        paths,
        raw_score_map=combined_raw_scores,
    )
    if combined_xgb_pipe is not None:
        save_xgb_importance(combined_xgb_pipe, test_x_all[feature_sets["combined"]], paths)
    if combined_rf_pipe is not None:
        save_rf_importance(combined_rf_pipe, paths)
    if combined_lgbm_pipe is not None:
        save_lgbm_importance(combined_lgbm_pipe, paths)
    if combined_catboost_pipe is not None:
        save_catboost_importance(combined_catboost_pipe, paths)
    if combined_ebm_pipe is not None:
        save_ebm_explanations(combined_ebm_pipe, paths)
    if combined_ft_pipe is not None:
        save_ft_transformer_token_norm(combined_ft_pipe, paths)
        save_neural_training_history(combined_ft_pipe, "ft_transformer", paths)
    if combined_mlp_pipe is not None:
        save_neural_training_history(combined_mlp_pipe, "mlp", paths)

    predictions = pd.DataFrame(
        {
            "test_position": np.arange(len(combined_y_test)),
            "actual_default": combined_y_test.to_numpy(dtype=int),
        }
    )
    for model_name, scores in combined_scores.items():
        predictions[model_name] = np.asarray(scores, dtype=float)
    predictions.to_csv(
        paths["tables"] / "combined_test_predictions.csv",
        index=False,
    )
    raw_predictions = pd.DataFrame(
        {
            "test_position": np.arange(len(combined_y_test)),
            "actual_default": combined_y_test.to_numpy(dtype=int),
        }
    )
    for model_name, scores in combined_raw_scores.items():
        raw_predictions[model_name] = np.asarray(scores, dtype=float)
    raw_predictions.to_csv(
        paths["tables"] / "combined_test_predictions_raw.csv",
        index=False,
    )
    calibration_columns = [
        "run_id",
        "split",
        "feature_set",
        "model",
        "probability_calibration",
        "calibration_slope",
        "calibration_intercept",
        "raw_validation_brier_score",
        "validation_brier_score",
        "raw_brier_score",
        "brier_score",
        "raw_probability_mean",
        "calibrated_probability_mean",
    ]
    result.loc[:, calibration_columns].to_csv(
        paths["tables"] / "probability_calibration_parameters.csv",
        index=False,
    )
    tree_pipes = {
        "XGBoost": combined_xgb_pipe,
        "LightGBM": combined_lgbm_pipe,
        "CatBoost": combined_catboost_pipe,
        "Random Forest": combined_rf_pipe,
    }
    available_tree_pipes = {name: pipe for name, pipe in tree_pipes.items() if pipe is not None}
    if available_tree_pipes:
        tree_rows = rq1.loc[rq1["model"].isin(available_tree_pipes)]
        if not tree_rows.empty:
            # Select the model to explain without consulting final test performance.
            best_tree_name = tree_rows.sort_values("validation_roc_auc", ascending=False).iloc[0]["model"]
            save_best_tree_shap(
                best_tree_name,
                available_tree_pipes[best_tree_name],
                test_x_all[feature_sets["combined"]],
                paths,
                shap_sample_rows,
                random_state,
            )
    return result


def write_experiment_summary(
    run_root: Path,
    run_id: str,
    metadata: dict[str, object],
    representativeness: dict[str, object],
    results: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# LendingClub Unified 500,000-Loan Experiment",
        "",
        f"Run ID: `{run_id}`",
        "",
        "## Design",
        "",
        f"- Completed-loan population after cleaning/filtering: {metadata['rows_after_term_filter']:,}",
        f"- Modelling sample: {metadata['model_sample_rows']:,}",
        "- Sampling: proportional stratified random sampling by default and issue year",
        f"- Random state: {metadata['random_state']}",
        "- Split shares: 64% train / 16% validation / 20% test",
        "- Random split: main comparison",
        "- Temporal split: out-of-time robustness check",
        "- RQ1: ten-model comparison using Combined features.",
        "- RQ2: Logistic Regression, LightGBM, and EBM across five feature groups.",
        "- The outer data, validation, calibration, and evaluation protocol is common across estimators; imbalance controls use prespecified estimator-appropriate mechanisms rather than forcing class weights onto every model.",
        f"- RQ2 grouping definition: `{metadata['rq2_grouping_version']}`. "
        "FICO score and credit-history length are assigned to the Financial/Credit block; "
        "the Borrower block contains borrower and application characteristics.",
        f"- RQ3 SHAP sample: {metadata['shap_sample_rows']:,} randomly selected final-test observations.",
        f"- Compute backend: {metadata['compute_backend']}",
        "- Logistic predictive probabilities, main interpretation coefficients, and odds ratios are extracted from the same full-training fitted model.",
        "- Predictive Logistic and Probit models use the complete training partition; no separate subsample supplies the main coefficient interpretation.",
        "- MLP and FT-Transformer use the complete training partition; validation-based early stopping controls epochs, not sample size.",
        "- The test set was used only for final evaluation; validation selected parameters, epochs, and thresholds.",
        "- Hyperparameters, thresholds, and calibration maps were selected separately within the random and temporal designs using their respective validation partitions.",
        "- Every model first produces its raw score or native probability. A common sigmoid map is then fitted on that model's validation predictions and applied unchanged to its final-test predictions. Main Brier scores and reliability curves therefore use a uniform validation-based calibration protocol; raw probability outputs are retained separately for audit and appendix comparison.",
        "",
        "## Sample Representativeness",
        "",
        f"- Maximum categorical share difference: {representativeness.get('maximum_categorical_share_difference', float('nan')):.6f}",
        f"- Maximum absolute standardized mean difference: {representativeness.get('maximum_absolute_standardized_mean_difference', float('nan')):.6f}",
        f"- Maximum numerical KS statistic: {representativeness.get('maximum_numerical_ks_statistic', float('nan')):.6f}",
        f"- Maximum default-by-issue-year stratum share difference: {representativeness.get('maximum_stratum_share_difference', float('nan')):.6f}",
        "",
    ]
    for split_name, result in results.items():
        split_title = split_name.replace("_", " ").title()
        lines.extend([f"## RQ1: {split_title} Combined-Feature Results", ""])
        combined = result.loc[result["feature_set"] == "combined"].sort_values(
            "roc_auc", ascending=False
        )
        lines.extend(
            [
                "| Model | AUC | Gini | KS | AP | Brier | Precision | Recall | F1 | Threshold |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in combined.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.roc_auc:.4f} | {row.gini:.4f} | "
                f"{row.ks_statistic:.4f} | {row.avg_precision:.4f} | "
                f"{row.brier_score:.4f} | {row.precision:.4f} | {row.recall:.4f} | "
                f"{row.f1:.4f} | {row.threshold:.4f} |"
            )
        lines.extend(["", f"## RQ2: {split_title} Feature-Group Comparison", ""])
        rq2 = result.loc[
            result["model"].isin(
                [
                    "Logistic Regression",
                    "LightGBM",
                    "Explainable Boosting Machine",
                ]
            )
        ].sort_values(["model", "feature_set"])
        rq2 = rq2.assign(
            feature_group_label=rq2["feature_set"].map(RQ2_FEATURE_SET_LABELS)
        )
        lines.extend(
            [
                "| Model | Feature Group | AUC | AP | Brier | F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rq2.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.feature_group_label} | {row.roc_auc:.4f} | "
                f"{row.avg_precision:.4f} | {row.brier_score:.4f} | {row.f1:.4f} |"
            )
        tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
        best_tree = combined.loc[combined["model"].isin(tree_models)].sort_values(
            "validation_roc_auc", ascending=False
        )
        if not best_tree.empty:
            lines.extend(
                [
                    "",
                    f"## RQ3: {split_title} Interpretability",
                    "",
                    f"- Tree-based model selected for SHAP by validation AUC: {best_tree.iloc[0]['model']} "
                    f"(validation AUC {best_tree.iloc[0]['validation_roc_auc']:.4f}; "
                    f"final-test AUC {best_tree.iloc[0]['roc_auc']:.4f}).",
                    "- Logistic Regression coefficients and odds ratios are extracted directly from the same full-training fitted model used for prediction.",
                    "- SHAP summary, SHAP bar plot, and top-feature table use only the configured random sample from the final test set.",
                    "- EBM global importance and key-variable shape functions are in the split figures and tables directories.",
                    "- Dedicated EBM plots for int_rate, dti, loan_amnt, annual_inc, and fico_score use original cleaned feature units rather than standardised model-input values.",
                ]
            )
        lines.append("")

    random_result = results.get("random")
    temporal_result = results.get("temporal")
    random_combined = (
        random_result.loc[random_result["feature_set"] == "combined"].copy()
        if random_result is not None
        else pd.DataFrame()
    )
    temporal_combined = (
        temporal_result.loc[temporal_result["feature_set"] == "combined"].copy()
        if temporal_result is not None
        else pd.DataFrame()
    )

    lines.extend(["## RQ3: Integrated Interpretability Findings", ""])
    shap_path = (
        run_root
        / "random_split"
        / "tables"
        / "best_tree_shap_top_features_combined.csv"
    )
    if shap_path.exists():
        shap_table = pd.read_csv(shap_path).head(10)
        lines.extend(
            [
                "### SHAP top-feature interpretation",
                "",
                "The final-test SHAP sample identifies the following leading contributors to the best tree-based model:",
                "",
                "| Rank | Feature | Mean absolute SHAP |",
                "|---:|---|---:|",
            ]
        )
        for rank, row in enumerate(shap_table.itertuples(index=False), start=1):
            lines.append(f"| {rank} | {row.feature} | {row.mean_abs_shap:.4f} |")
        lines.extend(
            [
                "",
                "Higher interest rates, longer terms, higher DTI, and more recently opened accounts generally increase predicted risk, while higher FICO scores reduce it. These are predictive associations, not causal effects.",
                "",
            ]
        )

    ebm_path = (
        run_root
        / "random_split"
        / "tables"
        / "ebm_global_importance.csv"
    )
    if ebm_path.exists():
        ebm_table = pd.read_csv(ebm_path).head(10)
        lines.extend(
            [
                "### EBM global importance and shape functions",
                "",
                "| Rank | Feature or term | Importance |",
                "|---:|---|---:|",
            ]
        )
        for row in ebm_table.itertuples(index=False):
            lines.append(f"| {int(row.rank)} | {row.feature} | {row.importance:.4f} |")
        lines.extend(
            [
                "",
                "The original-scale EBM plots show increasing risk contributions for DTI and loan amount over much of their ranges, and a decreasing contribution as FICO score rises. Interest-rate and annual-income functions are more locally non-linear and should not be interpreted causally.",
                "",
            ]
        )

    logistic_path = (
        run_root
        / "random_split"
        / "tables"
        / "logistic_key_coefficients_for_interpretation.csv"
    )
    if logistic_path.exists():
        logistic_table = pd.read_csv(logistic_path)
        lines.extend(
            [
                "### Logistic Regression coefficients and odds ratios",
                "",
                "The coefficients and odds ratios below are extracted directly from the same full-training fitted Logistic model that generated the reported probabilities and performance metrics.",
                "",
                "Numeric coefficients use the model's standardised, winsorised input scale; odds ratios therefore represent a one-standard-deviation increase in the cleaned feature, holding other variables constant.",
                "",
                "| Feature | Coefficient | Odds ratio | Sign | Direction |",
                "|---|---:|---:|---|---|",
            ]
        )
        for row in logistic_table.itertuples(index=False):
            lines.append(
                f"| {row.feature} | {row.coefficient:.4f} | {row.odds_ratio:.4f} | "
                f"{row.sign} | {row.interpretation_direction} |"
            )
        lines.append("")

    if not random_combined.empty and not temporal_combined.empty:
        random_best = random_combined.sort_values("roc_auc", ascending=False).iloc[0]
        random_logit = random_combined.loc[
            random_combined["model"] == "Logistic Regression"
        ].iloc[0]
        temporal_best = temporal_combined.sort_values("roc_auc", ascending=False).iloc[0]
        temporal_logit = temporal_combined.loc[
            temporal_combined["model"] == "Logistic Regression"
        ].iloc[0]
        lines.extend(
            [
                "## Run-Level Model Comparison",
                "",
                f"The nominally highest random-split model was {random_best['model']} (AUC {random_best['roc_auc']:.4f}), compared with Logistic Regression at {random_logit['roc_auc']:.4f}, a small absolute AUC gain of {random_best['roc_auc'] - random_logit['roc_auc']:.4f}. "
                f"Under the temporal split, the corresponding difference was {temporal_best['roc_auc'] - temporal_logit['roc_auc']:.4f} ({temporal_best['roc_auc']:.4f} versus {temporal_logit['roc_auc']:.4f}). Small differences between leading models should not be overinterpreted without paired uncertainty assessment.",
                "",
                "All reported Brier scores and reliability curves use model-specific sigmoid maps fitted on validation predictions and applied unchanged to test predictions. Raw outputs are retained separately to document the effects of score scale and class weighting before calibration.",
                "",
            ]
        )

    lines.extend(
        [
            "## Reporting Exports",
            "",
            "- `random_split/tables/main_model_comparison_random_combined.csv`",
            "- `temporal_split/tables/model_metrics_temporal_combined.csv`",
            "- `random_split/tables/feature_group_model_performance.csv` and the corresponding temporal table",
            "- `random_split/figures/predictive_performance_by_feature_group.png` and the corresponding temporal figure",
            "- `random_split/tables/ebm_global_importance.csv` and the corresponding temporal table",
            "- `random_split/tables/logistic_key_coefficients_for_interpretation.csv` and the corresponding temporal table",
            "- Sample-representativeness tables are retained in `common/tables`.",
            "",
            "## Consistency",
            "",
            "All tables, metrics, figures, and this summary were generated inside this unique run directory. "
            "Logistic predictive performance and the main coefficient/odds-ratio interpretation refer to the same full-training fitted estimator. "
            "See `artifact_manifest.csv` for timestamps, sizes, and SHA-256 checksums.",
            "",
        ]
    )
    (run_root / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_reporting_exports(
    run_root: Path,
    results: dict[str, pd.DataFrame],
) -> None:
    metric_columns = {
        "model": "Model",
        "roc_auc": "AUC",
        "gini": "Gini",
        "ks_statistic": "KS_statistic",
        "avg_precision": "Average_Precision",
        "brier_score": "Brier_score",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1_score",
        "threshold": "selected_threshold",
    }
    rq2_models = {
        "Logistic Regression",
        "LightGBM",
        "Explainable Boosting Machine",
    }
    feature_order = [
        RQ2_FEATURE_SET_LABELS["borrower_only"],
        RQ2_FEATURE_SET_LABELS["financial_only"],
        RQ2_FEATURE_SET_LABELS["grade_only"],
        RQ2_FEATURE_SET_LABELS["borrower_financial"],
        RQ2_FEATURE_SET_LABELS["combined"],
    ]
    key_logistic_features = [
        "int_rate",
        "fico_score",
        "dti",
        "annual_inc",
        "loan_amnt",
        "term_months",
        "revol_util",
        "credit_history_months",
    ]

    for split_name, result in results.items():
        split_root = run_root / f"{split_name}_split"
        tables = split_root / "tables"
        figures = split_root / "figures"

        combined = (
            result.loc[result["feature_set"] == "combined"]
            .sort_values("roc_auc", ascending=False)
            .copy()
        )
        clean_metrics = combined[list(metric_columns)].rename(columns=metric_columns)
        metric_filename = (
            "main_model_comparison_random_combined.csv"
            if split_name == "random"
            else "model_metrics_temporal_combined.csv"
        )
        clean_metrics.to_csv(tables / metric_filename, index=False)

        rq2 = result.loc[result["model"].isin(rq2_models)].copy()
        rq2["feature_group_label"] = rq2["feature_set"].map(RQ2_FEATURE_SET_LABELS)
        rq2_export_columns = {
            "model": "Model",
            "feature_set": "Feature_Group_ID",
            "feature_group_label": "Feature_Group",
            "roc_auc": "AUC",
            "gini": "Gini",
            "ks_statistic": "KS_statistic",
            "avg_precision": "Average_Precision",
            "brier_score": "Brier_score",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1_score",
            "threshold": "selected_threshold",
        }
        rq2[list(rq2_export_columns)].rename(columns=rq2_export_columns).to_csv(
            tables / "feature_group_model_performance.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.pointplot(
            data=rq2,
            x="feature_group_label",
            y="roc_auc",
            hue="model",
            order=feature_order,
            markers=["o", "s", "D"],
            ax=ax,
        )
        ax.set_title("Predictive Performance by Feature Group")
        ax.set_xlabel("Feature group")
        ax.set_ylabel("Test ROC AUC")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Model", loc="best")
        fig.tight_layout()
        fig.savefig(
            figures / "predictive_performance_by_feature_group.png", dpi=180
        )
        plt.close(fig)

        ebm_source = tables / "ebm_global_feature_importance_combined.csv"
        if ebm_source.exists():
            ebm = pd.read_csv(ebm_source).sort_values(
                "importance", ascending=False
            ).reset_index(drop=True)
            ebm.insert(0, "rank", np.arange(1, len(ebm) + 1))
            ebm = ebm.rename(columns={"term": "feature"})
            ebm[["feature", "importance", "rank", "term_type"]].to_csv(
                tables / "ebm_global_importance.csv", index=False
            )

        logistic_source = tables / "logit_coefficients_combined.csv"
        if logistic_source.exists():
            logistic = pd.read_csv(logistic_source)
            logistic = logistic.loc[
                logistic["feature"].isin(key_logistic_features)
            ].copy()
            logistic["feature"] = pd.Categorical(
                logistic["feature"], categories=key_logistic_features, ordered=True
            )
            logistic = logistic.sort_values("feature")
            logistic["sign"] = np.where(
                logistic["coefficient"] > 0,
                "positive",
                np.where(logistic["coefficient"] < 0, "negative", "zero"),
            )
            logistic["interpretation_direction"] = np.where(
                logistic["coefficient"] > 0,
                "Higher values are associated with higher default odds",
                np.where(
                    logistic["coefficient"] < 0,
                    "Higher values are associated with lower default odds",
                    "No directional association",
                ),
            )
            logistic["coefficient_scale"] = (
                "Per one-standard-deviation increase in the cleaned, winsorised feature"
            )
            logistic["interpretation_scope"] = (
                "Predictive association from the full-training fitted Logistic model"
            )
            logistic[
                [
                    "feature",
                    "coefficient",
                    "odds_ratio",
                    "sign",
                    "interpretation_direction",
                    "coefficient_scale",
                    "coefficient_source",
                    "coefficient_training_rows",
                    "prediction_estimator",
                    "interpretation_scope",
                ]
            ].to_csv(
                tables / "logistic_key_coefficients_for_interpretation.csv",
                index=False,
            )


def write_artifact_manifest(run_root: Path, run_id: str) -> None:
    rows = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        rows.append(
            {
                "run_id": run_id,
                "relative_path": str(path.relative_to(run_root)),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(run_root / "artifact_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    if not 3_000 <= args.shap_sample_rows <= 5_000:
        raise ValueError("--shap-sample-rows must be between 3,000 and 5,000.")
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    validate_requested_dependencies(args)
    requested_rq1_models, requested_rq2_models = requested_model_names(args)
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("run_%Y%m%dT%H%M%SZ")
    run_root = args.output_dir / run_id
    common_paths = ensure_dirs(run_root / "common")
    (run_root / "code").mkdir(parents=True, exist_ok=True)

    raw, status_counts, metadata = load_completed_loans(args.archive, args.chunksize)
    clean = clean_data(raw)
    cleaning_report = clean.attrs.get("cleaning_report", {})
    clean = clean.dropna(subset=["default"]).reset_index(drop=True)
    if {"term_months", "issue_date"}.issubset(clean.columns):
        cohort = (
            clean.assign(issue_year=clean["issue_date"].dt.year)
            .groupby(["issue_year", "term_months"], dropna=False)
            .size()
            .reset_index(name="completed_loan_count")
            .sort_values(["issue_year", "term_months"])
        )
        cohort.to_csv(
            common_paths["tables"] / "completed_loans_by_term_issue_year.csv", index=False
        )

    rows_before_term_filter = len(clean)
    if args.term_filter != "all":
        term_value = float(args.term_filter)
        clean = clean.loc[clean["term_months"] == term_value].copy().reset_index(drop=True)
    rows_after_term_filter = len(clean)
    if rows_after_term_filter == 0:
        raise ValueError(f"No rows remain after applying term filter: {args.term_filter}")

    sampled = stratified_sample(clean, args.sample_rows, args.random_state)
    representativeness = save_sample_representativeness(clean, sampled, common_paths)

    borrower = available_features(sampled, BORROWER_FEATURES)
    risk_assessment = available_features(sampled, RISK_ASSESSMENT_FEATURES)
    financial = available_features(sampled, FINANCIAL_FEATURES)
    cuda_available = bool(HAS_TORCH and torch.cuda.is_available())
    compute_backend = (
        f"Hybrid: CUDA ({torch.cuda.get_device_name(0)}) for MLP/FT-Transformer; "
        "CPU for statistical and classical/tree models"
        if cuda_available
        else "CPU"
    )
    metadata.update(
        {
            "completed_rows_after_cleaning": int(len(clean)),
            "completed_rows_before_term_filter": int(rows_before_term_filter),
            "term_filter": args.term_filter,
            "rows_after_term_filter": int(rows_after_term_filter),
            "model_sample_rows": int(len(sampled)),
            "model_default_rate": float(sampled["default"].mean()),
            "run_id": run_id,
            "run_started_utc": run_started.isoformat(),
            "random_state": args.random_state,
            "sampling_method": "proportional_stratified_by_default_and_issue_year",
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "training_size": round(1 - args.validation_size - args.test_size, 10),
            "experiments": (
                ["random", "temporal"] if args.split == "both" else [args.split]
            ),
            "sample_representativeness": representativeness,
            "tree_model_tuning": not args.skip_tree_tuning,
            "tuning_max_rows": args.tuning_max_rows,
            "shap_sample_rows": args.shap_sample_rows,
            "ebm_original_scale_shape_features": EBM_ORIGINAL_SCALE_FEATURES,
            "rq1_models": requested_rq1_models,
            "rq2_models": requested_rq2_models,
            "rq2_feature_groups": [
                "borrower_only",
                "financial_only",
                "grade_only",
                "borrower_financial",
                "combined",
            ],
            "rq2_feature_group_labels": RQ2_FEATURE_SET_LABELS,
            "rq2_grouping_version": RQ2_GROUPING_VERSION,
            "rq2_grouping_note": (
                "Borrower-only contains borrower and application characteristics. "
                "Financial-only contains financial, loan, and credit-profile characteristics, "
                "including fico_score and credit_history_months. Grade-only contains the "
                "platform-assigned LendingClub grade."
            ),
            "compute_backend": compute_backend,
            "cuda_available": cuda_available,
            "predictive_glm_max_train_rows": args.max_statmodels_rows,
            "glm_inference_rows": args.glm_inference_rows,
            "predictive_glm_training_scope": (
                "complete_training_set"
                if args.max_statmodels_rows <= 0
                else "capped_training_set"
            ),
            "logistic_main_interpretation_source": (
                "same_full_training_fitted_model_as_predictive_probabilities"
            ),
            "glm_uncertainty_analysis_role": (
                "disabled" if args.glm_inference_rows <= 0 else "supplementary_appendix_check_only"
            ),
            "ft_transformer_included": bool(HAS_TORCH and not args.skip_ft_transformer),
            "neural_max_train_rows": args.ft_max_train_rows,
            "neural_training_scope": (
                "complete_training_set"
                if args.ft_max_train_rows <= 0
                else "capped_training_set"
            ),
            "ft_epochs": args.ft_epochs,
            "ft_batch_size": args.ft_batch_size,
            "probability_calibration_protocol": (
                "uniform_validation_sigmoid_on_each_models_raw_log_odds"
            ),
            "cleaning_report": cleaning_report,
            "completed_only_selection_note": (
                "Models use completed loans only: Fully Paid vs Charged Off/Default. "
                "Current, late, and grace-period loans are excluded; this completed-loan "
                "restriction introduces maturity and outcome-selection constraints."
            ),
            "borrower_features": borrower,
            "risk_assessment_features": risk_assessment,
            "financial_features": financial,
            "xgboost_available": HAS_XGBOOST,
            "lightgbm_available": HAS_LIGHTGBM,
            "catboost_available": HAS_CATBOOST,
            "ebm_available": HAS_EBM,
            "pytorch_available": HAS_TORCH,
            "lightgbm_included": bool(HAS_LIGHTGBM and not args.skip_lightgbm),
            "catboost_included": bool(HAS_CATBOOST and not args.skip_catboost),
            "ebm_included": bool(HAS_EBM and not args.skip_ebm),
            "batchnorm_mlp_used": bool(HAS_TORCH and not args.skip_mlp),
        }
    )
    (common_paths["tables"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    rq2_information_groups = [
        *[
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "information_group": "Borrower and application characteristics",
                "feature": feature,
            }
            for feature in borrower
        ],
        *[
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "information_group": "Financial and credit characteristics",
                "feature": feature,
            }
            for feature in financial
        ],
        *[
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "information_group": "Platform-assigned risk information",
                "feature": feature,
            }
            for feature in risk_assessment
        ],
    ]
    pd.DataFrame(rq2_information_groups).to_csv(
        common_paths["tables"] / "rq2_information_group_definitions.csv",
        index=False,
    )
    information_group_by_feature = {
        row["feature"]: row["information_group"]
        for row in rq2_information_groups
    }
    preprocessing_rows = []
    for feature in dict.fromkeys(borrower + financial + risk_assessment):
        if feature in COUNT_FEATURES:
            variable_type = "count"
            preprocessing = "median imputation; p1-p99 winsorisation; log1p; standardisation"
        elif feature in NUMERIC_DERIVED:
            variable_type = "continuous_numeric"
            preprocessing = "median imputation; p1-p99 winsorisation; standardisation"
        else:
            variable_type = "categorical"
            preprocessing = (
                "most-frequent imputation; training-fitted rare-category pooling; one-hot encoding"
            )
        preprocessing_rows.append(
            {
                "feature": feature,
                "information_group": information_group_by_feature.get(feature, ""),
                "variable_type": variable_type,
                "preprocessing_class": preprocessing,
                "learned_steps_fitted_on": "training_partition_only",
            }
        )
    pd.DataFrame(preprocessing_rows).to_csv(
        common_paths["tables"] / "variable_preprocessing_mapping.csv", index=False
    )
    complexity_profiles = [
        ("Logistic Regression", "Low", "Low", "Low", "Low", "Linear-additive fit; common validation sigmoid applied; coefficients directly inspectable."),
        ("Probit Regression", "Low", "Low", "Low", "Low", "Linear-additive latent-index fit; common validation sigmoid applied; coefficients directly inspectable."),
        ("Linear SVM", "Low", "Moderate", "Moderate", "Moderate", "Linear margin fit with validation-selected controls; a probability map is essential; weights are scale-dependent."),
        ("Random Forest", "High", "Moderate", "Low", "High", "Nonlinear bagged ensemble; validation selection and common sigmoid map; post-hoc explanation needed."),
        ("XGBoost", "High", "High", "Low", "High", "Regularised boosted trees with interacting controls; validation selection and common sigmoid map; post-hoc explanation needed."),
        ("LightGBM", "High", "High", "Low", "High", "Leaf-wise boosted trees with interacting controls; validation selection and common sigmoid map; SHAP used for explanation."),
        ("CatBoost", "High", "High", "Low", "High", "Symmetric boosted trees with interacting controls; validation selection and common sigmoid map; post-hoc explanation needed."),
        ("Explainable Boosting Machine", "Moderate", "Moderate", "Low", "Moderate", "Nonlinear additive components fitted iteratively; common sigmoid map; components remain inspectable but scale-dependent."),
        ("MLP Neural Network", "High", "High", "Moderate", "High", "Multilayer optimisation with early stopping; probability mapping from logits is essential; explanation and monitoring are not intrinsic."),
        ("Modified FT-Transformer-style", "High", "High", "Moderate", "High", "Attention-based tabular adaptation with early stopping; probability mapping from logits is essential; explanation and monitoring are not intrinsic."),
    ]
    pd.DataFrame(
        complexity_profiles,
        columns=[
            "model",
            "functional_flexibility",
            "development_and_estimation_burden",
            "probability_treatment_burden",
            "explanation_and_governance_burden",
            "brief_basis_under_implemented_protocol",
        ],
    ).to_csv(common_paths["tables"] / "model_complexity_profiles.csv", index=False)
    pd.DataFrame(
        [
            ("Functional flexibility", "Low", "Linear-additive or linear-margin representation."),
            ("Functional flexibility", "Moderate", "Inspectably nonlinear additive representation."),
            ("Functional flexibility", "High", "Ensemble or neural representation capable of nonlinear interactions."),
            ("Development and estimation burden", "Low", "Direct estimation with few controlling choices."),
            ("Development and estimation burden", "Moderate", "Validation selection or iterative fitting with a bounded candidate set."),
            ("Development and estimation burden", "High", "Multiple interacting modelling controls, architecture or optimisation choices, or iterative minibatch training."),
            ("Probability-treatment burden", "Low", "Native probabilities or link-scale scores followed by the single common validation-fitted sigmoid map."),
            ("Probability-treatment burden", "Moderate", "A probability map is essential because the fitted model supplies a margin or logit rather than a native probability."),
            ("Probability-treatment burden", "High", "Not used in this study."),
            ("Explanation and governance burden", "Low", "Global parameters are directly inspectable."),
            ("Explanation and governance burden", "Moderate", "Inspectability is partial or scale-dependent."),
            ("Explanation and governance burden", "High", "Post-hoc explanation and more extensive monitoring are required."),
        ],
        columns=["dimension", "rating", "classification_rule"],
    ).to_csv(common_paths["tables"] / "model_complexity_rubric.csv", index=False)
    pd.DataFrame(
        [
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "feature_set_id": "borrower_only",
                "feature_set_label": RQ2_FEATURE_SET_LABELS["borrower_only"],
                "included_information_groups": "Borrower and application characteristics",
            },
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "feature_set_id": "financial_only",
                "feature_set_label": RQ2_FEATURE_SET_LABELS["financial_only"],
                "included_information_groups": "Financial and credit characteristics",
            },
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "feature_set_id": "grade_only",
                "feature_set_label": RQ2_FEATURE_SET_LABELS["grade_only"],
                "included_information_groups": "Platform-assigned risk information",
            },
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "feature_set_id": "borrower_financial",
                "feature_set_label": RQ2_FEATURE_SET_LABELS["borrower_financial"],
                "included_information_groups": (
                    "Borrower and application characteristics; "
                    "Financial and credit characteristics"
                ),
            },
            {
                "grouping_version": RQ2_GROUPING_VERSION,
                "feature_set_id": "combined",
                "feature_set_label": RQ2_FEATURE_SET_LABELS["combined"],
                "included_information_groups": (
                    "Borrower and application characteristics; "
                    "Financial and credit characteristics; "
                    "Platform-assigned risk information"
                ),
            },
        ]
    ).to_csv(
        common_paths["tables"] / "rq2_feature_set_definitions.csv",
        index=False,
    )

    sampled.to_csv(common_paths["tables"] / "clean_model_sample.csv", index=False)
    save_eda(sampled, status_counts, common_paths, borrower + risk_assessment, financial)

    experiments = ["random", "temporal"] if args.split == "both" else [args.split]
    results: dict[str, pd.DataFrame] = {}
    for split_name in experiments:
        print(f"\n######## {split_name.upper()} SPLIT EXPERIMENT ########", flush=True)
        split_paths = ensure_dirs(run_root / f"{split_name}_split")
        results[split_name] = run_models(
            sampled,
            split_paths,
            split_name,
            args.validation_size,
            args.test_size,
            args.random_state,
            run_id,
            args.max_statmodels_rows,
            args.glm_inference_rows,
            args.skip_mlp,
            args.skip_ft_transformer,
            args.ft_max_train_rows,
            args.ft_epochs,
            args.ft_batch_size,
            not args.skip_tree_tuning,
            args.tuning_max_rows,
            args.skip_lightgbm,
            args.skip_catboost,
            args.skip_ebm,
            args.shap_sample_rows,
        )

    expected_rq1_models = set(requested_rq1_models)
    expected_rq2_models = set(requested_rq2_models)
    expected_rq2_feature_sets = set(RQ2_FEATURE_SET_LABELS)
    for split_name, split_result in results.items():
        combined_models = set(
            split_result.loc[
                split_result["feature_set"].eq("combined"), "model"
            ].tolist()
        )
        missing_rq1 = expected_rq1_models - combined_models
        rq2_result = split_result.loc[
            split_result["model"].isin(expected_rq2_models)
            & split_result["feature_set"].isin(expected_rq2_feature_sets)
        ]
        observed_rq2_pairs = set(
            zip(rq2_result["model"], rq2_result["feature_set"])
        )
        expected_rq2_pairs = {
            (model, feature_set)
            for model in expected_rq2_models
            for feature_set in expected_rq2_feature_sets
        }
        missing_rq2 = expected_rq2_pairs - observed_rq2_pairs
        if missing_rq1 or missing_rq2:
            raise RuntimeError(
                f"Incomplete {split_name} experiment: missing RQ1 models="
                f"{sorted(missing_rq1)}; missing RQ2 model/feature pairs="
                f"{sorted(missing_rq2)}"
            )

    combined_metrics = pd.concat(results.values(), ignore_index=True)
    combined_metrics.to_csv(run_root / "all_model_metrics.csv", index=False)
    expected_calibration = "validation_sigmoid_on_raw_log_odds"
    observed_calibration = set(
        combined_metrics["probability_calibration"].dropna().astype(str)
    )
    if observed_calibration != {expected_calibration}:
        raise RuntimeError(
            "Uniform probability calibration audit failed: "
            f"observed={sorted(observed_calibration)}"
        )

    partition_tables = []
    selected_configuration_tables = []
    for split_name in experiments:
        split_table_dir = run_root / f"{split_name}_split" / "tables"
        partition_file = split_table_dir / "partition_composition.csv"
        if partition_file.exists():
            partition_tables.append(pd.read_csv(partition_file))
        configuration_file = split_table_dir / "selected_model_configurations.csv"
        if configuration_file.exists():
            selected_configuration_tables.append(pd.read_csv(configuration_file))
    if partition_tables:
        pd.concat(partition_tables, ignore_index=True).to_csv(
            run_root / "partition_composition_all_splits.csv", index=False
        )
    if selected_configuration_tables:
        selected_configurations = pd.concat(
            selected_configuration_tables, ignore_index=True
        )
        combined_training = combined_metrics.loc[
            combined_metrics["feature_set"].eq("combined"),
            [
                "split",
                "model",
                "training_rows",
                "training_device",
                "epochs_trained",
            ],
        ].drop_duplicates(["split", "model"])
        selected_configurations = selected_configurations.merge(
            combined_training,
            on=["split", "model"],
            how="left",
        )
        selected_configurations.to_csv(
            run_root / "selected_model_configurations_all_splits.csv", index=False
        )

    gpu_name = "not used"
    if HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    reproducibility_rows = [
        {"item": "final_run_identifier", "value": run_id},
        {"item": "random_state", "value": args.random_state},
        {"item": "python_version", "value": platform.python_version()},
        {"item": "operating_system", "value": platform.platform()},
        {"item": "cpu", "value": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")},
        {"item": "logical_cpu_count", "value": os.cpu_count()},
        {"item": "gpu", "value": gpu_name},
        {"item": "numpy_version", "value": installed_version("numpy")},
        {"item": "pandas_version", "value": installed_version("pandas")},
        {"item": "scikit_learn_version", "value": installed_version("scikit-learn")},
        {"item": "statsmodels_version", "value": installed_version("statsmodels")},
        {"item": "xgboost_version", "value": installed_version("xgboost")},
        {"item": "lightgbm_version", "value": installed_version("lightgbm")},
        {"item": "catboost_version", "value": installed_version("catboost")},
        {"item": "interpret_version", "value": installed_version("interpret")},
        {"item": "torch_version", "value": installed_version("torch")},
        {"item": "shap_version", "value": installed_version("shap")},
    ]
    pd.DataFrame(reproducibility_rows).to_csv(
        run_root / "reproducibility_environment.csv", index=False
    )
    training_audit_columns = [
        "run_id",
        "split",
        "feature_set",
        "model",
        "training_rows",
        "inference_rows",
        "coefficient_source",
        "coefficient_training_rows",
        "supplementary_uncertainty_rows",
        "training_device",
        "epochs_trained",
        "best_validation_auc_internal",
    ]
    combined_metrics.loc[
        combined_metrics["feature_set"].eq("combined"),
        training_audit_columns,
    ].to_csv(run_root / "model_training_sample_audit.csv", index=False)
    save_reporting_exports(run_root, results)
    write_experiment_summary(run_root, run_id, metadata, representativeness, results)
    shutil.copy2(Path(__file__), run_root / "code" / Path(__file__).name)
    requirement_file = Path(__file__).with_name("requirements-lendingclub-ft-transformer.txt")
    if requirement_file.exists():
        shutil.copy2(requirement_file, run_root / "code" / requirement_file.name)
    completion = {
        "run_id": run_id,
        "status": "complete",
        "started_utc": run_started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
        "model_metric_rows": int(len(combined_metrics)),
    }
    (run_root / "run_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    write_artifact_manifest(run_root, run_id)

    print("\nTop combined-feature model rows:")
    print(
        combined_metrics.loc[combined_metrics["feature_set"] == "combined"]
        .sort_values(["split", "roc_auc"], ascending=[True, False])
        .head(20)
        .to_string(index=False)
    )
    print(f"\nOutputs saved to: {run_root.resolve()}")


if __name__ == "__main__":
    main()
