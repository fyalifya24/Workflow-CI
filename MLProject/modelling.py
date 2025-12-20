import argparse
import glob
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def find_dataset_file(data_dir: str) -> str:
    patterns = ["*.csv", "*.parquet", "*.pkl", "*.pickle"]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(Path(data_dir) / p)))

    if not files:
        raise FileNotFoundError(
            f"Tidak ada file dataset di '{data_dir}'. "
            f"Pastikan folder ini berisi hasil preprocessing (csv/parquet/pkl)."
        )

    return sorted(files)[0]


def load_df(file_path: str) -> pd.DataFrame:
    fp = file_path.lower()
    if fp.endswith(".csv"):
        return pd.read_csv(file_path)
    if fp.endswith(".parquet"):
        return pd.read_parquet(file_path)
    if fp.endswith(".pkl") or fp.endswith(".pickle"):
        return pd.read_pickle(file_path)
    raise ValueError(f"Format file tidak didukung: {file_path}")


def detect_problem_type(y: pd.Series) -> str:
    if y.dtype == "bool":
        return "classification"
    if str(y.dtype) in ["object", "category"]:
        return "classification"

    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    return "regression"


def resolve_target_column(df: pd.DataFrame, target_col_raw: str) -> str:
    candidates = [
        target_col_raw,
        target_col_raw.replace("_", " "),
        target_col_raw.replace(" ", "_"),
        target_col_raw.strip(),
        target_col_raw.strip().replace("_", " "),
        target_col_raw.strip().replace(" ", "_"),
    ]

    for c in candidates:
        if c in df.columns:
            print(f"[INFO] target_col resolved: '{target_col_raw}' -> '{c}'")
            return c

    raise ValueError(
        f"Kolom target '{target_col_raw}' tidak ditemukan.\n"
        f"Kolom yang ada: {list(df.columns)}"
    )


def start_mlflow_run(experiment_name: str):
    """
    FIX untuk MLflow Projects:
    - kalau jalan lewat `mlflow run`, MLflow sudah bikin run_id dan set env MLFLOW_RUN_ID.
      Jadi kita HARUS attach ke run itu, bukan bikin run baru.
    """
    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()

    if run_id:
        # attach ke run dari mlflow projects
        return mlflow.start_run(run_id=run_id)
    else:
        # jalan manual (python modelling.py ...)
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name="baseline_model")


def main(data_path: str, out_dir: str, target_col: str, experiment_name: str):
    data_path = str(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tentuin dataset_path
    if Path(data_path).is_dir():
        dataset_path = find_dataset_file(data_path)
    else:
        dataset_path = data_path

    df = load_df(dataset_path)

    # resolve target col (math_score -> math score)
    target_col = resolve_target_column(df, target_col)

    # drop null target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # bool -> int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    problem_type = detect_problem_type(y)
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    if problem_type == "classification":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # MLflow autolog - TAMBAHKAN exclusive=True untuk hindari konflik
    mlflow.sklearn.autolog(log_models=True, exclusive=True)

    # >>> PERBAIKAN UTAMA DI SINI: HAPUS mlflow.log_param yang duplicate <<<
    with start_mlflow_run(experiment_name):
        # GANTI mlflow.log_param dengan mlflow.set_tag untuk info tambahan
        mlflow.set_tag("dataset_file", str(Path(dataset_path).name))
        mlflow.set_tag("target_col_resolved", target_col)
        mlflow.set_tag("problem_type", problem_type)
        mlflow.set_tag("model_type", type(model).__name__)

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if problem_type == "classification":
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            mlflow.log_metric("accuracy_manual", float(acc))
            mlflow.log_metric("precision_weighted_manual", float(prec))
            mlflow.log_metric("recall_weighted_manual", float(rec))
            mlflow.log_metric("f1_weighted_manual", float(f1))

            if hasattr(pipeline.named_steps["model"], "predict_proba") and y_test.nunique() == 2:
                proba = pipeline.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, proba)
                    mlflow.log_metric("roc_auc_manual", float(auc))
                except Exception:
                    pass
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            mlflow.log_metric("rmse_manual", rmse)
            mlflow.log_metric("mae_manual", mae)
            mlflow.log_metric("r2_manual", r2)

        model_path = out_dir / "model_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        metrics_path = out_dir / "metrics.txt"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"problem_type={problem_type}\n")
            f.write(f"target_col={target_col}\n")
            f.write(f"train_samples={len(X_train)}\n")
            f.write(f"test_samples={len(X_test)}\n")
            if problem_type == "regression":
                f.write(f"rmse={rmse:.4f}\n")
                f.write(f"mae={mae:.4f}\n")
                f.write(f"r2={r2:.4f}\n")
            else:
                f.write(f"accuracy={acc:.4f}\n")
                f.write(f"f1_weighted={f1:.4f}\n")

        mlflow.log_artifact(str(metrics_path))

    print("=" * 50)
    print("Selesai training + logging ke MLflow.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Output tersimpan di: {out_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--target_col", type=str, default=os.getenv("TARGET_COL", "math score").strip())
    parser.add_argument("--experiment_name", type=str, default="kriteria3_workflow_ci")

    args = parser.parse_args()

    if not args.target_col:
        raise ValueError(
            "TARGET belum ditentukan.\n"
            "Pilih salah satu:\n"
            "1) set env var: TARGET_COL=nama_kolom_target\n"
            "2) atau jalankan dengan argumen: --target_col nama_kolom_target"
        )

    main(
        data_path=args.data_path,
        out_dir=args.out_dir,
        target_col=args.target_col,
        experiment_name=args.experiment_name,
    )
