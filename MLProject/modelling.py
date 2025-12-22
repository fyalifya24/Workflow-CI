import argparse
import glob
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def find_dataset_file(data_dir: str) -> str:
    """
    Cari file dataset (csv/parquet/pkl/pickle) di folder data_dir.
    Kalau kosong, fallback cari di current working directory.
    """
    patterns = ["*.csv", "*.parquet", "*.pkl", "*.pickle"]
    files = []

    search_dir = Path(data_dir)
    if search_dir.exists() and search_dir.is_dir():
        for p in patterns:
            files.extend(glob.glob(str(search_dir / p)))

    if not files:
        for p in patterns:
            files.extend(glob.glob(p))

    # Coba prioritas nama file yang sering dipakai
    preferred = [
        "students_performance_preprocessing.csv",
        "students_performance_preprocessing.parquet",
        "students_performance_preprocessing.pkl",
        "students_performance_preprocessing.pickle",
    ]
    for f in preferred:
        if (search_dir / f).exists():
            return str(search_dir / f)
        if Path(f).exists():
            return str(Path(f))

    if not files:
        raise FileNotFoundError(
            f"Tidak ada file dataset ditemukan.\n"
            f"Pastikan ada file preprocessing (.csv/.parquet/.pkl/.pickle) di '{data_dir}' atau root project."
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


def resolve_target_column(df: pd.DataFrame, target_col_raw: str) -> str:
    """
    Normalisasi nama target (math_score <-> math score), strip, dll.
    """
    raw = (target_col_raw or "").strip()

    candidates = [
        raw,
        raw.replace("_", " "),
        raw.replace(" ", "_"),
        raw.lower(),
        raw.lower().replace("_", " "),
        raw.lower().replace(" ", "_"),
    ]

    # mapping kolom ke lowercase untuk pencocokan aman
    cols_lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            print(f"[INFO] target_col resolved: '{target_col_raw}' -> '{c}'")
            return c
        if c.lower() in cols_lower_map:
            real = cols_lower_map[c.lower()]
            print(f"[INFO] target_col resolved: '{target_col_raw}' -> '{real}'")
            return real

    raise ValueError(
        f"Kolom target '{target_col_raw}' tidak ditemukan.\n"
        f"Kolom yang ada: {list(df.columns)}"
    )


def detect_problem_type(y: pd.Series) -> str:
    """
    Heuristik sederhana:
    - bool/object/category => classification
    - numeric dengan unique <= 20 => classification
    - sisanya regression
    """
    if y.dtype == "bool":
        return "classification"
    if str(y.dtype) in ["object", "category"]:
        return "classification"

    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    return "regression"


def get_or_create_run(experiment_name: str):
    """
    Aman untuk:
    - MLflow Projects/CI (`mlflow run`): MLflow sudah bikin run & set MLFLOW_RUN_ID.
      => kita resume run itu (biar gak bentrok experiment).
    - Local run (python modelling.py): kita set experiment & start run sendiri.
    """
    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()

    # CI / mlflow run
    if run_id:
        return mlflow.start_run(run_id=run_id)

    # local biasa
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name="baseline_model")


def main(data_path: str, out_dir: str, target_col: str, experiment_name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tentuin dataset_path
    p = Path(data_path)
    if p.exists() and p.is_dir():
        dataset_path = find_dataset_file(str(p))
    else:
        dataset_path = data_path

    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = load_df(dataset_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # resolve target col
    target_col = resolve_target_column(df, target_col)

    # drop null target
    before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} rows with null target")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # bool -> int (biar aman kalau ada bool di fitur)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    problem_type = detect_problem_type(y)
    print(f"[INFO] Problem type detected: {problem_type}")

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"[INFO] Categorical columns: {cat_cols}")
    print(f"[INFO] Numerical columns: {num_cols}")

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
        print("[INFO] Using LogisticRegression (classification)")
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        print("[INFO] Using RandomForestRegressor (regression)")

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # ✅ Kriteria 2: AUTLOG UMUM
    # ❌ Tidak ada logging manual (log_param/log_metric/log_artifact/set_tag)
    mlflow.autolog()

    with get_or_create_run(experiment_name):
        print("[INFO] Training started...")
        pipeline.fit(X_train, y_train)
        print("[INFO] Training completed.")

        # Simpan model ke folder output (bukan logging MLflow manual)
        model_path = out_dir / "model_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        print(f"[INFO] Model saved: {model_path}")

    print("Selesai training + autolog ke MLflow.")
    print("Jalankan MLflow UI: mlflow ui --port 5000")
    print("Buka: http://127.0.0.1:5000")
    print(f"Output tersimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--target_col", type=str, default=os.getenv("TARGET_COL", "math score").strip())
    parser.add_argument("--experiment_name", type=str, default="kriteria2_basic_modelling")

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
