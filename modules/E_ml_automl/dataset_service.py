from __future__ import annotations

from pathlib import Path
import pandas as pd
import logging
from sklearn.datasets import load_digits, load_iris, load_wine, load_diabetes, load_breast_cancer

# Set up logging for the user to debug dataset loading
import os
from datetime import datetime
os.makedirs("data/logs", exist_ok=True)

def log_debug(msg):
    with open("data/logs/dataset_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | DEBUG | {msg}\n")
        print(f"DEBUG: {msg}")

def log_info(msg):
    with open("data/logs/dataset_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO | {msg}\n")
        print(f"INFO: {msg}")

def log_warn(msg):
    with open("data/logs/dataset_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | WARN | {msg}\n")
        print(f"WARN: {msg}")

def log_error(msg):
    with open("data/logs/dataset_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {msg}\n")
        print(f"ERROR: {msg}")

from modules.F_stateless_info.kaggle_service import resolve_best_dataset_ref, list_dataset_files, download_dataset


TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".parquet"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _bundle_from_dataframe(dataset_id: str, df: pd.DataFrame, target_col: str | None) -> dict:
    import os
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{dataset_id}.csv"
    df.to_csv(temp_file, index=False)
    
    profile = _profile_dataframe(df, target_col)
    return {
        "success": True,
        "dataset": dataset_id,
        "dataset_info": {
            "name": dataset_id,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_names": list(df.columns),
            "target_name": target_col,
            "profile": profile,
            "preview_file": str(temp_file.resolve()),
        },
        "dataset_preview": df.head(10).to_dict(orient="records"),
        "dataset_files": [],
        "dataset_profile": profile,
    }


def _builtin_fallback(query: str) -> dict:
    log_info(f"[_builtin_fallback] Attempting to find local/fallback datasets for query: '{query}'")
    q = query.lower().strip()
    try:
        if q in {"mnist", "digits"}:
            bunch = load_digits(as_frame=True)
            df = bunch.frame.copy()
            return _bundle_from_dataframe("mnist", df, "target")
        if q == "iris":
            bunch = load_iris(as_frame=True)
            return _bundle_from_dataframe("iris", bunch.frame.copy(), "target")
        if q == "wine":
            bunch = load_wine(as_frame=True)
            return _bundle_from_dataframe("wine", bunch.frame.copy(), "target")
        if q == "diabetes":
            bunch = load_diabetes(as_frame=True)
            return _bundle_from_dataframe("diabetes", bunch.frame.copy(), "target")
        if q in {"breast cancer", "breast_cancer"}:
            bunch = load_breast_cancer(as_frame=True)
            return _bundle_from_dataframe("breast_cancer", bunch.frame.copy(), "target")
        if q == "titanic":
            df = pd.DataFrame({
                "pclass": [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,1,1,2,2,3,3,1,2,3,3,2,1],
                "sex": ["female","male","female","male","female","male","female","male","female","male","female","male","female","male","female","male","female","male","female","male","female","male","female","male","male","male","female","male","female","female"],
                "age": [29,36,25,54,2,45,30,34,14,28,50,62,22,18,31,40,4,16,38,65,27,19,24,32,48,35,8,26,44,42],
                "sibsp": [0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,4,0,0,0,1,0,0,0],
                "parch": [0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,2,0,0,0],
                "fare": [211.34,71.28,151.55,51.86,120.00,26.55,13.00,26.00,30.07,13.00,26.00,10.50,7.25,8.05,16.10,7.90,29.13,7.75,80.00,26.55,10.50,10.50,31.39,7.85,76.73,26.00,21.08,7.05,26.00,227.53],
                "embarked": ["S","C","S","S","S","S","S","S","C","S","S","S","S","S","S","S","Q","Q","C","S","S","S","S","S","C","S","S","S","S","C"],
                "survived": [1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,1],
            })
            return _bundle_from_dataframe("titanic", df, "survived")
    except Exception as e:
        log_error(f"[_builtin_fallback] Exception during generic built-in check: {e}")
        return {"success": False, "error": str(e)}

    # ── OpenML fallback: try to fetch any dataset by name ──
    log_info(f"[_builtin_fallback] Query '{query}' not found in static builtins. Trying OpenML Network Fallback...")
    return _openml_fallback(query)


def _openml_fallback(query: str) -> dict:
    """Try to fetch a dataset from OpenML via sklearn.datasets.fetch_openml."""
    try:
        from sklearn.datasets import fetch_openml

        # Clean the query into a plausible OpenML name
        clean = query.strip()

        # Try fetching by name (OpenML search)
        log_debug(f"[_openml_fallback] Fetching perfectly matched name: {clean}")
        bunch = fetch_openml(name=clean, as_frame=True, parser="auto")
        df = bunch.frame
        if df is None or df.empty:
            log_warn(f"[_openml_fallback] Fetch success but empty DataFrame for {clean}")
            return {"success": False, "error": f"OpenML returned empty data for '{query}'."}

        # Limit to first 2000 rows to keep things snappy
        df = df.head(2000).copy()

        target_col = _guess_target_column(df)
        dataset_id = clean.lower().replace(" ", "_")
        return _bundle_from_dataframe(dataset_id, df, target_col)
    except Exception:
        pass

    # If exact name fails, try a fuzzy keyword search via OpenML API
    try:
        log_debug(f"[_openml_fallback] Exact match failed. Falling back to fuzzy keyword search on OpenML.")
        import openml
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        # Search by keyword in dataset name
        keywords = query.lower().split()
        mask = datasets["name"].str.lower().apply(
            lambda n: all(kw in n for kw in keywords)
        )
        matches = datasets[mask].sort_values("NumberOfInstances", ascending=False)
        if not matches.empty:
            top = matches.iloc[0]
            oml_ds = openml.datasets.get_dataset(int(top["did"]))
            X, y, _, _ = oml_ds.get_data(dataset_format="dataframe")
            df = X.copy()
            if y is not None:
                df["target"] = y
            df = df.head(2000)
            target_col = "target" if y is not None else _guess_target_column(df)
            dataset_id = str(top["name"]).lower().replace(" ", "_")
            return _bundle_from_dataframe(dataset_id, df, target_col)
    except Exception:
        pass

    # ── Hugging Face Datasets Fallback ──
    try:
        log_debug(f"[_huggingface_fallback] OpenML lookup failed. Trying Hugging Face APIs for: '{query}'")
        from huggingface_hub import HfApi
        import requests
        api = HfApi()
        # Search Hugging Face
        matches = list(api.list_datasets(search=query, limit=1))
        
        # Super backup: reduce the query to fewer words if multiple words to increase matches
        if not matches and len(query.split()) > 2:
           reduced = " ".join(query.split()[:2])
           log_debug(f"[_huggingface_fallback] Zero exact matches. Trying relaxed query: '{reduced}'")
           matches = list(api.list_datasets(search=reduced, limit=1))
           
        if matches:
            hf_id = matches[0].id
            log_info(f"[_huggingface_fallback] Match found on HuggingFace: {hf_id}. Requesting parquet distribution.")
            url = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
            res = requests.get(url).json()
            if "parquet_files" in res and len(res["parquet_files"]) > 0:
                parquet_url = res["parquet_files"][0]["url"]
                log_info(f"[_huggingface_fallback] Parquet URL located, downloading frame...")
                df = pd.read_parquet(parquet_url).head(2000)
                target_col = _guess_target_column(df)
                dataset_id = hf_id.split("/")[-1].replace("-", "_")
                log_info(f"[_huggingface_fallback] Successfully downloaded DataFrame for {dataset_id}")
                return _bundle_from_dataframe(dataset_id, df, target_col)
    except Exception as e:
        log_error(f"[_huggingface_fallback] HuggingFace exception encountered: {e}")
        pass

    log_warn(f"[FINAL FAULT] Exhausted all remote & local fallback strategies for: '{query}'")
    return {"success": False, "error": (
        f"Could not find a dataset matching '{query}'.\n\n"
        f"**Built-in datasets** (always available): iris, mnist, titanic, wine, diabetes, breast_cancer.\n\n"
        f"**To load any Kaggle dataset**, set up the Kaggle CLI:\n"
        f"1. `pip install kaggle`\n"
        f"2. Go to kaggle.com → Account → Create New API Token\n"
        f"3. Place `kaggle.json` in `~/.kaggle/`\n\n"
        f"Then retry: *'hey mycroft load {query}'*"
    )}


def _choose_table_file(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".csv" else 1, p.stat().st_size))
    return candidates[0]


def _count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path, nrows=500)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=500)
    if s == ".txt":
        return pd.read_csv(path, nrows=500)
    if s == ".json":
        return pd.read_json(path).head(500)
    if s == ".parquet":
        return pd.read_parquet(path).head(500)
    raise ValueError(f"Unsupported table file '{path.name}'.")


def _guess_target_column(df: pd.DataFrame) -> str | None:
    preferred = ["target", "label", "class", "y", "survived", "price", "saleprice"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]
    return df.columns[-1] if len(df.columns) > 1 else None


def _profile_dataframe(df: pd.DataFrame, target_col: str | None) -> dict:
    if target_col and target_col in df.columns:
        target = df[target_col]
        unique = target.nunique(dropna=True)
        if str(target.dtype).startswith(("object", "category", "bool")) or unique <= 30:
            task = "tabular_classification"
            suggested_model = "xgboost"
        else:
            task = "tabular_regression"
            suggested_model = "xgboost"
    else:
        task = "tabular_generic"
        suggested_model = "xgboost"

    return {
        "modality": "tabular",
        "task_family": task,
        "suggested_model": suggested_model,
        "target_column": target_col,
    }


def _profile_image_dataset(root: Path) -> dict:
    return {
        "modality": "image",
        "task_family": "image_classification",
        "suggested_model": "cnn",
        "target_column": None,
        "image_count": _count_images(root),
    }


def load_dataset_by_query(dataset_query: str, base_dir: str = "data/kaggle_cache") -> dict:
    log_info(f"====== [load_dataset_by_query] INITIATING DATASET LOAD: '{dataset_query}' ======")
    resolved = resolve_best_dataset_ref(dataset_query)
    if not resolved["success"]:
        log_warn(f"[load_dataset_by_query] Kaggle resolution failed: {resolved.get('error')}. Proceeding to fallback tree.")
        return _builtin_fallback(dataset_query)

    dataset_ref = resolved["dataset_ref"]
    target_dir = Path(base_dir) / dataset_ref.replace("/", "__")

    files_result = list_dataset_files(dataset_ref)
    if not files_result["success"]:
        return _builtin_fallback(dataset_query)

    download_result = download_dataset(dataset_ref, str(target_dir))
    if not download_result["success"]:
        return _builtin_fallback(dataset_query)

    table_file = _choose_table_file(target_dir)
    image_count = _count_images(target_dir)

    dataset_files = files_result["files"]

    if table_file is not None:
        try:
            df = _read_table(table_file)
            target_col = _guess_target_column(df)
            profile = _profile_dataframe(df, target_col)
            return {
                "success": True,
                "dataset": dataset_query.strip().lower(),
                "dataset_info": {
                    "name": dataset_query.strip().lower(),
                    "kaggle_ref": dataset_ref,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "feature_names": list(df.columns),
                    "target_name": target_col,
                    "profile": profile,
                    "download_dir": str(target_dir.resolve()),
                    "preview_file": str(table_file.resolve()),
                },
                "dataset_preview": df.head(10).to_dict(orient="records"),
                "dataset_files": dataset_files,
                "dataset_profile": profile,
            }
        except Exception:
            pass

    if image_count > 0:
        profile = _profile_image_dataset(target_dir)
        sample_files = [str(p.name) for p in list(target_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES][:10]
        return {
            "success": True,
            "dataset": dataset_query.strip().lower(),
            "dataset_info": {
                "name": dataset_query.strip().lower(),
                "kaggle_ref": dataset_ref,
                "rows": image_count,
                "columns": 0,
                "feature_names": [],
                "target_name": None,
                "profile": profile,
                "download_dir": str(target_dir.resolve()),
            },
            "dataset_preview": [{"sample_image_file": f} for f in sample_files],
            "dataset_files": dataset_files,
            "dataset_profile": profile,
        }

    return {
        "success": True,
        "dataset": dataset_query.strip().lower(),
        "dataset_info": {
            "name": dataset_query.strip().lower(),
            "kaggle_ref": dataset_ref,
            "rows": 0,
            "columns": 0,
            "feature_names": [],
            "target_name": None,
            "profile": {"modality": "unknown", "task_family": "generic", "suggested_model": "xgboost"},
            "download_dir": str(target_dir.resolve()),
        },
        "dataset_preview": [],
        "dataset_files": dataset_files,
        "dataset_profile": {"modality": "unknown", "task_family": "generic", "suggested_model": "xgboost"},
    }