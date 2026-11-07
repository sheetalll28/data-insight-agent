import re
import pandas as pd
import numpy as np
import sys
import os

# Import our from-scratch math engine!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from outlier_detection import calculate_iqr_bounds

# A text column is treated as numeric when at least this share of its
# non-empty values parse as numbers (e.g. a numeric column polluted by "UNKNOWN").
NUMERIC_COERCION_THRESHOLD = 0.8


# File types the app accepts, mapped to their field separator.
SUPPORTED_EXTENSIONS = {"csv": ",", "tsv": "\t"}


def read_tabular_file(file, filename):
    """
    Read an uploaded CSV or TSV into a DataFrame.
    The separator comes from the file extension (.csv -> comma, .tsv -> tab),
    so a comma inside a TSV value (e.g. "Smith, John") is never split.
    Raises ValueError for unsupported extensions or files with no rows/columns.
    """
    ext = os.path.splitext(str(filename))[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type '.{ext}' (use .csv or .tsv)")
    df = pd.read_csv(file, sep=SUPPORTED_EXTENSIONS[ext])
    if df.empty or df.shape[1] == 0:
        raise ValueError("the file has no rows or no columns")
    if ext == "csv" and df.shape[1] == 1 and "\t" in str(df.columns[0]):
        raise ValueError("this looks tab-separated; rename it to .tsv")
    return df


def detect_identifier_columns(df):
    """
    Flags columns that look like row identifiers (e.g. "Patient ID").
    A column is an identifier when (almost) every row has a value, every value
    is unique, AND it is either named like an ID or is a short text key. Identifiers carry no
    statistical signal, so they are excluded from clipping, imputation,
    correlation, the neural network and clustering.
    """
    ids = []
    n = len(df)
    if n < 2:
        return ids
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < max(2, int(0.95 * n)) or s.nunique() != len(s):
            continue
        name = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        named_like_id = bool(re.search(r"(^|_)(id|uuid|key|identifier)$", name)) or (
            name.endswith("id") and len(name) <= 12          # e.g. "PatientID"
        )
        is_text = not pd.api.types.is_numeric_dtype(s)
        # Unnamed unique text (codes, emails, names) only counts once there are
        # enough rows for "every value is different" to be meaningful.
        short_unique_text = is_text and len(s) >= 20 and s.astype(str).str.len().mean() < 40
        if named_like_id or short_unique_text:
            ids.append(col)
    return ids


def run_cleaning_pipeline(raw_df, clip_outliers=True, return_report=False):
    """
    Takes a raw, messy pandas DataFrame and runs it through our
    from-scratch mathematical cleaning engines.

    Steps (each one is recorded in an audit log):
      1. Coerce mostly-numeric text columns to numbers (junk text -> NaN).
      2. Drop exact duplicate rows (only rows identical in EVERY column).
      3. Detect IQR outliers and, if clip_outliers=True, clip them to the bounds.
      4. Impute missing numeric values with the column mean and missing
         categorical values with the column mode.

    The input DataFrame is never modified.
    Returns the cleaned DataFrame, or (cleaned_df, report) if return_report=True.
    """
    df = raw_df.copy()
    report = {
        "rows_in": int(len(df)),
        "coerced_to_numeric": {},
        "identifier_columns": [],
        "exact_duplicates_removed": 0,
        "near_duplicates_ignoring_ids": 0,
        "outliers": {},
        "imputed": {},
        "rows_out": None,
    }

    # 1. Coerce strings that are actually numbers
    for col in df.columns:
        # Text columns are dtype "object" in pandas < 3 and "str" in pandas >= 3.
        if not pd.api.types.is_numeric_dtype(df[col]) and (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            non_empty = df[col].dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            if len(non_empty) == 0:
                continue
            parsed = pd.to_numeric(non_empty, errors='coerce')
            share = parsed.notna().mean()
            if share >= NUMERIC_COERCION_THRESHOLD:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                report["coerced_to_numeric"][col] = int(parsed.isna().sum())

    # 2. Deduplication (exact rows only: identical-looking rows with different
    # IDs are NOT assumed to be the same entity, they are only counted)
    before = len(df)
    df = df.drop_duplicates(keep='first')
    report["exact_duplicates_removed"] = int(before - len(df))

    # Identifier detection runs AFTER exact dedup: a copied row repeats its ID,
    # which would otherwise hide the ID column.
    ids = detect_identifier_columns(df)
    report["identifier_columns"] = ids
    non_id = [c for c in df.columns if c not in ids]
    if ids and non_id:
        report["near_duplicates_ignoring_ids"] = int(df.duplicated(subset=non_id).sum())

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ids]

    # 3. Outlier Detection (& optional clipping) using our from-scratch IQR engine
    for col in numeric_cols:
        valid_data = df[col].dropna().values
        if len(valid_data) == 0:
            continue
        lower, upper = calculate_iqr_bounds(valid_data)
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out:
            report["outliers"][col] = {
                "count": n_out, "lower": float(lower), "upper": float(upper),
                "action": "clipped" if clip_outliers else "flagged only",
            }
            if clip_outliers:
                df[col] = df[col].clip(lower, upper)

    # 4. Imputation
    for col in df.columns:
        if col in ids:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if col in numeric_cols:
            fill = df[col].mean()
            method = "mean"
        else:
            mode = df[col].mode(dropna=True)
            if mode.empty:
                continue
            fill = mode.iloc[0]
            method = "mode"
        df[col] = df[col].fillna(fill)
        report["imputed"][col] = {"count": n_missing, "method": method}

    report["rows_out"] = int(len(df))
    return (df, report) if return_report else df


def calculate_correlation_matrix(df):
    """
    Calculates the Pearson Correlation Matrix entirely from scratch using NumPy.
    Formula: Covariance(X,Y) / (StdDev(X) * StdDev(Y))

    Vectorised: one matrix product instead of a Python double loop
    (O(n*p^2) in BLAS rather than p^2 interpreted iterations).
    Pairs involving a zero-variance column get r = 0.
    Note: non-numeric columns are label-encoded, so their r only measures a
    monotone trend in arbitrary category codes; treat it as a rough signal.
    """
    df_encoded = df.copy()
    
    # Auto-encode categorical variables so they are included in the math
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            df_encoded[col] = pd.factorize(df_encoded[col])[0]
            
    cols = df_encoded.columns.tolist()
    data = df_encoded[cols].to_numpy(dtype=float)

    centered = data - data.mean(axis=0)
    cov = centered.T @ centered                      # sum of cross-products
    norms = np.sqrt(np.diag(cov))                    # sqrt(sum of squares)
    denom = np.outer(norms, norms)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.where(denom > 0, cov / denom, 0.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
                
    return pd.DataFrame(corr_matrix, index=cols, columns=cols)
