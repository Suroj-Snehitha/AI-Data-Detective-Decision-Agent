import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def profile(df: pd.DataFrame) -> dict:
    """Basic data-quality facts."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "text_cols": df.select_dtypes(include="object").columns.tolist(),
    }


def find_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    """Flag unusual rows using Isolation Forest."""
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] == 0 or len(num) < 20:
        return pd.DataFrame()
    X = StandardScaler().fit_transform(num)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)          # -1 = anomaly
    scores = model.score_samples(X)        # lower = more abnormal
    out = df.loc[num.index].copy()
    out["anomaly_score"] = scores
    return out[labels == -1].sort_values("anomaly_score").head(20)


def strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list:
    corr = df.select_dtypes(include=np.number).corr()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if abs(corr.loc[a, b]) >= threshold:
                pairs.append({"col_a": a, "col_b": b, "corr": round(corr.loc[a, b], 3)})
    return pairs


def outliers_iqr(df: pd.DataFrame) -> dict:
    """Per-column outlier counts using the IQR rule."""
    result = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n:
            result[col] = n
    return result


def investigate(df: pd.DataFrame) -> dict:
    """Run everything and return one evidence dict."""
    anomalies = find_anomalies(df)
    return {
        "profile": profile(df),
        "strong_correlations": strong_correlations(df),
        "iqr_outliers": outliers_iqr(df),
        "top_anomalies": anomalies.head(5).to_dict(orient="records"),
        "anomaly_count": len(anomalies),
    }