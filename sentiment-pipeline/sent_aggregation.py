import os
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_DATASET = "mateiplescan/processed-financial-news-XXL-enriched"
TARGET_DATASET = "mateiplescan/processed-financial-news-XXL-aggregated"
HF_TOKEN = "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"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Load & explode
# ---------------------------------------------------------------------------
def load_and_explode(source: str, token: str) -> pd.DataFrame:
    log.info("Loading dataset: %s", source)
    
    # Dataset uses custom shard splits instead of "train"
    ds_info = load_dataset(source, token=token)
    available_splits = list(ds_info.keys())
    log.info("Found %d splits. First few: %s", len(available_splits), available_splits[:5])

    if "train" in available_splits:
        ds = ds_info["train"]
    else:
        # Concatenate all shards into one dataset
        from datasets import concatenate_datasets
        log.info("Concatenating %d shards...", len(available_splits))
        ds = concatenate_datasets([ds_info[s] for s in available_splits])

    df = ds.to_pandas()
    log.info("Raw shape after load: %s", df.shape)

    df = df.explode("symbol_label").rename(columns={"symbol_label": "symbol"})
    df = df.dropna(subset=["symbol", "date"])

    # Normalise dates: handle both tz-aware and tz-naive sources robustly
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)   # parse & standardise to UTC
        .dt.tz_convert(None)                   # strip tz info (safer than tz_localize)
        .dt.normalize()                        # floor to midnight / day boundary
    )

    log.info("Shape after explode & date normalisation: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 – Daily aggregation
# ---------------------------------------------------------------------------
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse individual articles into one row per (date, symbol)."""
    log.info("Aggregating to daily (date × symbol) level...")
    daily = (
        df.groupby(["date", "symbol"], sort=True)
        .agg(
            day_avg_sentiment=("sentiment_score", "mean"),
            sentiment_volatility=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    log.info("Daily shape: %s  |  unique symbols: %d", daily.shape, daily["symbol"].nunique())
    return daily


# ---------------------------------------------------------------------------
# Step 3 – Per-symbol calendar gap-filling (bounded, no global explosion)
# ---------------------------------------------------------------------------
def fill_calendar_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex each symbol to a *per-symbol* continuous daily calendar.
    This avoids the n_symbols × global_date_range blowup that occurs when
    symbols have very different listing histories.
    """
    log.info("Filling calendar gaps (per-symbol bounded reindex)...")

    def _reindex_symbol(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.set_index("date").sort_index()
        full_range = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_range)
        grp["symbol"] = grp["symbol"].ffill().bfill()   # restore symbol name on gap rows
        return grp.reset_index(names="date")

    filled = (
        daily.groupby("symbol", group_keys=False)
        .apply(_reindex_symbol)
        .reset_index(drop=True)
    )

    log.info("Shape after gap-fill: %s", filled.shape)
    return filled


# ---------------------------------------------------------------------------
# Step 4 – Structural corrections & forward-fill
# ---------------------------------------------------------------------------
def apply_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Fix structural artefacts introduced by gap-filling and single-article days."""
    # Gap rows have no articles
    df["article_count"] = df["article_count"].fillna(0).astype("int32")

    # std() is NaN when there is exactly one article — define it as 0
    df.loc[df["article_count"] == 1, "sentiment_volatility"] = 0.0

    # Forward-fill sentiment per symbol across calendar gaps (weekends / holidays)
    for col in ("day_avg_sentiment", "sentiment_volatility"):
        df[col] = df.groupby("symbol")[col].ffill()

    return df


# ---------------------------------------------------------------------------
# Step 5 – Rolling / time-series features
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rolling features in grouped transforms.
    Sorting by (symbol, date) is required before calling this function.
    """
    log.info("Computing rolling features...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    grp = df.groupby("symbol", sort=False)["day_avg_sentiment"]

    df["sma_3"]   = grp.transform(lambda x: x.rolling(3,  min_periods=1).mean())
    df["sma_7"]   = grp.transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["ewma_7"]  = grp.transform(lambda x: x.ewm(span=7, adjust=False).mean())

    roll30_mean = grp.transform(lambda x: x.rolling(30, min_periods=1).mean())
    roll30_std  = grp.transform(lambda x: x.rolling(30, min_periods=2).std())

    # Z-score: 0.0 where std is undefined (< 2 observations) — neutral baseline
    df["z_score_30"] = np.where(
        roll30_std.isna() | (roll30_std == 0),
        0.0,
        (df["day_avg_sentiment"] - roll30_mean) / roll30_std,
    )

    df["momentum_1d"] = grp.transform(lambda x: x.diff(1))

    return df


# ---------------------------------------------------------------------------
# Step 6 – Final cleanup & dtype optimisation
# ---------------------------------------------------------------------------
def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf artefacts and downcast floats to save ~40 % memory."""
    df = df.replace([np.inf, -np.inf], np.nan)

    # Downcast float64 → float32 (sufficient precision for sentiment scores)
    f64_cols = df.select_dtypes("float64").columns
    df[f64_cols] = df[f64_cols].astype("float32")

    log.info("Final shape: %s", df.shape)
    log.info("Dtypes:\n%s", df.dtypes.to_string())
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    df      = load_and_explode(SOURCE_DATASET, HF_TOKEN)
    daily   = aggregate_daily(df)
    filled  = fill_calendar_gaps(daily)
    cleaned = apply_corrections(filled)
    featured = add_rolling_features(cleaned)
    final   = finalise(featured)

    log.info("Uploading to %s ...", TARGET_DATASET)
    final_ds = Dataset.from_pandas(final, preserve_index=False)
    final_ds.push_to_hub(TARGET_DATASET, token=HF_TOKEN)
    log.info("Upload complete! 🎉")


if __name__ == "__main__":
    main()