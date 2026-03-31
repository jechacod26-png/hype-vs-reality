"""
FASE 4B — Merge de datos reales con dataset histórico
Combina los datos recolectados por el API collector
con el dataset histórico entrenado.
"""

import pandas as pd
import numpy as np
import json
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_collected_data():
    """Carga todos los JSONs recolectados por el collector."""
    pattern = os.path.join(DATA_DIR, "collected", "*.json")
    files = glob.glob(pattern)
    if not files:
        print("No hay datos recolectados aún. Corre collector.py primero.")
        return pd.DataFrame()

    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))
    print(f"Cargados {len(records)} juegos recolectados desde APIs reales.")
    return pd.DataFrame(records)


def normalize_collected(df_raw):
    """
    Transforma los datos crudos de APIs al formato del dataset histórico.
    Aplica las mismas transformaciones que build_dataset.py
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # Rellenar valores faltantes con medias del dataset histórico
    defaults = {
        "trailer_views_m": 10.0,
        "like_ratio": 0.85,
        "reddit_mentions_k": 20.0,
        "reddit_sentiment_pre": 0.72,
        "google_trends_peak": 50.0,
        "steam_wishlist_k": 500.0,
        "price_usd": 60.0,
        "had_beta": 0,
        "marketing_days": 365,
        "press_coverage_score": 7.0,
        "dev_track_record": 7.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # Feature engineering (mismo que en EDA)
    genre_hype_baseline = {
        "RPG": 7.1, "Shooter": 6.8, "Action": 7.0, "Platformer": 6.2,
        "Strategy": 5.8, "Sports": 6.0, "Horror": 6.5, "Simulation": 5.5,
        "Fighting": 6.8, "Adventure": 6.3, "Roguelike": 5.9,
    }
    genre_delta_baseline = {
        "RPG": 0.1, "Shooter": 0.5, "Action": -0.1, "Platformer": -1.2,
        "Strategy": -0.5, "Sports": 0.2, "Horror": -0.3, "Simulation": -0.8,
        "Fighting": 0.0, "Adventure": -0.6, "Roguelike": -1.5,
    }

    if "hype_score" not in df.columns:
        # Derivar hype de las métricas (si no está explícito)
        df["hype_score"] = (
            df["google_trends_peak"] / 100 * 3 +
            (df["trailer_views_m"] / 30).clip(0, 1) * 3 +
            df["reddit_sentiment_pre"] * 2 +
            (df["reddit_mentions_k"] / 100).clip(0, 1) * 2
        ).clip(1, 10).round(2)

    if "genre" not in df.columns:
        df["genre"] = "Action"
    if "publisher_tier" not in df.columns:
        df["publisher_tier"] = "AA"

    df["hype_vs_genre_avg"] = df.apply(
        lambda r: round(r["hype_score"] - genre_hype_baseline.get(r.get("genre", "Action"), 6.5), 3),
        axis=1
    )
    df["marketing_intensity"] = (
        (df["google_trends_peak"] / 100) * 0.35 +
        (df["trailer_views_m"] / 30).clip(0, 1) * 0.35 +
        (df["reddit_mentions_k"] / 100).clip(0, 1) * 0.30
    ).round(4)
    df["hype_sentiment_gap"] = (df["hype_score"] / 10 - df["reddit_sentiment_pre"]).round(4)
    df["wishlist_per_day"]   = (df["steam_wishlist_k"] / (df["marketing_days"] + 1)).round(3)
    df["is_premium"]         = (df["price_usd"] >= 60).astype(int)
    df["is_aaa"]             = (df.get("publisher_tier", "AA") == "AAA").astype(int)
    df["years_ago"]          = 2025 - df.get("year", 2025)
    df["genre_avg_delta"]    = df.get("genre", "Action").map(
        lambda g: genre_delta_baseline.get(g, 0.0)
    )

    return df


def merge_with_historical(df_new):
    """Agrega juegos nuevos al dataset histórico sin duplicar."""
    hist_path = os.path.join(DATA_DIR, "games_engineered.csv")
    df_hist = pd.read_csv(hist_path)

    # Solo agregar juegos que no estén ya en el histórico
    existing_names = set(df_hist["name"].str.lower())
    df_new_filtered = df_new[~df_new["name"].str.lower().isin(existing_names)]

    if df_new_filtered.empty:
        print("No hay juegos nuevos para agregar al histórico.")
        return df_hist

    # Alinear columnas
    cols = df_hist.columns.tolist()
    for col in cols:
        if col not in df_new_filtered.columns:
            df_new_filtered[col] = np.nan

    df_merged = pd.concat([df_hist, df_new_filtered[cols]], ignore_index=True)
    df_merged.to_csv(hist_path, index=False)
    print(f"Dataset actualizado: {len(df_hist)} → {len(df_merged)} juegos (+{len(df_new_filtered)} nuevos)")
    return df_merged


if __name__ == "__main__":
    df_raw = load_collected_data()
    if not df_raw.empty:
        df_normalized = normalize_collected(df_raw)
        df_final = merge_with_historical(df_normalized)
        print(f"\nDataset final: {len(df_final)} juegos, {df_final.shape[1]} columnas")
    else:
        print("No se encontraron datos recolectados. Corre collector.py primero.")
