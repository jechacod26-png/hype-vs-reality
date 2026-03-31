"""
FASE 2 — Análisis Exploratorio (EDA) y Feature Engineering
Hype Detector Project
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

df = pd.read_csv("/home/claude/hype_detector/data/games_dataset.csv")

print("=" * 60)
print("HYPE DETECTOR — EDA Report")
print("=" * 60)

# ─── 1. Overview ────────────────────────────────────────────
print(f"\n📊 Dataset: {len(df)} juegos | {df.shape[1]} variables")
print(f"   Años: {df['year'].min()} - {df['year'].max()}")
print(f"   Géneros únicos: {df['genre'].nunique()}")

# ─── 2. Estadísticas de las variables clave ─────────────────
print("\n📈 Estadísticas clave:")
key_cols = ["hype_score", "review_score_combined", "delta",
            "google_trends_peak", "trailer_views_m", "reddit_sentiment_pre"]
print(df[key_cols].describe().round(2).to_string())

# ─── 3. Correlaciones con delta (target) ────────────────────
print("\n🔗 Correlación de features con el DELTA (hype - realidad):")
feature_cols = [
    "hype_score", "google_trends_peak", "trailer_views_m", "like_ratio",
    "reddit_mentions_k", "reddit_sentiment_pre", "steam_wishlist_k",
    "press_coverage_score", "dev_track_record", "marketing_days",
    "had_beta", "price_usd"
]
correlations = df[feature_cols].corrwith(df["delta"]).sort_values(ascending=False)
for col, corr in correlations.items():
    bar = "█" * int(abs(corr) * 20)
    sign = "+" if corr >= 0 else "-"
    print(f"  {col:<30} {sign}{abs(corr):.3f} {bar}")

# ─── 4. Análisis por label ──────────────────────────────────
print("\n🎮 Análisis por categoría:")
for label in ["launch_disaster", "disappointment", "met_expectations", "positive_surprise"]:
    sub = df[df["label"] == label]
    print(f"\n  [{label.upper()}] n={len(sub)}")
    print(f"    Hype promedio:   {sub['hype_score'].mean():.2f}")
    print(f"    Review promedio: {sub['review_score_combined'].mean():.2f}")
    print(f"    Delta promedio:  {sub['delta'].mean():.2f}")
    print(f"    Google Trends:   {sub['google_trends_peak'].mean():.1f}")
    print(f"    Publisher AAA%:  {(sub['publisher_tier']=='AAA').mean()*100:.0f}%")

# ─── 5. Top decepciones y sorpresas (juegos reales) ─────────
real = df[~df["name"].str.startswith("Game_")].copy()

print("\n\n💥 Top 10 MAYORES DECEPCIONES (delta más alto):")
top_dec = real.nlargest(10, "delta")[["name", "year", "hype_score", "review_score_combined", "delta", "publisher_tier"]]
print(top_dec.to_string(index=False))

print("\n\n✨ Top 10 SORPRESAS POSITIVAS (delta más bajo):")
top_sur = real.nsmallest(10, "delta")[["name", "year", "hype_score", "review_score_combined", "delta", "publisher_tier"]]
print(top_sur.to_string(index=False))

# ─── 6. Feature Engineering ─────────────────────────────────
print("\n\n🔧 Feature Engineering...")

# Feature 1: Hype intensity ratio (qué tan extremo es el hype vs el promedio del género)
genre_hype_mean = df.groupby("genre")["hype_score"].transform("mean")
df["hype_vs_genre_avg"] = (df["hype_score"] - genre_hype_mean).round(3)

# Feature 2: Marketing intensity score (trends + views + mentions combinados)
df["marketing_intensity"] = (
    (df["google_trends_peak"] / 100) * 0.35 +
    (df["trailer_views_m"] / df["trailer_views_m"].max()) * 0.35 +
    (df["reddit_mentions_k"] / df["reddit_mentions_k"].max()) * 0.30
).round(4)

# Feature 3: Sentiment gap (alto hype pero bajo sentimiento = señal de alerta)
hype_norm = df["hype_score"] / 10
df["hype_sentiment_gap"] = (hype_norm - df["reddit_sentiment_pre"]).round(4)

# Feature 4: Wishlist efficiency (wishlists por día de marketing)
df["wishlist_per_day"] = (df["steam_wishlist_k"] / (df["marketing_days"] + 1)).round(3)

# Feature 5: Premium game flag (precio alto = más presión de expectativas)
df["is_premium"] = (df["price_usd"] >= 60).astype(int)

# Feature 6: AAA flag
df["is_aaa"] = (df["publisher_tier"] == "AAA").astype(int)

# Feature 7: Year recency (juegos recientes tienden a tener más hype digital)
df["years_ago"] = 2025 - df["year"]

# Feature 8: Encoded genre
genre_hype_avg = df.groupby("genre")["delta"].mean()
df["genre_avg_delta"] = df["genre"].map(genre_hype_avg).round(3)

print("  Features creadas: hype_vs_genre_avg, marketing_intensity,")
print("  hype_sentiment_gap, wishlist_per_day, is_premium, is_aaa,")
print("  years_ago, genre_avg_delta")

# ─── 7. Missing values check ────────────────────────────────
print(f"\n✅ Missing values: {df.isnull().sum().sum()}")

# ─── 8. Guardar dataset enriquecido ─────────────────────────
df.to_csv("/home/claude/hype_detector/data/games_engineered.csv", index=False)
df.to_json("/home/claude/hype_detector/data/games_engineered.json", orient="records", indent=2)

print(f"\n💾 Dataset enriquecido guardado: {df.shape[1]} columnas")

# ─── 9. Exportar estadísticas para el dashboard ─────────────
stats = {
    "total_games": len(df),
    "real_games": len(real),
    "label_counts": df["label"].value_counts().to_dict(),
    "genre_avg_delta": df.groupby("genre")["delta"].mean().round(2).to_dict(),
    "year_avg_delta": df.groupby("year")["delta"].mean().round(2).to_dict(),
    "tier_avg_delta": df.groupby("publisher_tier")["delta"].mean().round(2).to_dict(),
    "top_disappointments": real.nlargest(15, "delta")[
        ["name","year","genre","publisher_tier","hype_score","review_score_combined","delta","label"]
    ].to_dict("records"),
    "top_surprises": real.nsmallest(15, "delta")[
        ["name","year","genre","publisher_tier","hype_score","review_score_combined","delta","label"]
    ].to_dict("records"),
    "all_real_games": real[
        ["name","year","genre","publisher_tier","hype_score","metacritic_score",
         "steam_score","review_score_combined","delta","label",
         "google_trends_peak","trailer_views_m","reddit_sentiment_pre",
         "steam_wishlist_k"]
    ].to_dict("records"),
    "correlations": correlations.round(3).to_dict(),
}

with open("/home/claude/hype_detector/data/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("📊 Stats exportadas para el dashboard")
print("\n✅ EDA completo.")
