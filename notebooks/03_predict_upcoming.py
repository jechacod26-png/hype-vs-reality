"""
FASE 4 — Predictor de Juegos Futuros (2025-2026)
Aplica el modelo a juegos próximos a lanzarse.
"""
import os
import pandas as pd
import numpy as np
import pickle
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "models", "hype_model.pkl"), "rb") as f:
      json.dump(predictions_out, f, indent=2)

model    = model_data["model"]
FEATURES = model_data["features"]

# ─── Juegos próximos a lanzarse 2025-2026 ───────────────────
# Estimaciones basadas en información pública disponible
UPCOMING_GAMES = [
    {
        "name": "GTA VI",
        "year": 2025,
        "genre": "Action",
        "publisher_tier": "AAA",
        "hype_score": 9.9,
        "google_trends_peak": 99.0,
        "trailer_views_m": 75.0,
        "like_ratio": 0.97,
        "reddit_mentions_k": 142.0,
        "reddit_sentiment_pre": 0.88,
        "steam_wishlist_k": 1200.0,
        "press_coverage_score": 9.8,
        "dev_track_record": 9.5,
        "marketing_days": 730,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Monster Hunter Wilds",
        "year": 2025,
        "genre": "RPG",
        "publisher_tier": "AAA",
        "hype_score": 8.8,
        "google_trends_peak": 82.0,
        "trailer_views_m": 28.0,
        "like_ratio": 0.96,
        "reddit_mentions_k": 68.0,
        "reddit_sentiment_pre": 0.87,
        "steam_wishlist_k": 3800.0,
        "press_coverage_score": 8.5,
        "dev_track_record": 8.8,
        "marketing_days": 540,
        "had_beta": 1,
        "price_usd": 60,
    },
    {
        "name": "Death Stranding 2",
        "year": 2025,
        "genre": "Adventure",
        "publisher_tier": "AA",
        "hype_score": 7.8,
        "google_trends_peak": 70.0,
        "trailer_views_m": 18.0,
        "like_ratio": 0.91,
        "reddit_mentions_k": 42.0,
        "reddit_sentiment_pre": 0.75,
        "steam_wishlist_k": 950.0,
        "press_coverage_score": 8.0,
        "dev_track_record": 7.5,
        "marketing_days": 450,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Borderlands 4",
        "year": 2025,
        "genre": "Shooter",
        "publisher_tier": "AAA",
        "hype_score": 7.2,
        "google_trends_peak": 62.0,
        "trailer_views_m": 15.0,
        "like_ratio": 0.88,
        "reddit_mentions_k": 35.0,
        "reddit_sentiment_pre": 0.72,
        "steam_wishlist_k": 820.0,
        "press_coverage_score": 7.2,
        "dev_track_record": 7.0,
        "marketing_days": 360,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Ghost of Yotei",
        "year": 2025,
        "genre": "Action",
        "publisher_tier": "AAA",
        "hype_score": 8.5,
        "google_trends_peak": 78.0,
        "trailer_views_m": 22.0,
        "like_ratio": 0.95,
        "reddit_mentions_k": 55.0,
        "reddit_sentiment_pre": 0.86,
        "steam_wishlist_k": 0.0,
        "press_coverage_score": 8.2,
        "dev_track_record": 8.5,
        "marketing_days": 300,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Metroid Prime 4",
        "year": 2025,
        "genre": "Action",
        "publisher_tier": "AAA",
        "hype_score": 8.6,
        "google_trends_peak": 75.0,
        "trailer_views_m": 20.0,
        "like_ratio": 0.95,
        "reddit_mentions_k": 50.0,
        "reddit_sentiment_pre": 0.85,
        "steam_wishlist_k": 0.0,
        "press_coverage_score": 8.0,
        "dev_track_record": 9.0,
        "marketing_days": 480,
        "had_beta": 0,
        "price_usd": 60,
    },
    {
        "name": "Fable (2025)",
        "year": 2025,
        "genre": "RPG",
        "publisher_tier": "AAA",
        "hype_score": 7.5,
        "google_trends_peak": 60.0,
        "trailer_views_m": 12.0,
        "like_ratio": 0.87,
        "reddit_mentions_k": 38.0,
        "reddit_sentiment_pre": 0.71,
        "steam_wishlist_k": 0.0,
        "press_coverage_score": 7.5,
        "dev_track_record": 6.5,
        "marketing_days": 540,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Hollow Knight Silksong",
        "year": 2025,
        "genre": "Platformer",
        "publisher_tier": "Indie",
        "hype_score": 8.0,
        "google_trends_peak": 55.0,
        "trailer_views_m": 8.0,
        "like_ratio": 0.97,
        "reddit_mentions_k": 62.0,
        "reddit_sentiment_pre": 0.89,
        "steam_wishlist_k": 4200.0,
        "press_coverage_score": 7.8,
        "dev_track_record": 9.2,
        "marketing_days": 1200,
        "had_beta": 0,
        "price_usd": 20,
    },
    {
        "name": "Doom: The Dark Ages",
        "year": 2025,
        "genre": "Shooter",
        "publisher_tier": "AAA",
        "hype_score": 8.2,
        "google_trends_peak": 72.0,
        "trailer_views_m": 18.0,
        "like_ratio": 0.94,
        "reddit_mentions_k": 44.0,
        "reddit_sentiment_pre": 0.84,
        "steam_wishlist_k": 1100.0,
        "press_coverage_score": 8.0,
        "dev_track_record": 8.8,
        "marketing_days": 420,
        "had_beta": 0,
        "price_usd": 70,
    },
    {
        "name": "Mafia: The Old Country",
        "year": 2025,
        "genre": "Action",
        "publisher_tier": "AA",
        "hype_score": 7.0,
        "google_trends_peak": 55.0,
        "trailer_views_m": 10.0,
        "like_ratio": 0.89,
        "reddit_mentions_k": 28.0,
        "reddit_sentiment_pre": 0.78,
        "steam_wishlist_k": 450.0,
        "press_coverage_score": 7.0,
        "dev_track_record": 7.2,
        "marketing_days": 360,
        "had_beta": 0,
        "price_usd": 50,
    },
]

df_up = pd.DataFrame(UPCOMING_GAMES)

# ─── Feature Engineering (mismo que entrenamiento) ──────────
# Valores de referencia del dataset histórico
genre_hype_baseline = {
    "RPG": 7.1, "Shooter": 6.8, "Action": 7.0, "Platformer": 6.2,
    "Strategy": 5.8, "Sports": 6.0, "Horror": 6.5, "Simulation": 5.5,
    "Fighting": 6.8, "Adventure": 6.3, "Roguelike": 5.9, "Battle": 6.5
}
genre_delta_baseline = {
    "RPG": 0.1, "Shooter": 0.5, "Action": -0.1, "Platformer": -1.2,
    "Strategy": -0.5, "Sports": 0.2, "Horror": -0.3, "Simulation": -0.8,
    "Fighting": 0.0, "Adventure": -0.6, "Roguelike": -1.5, "Battle": -0.3
}
max_trailer_views = 32.98
max_reddit_mentions = 145.0

df_up["hype_vs_genre_avg"] = df_up.apply(
    lambda r: round(r["hype_score"] - genre_hype_baseline.get(r["genre"], 6.5), 3), axis=1
)
df_up["marketing_intensity"] = (
    (df_up["google_trends_peak"] / 100) * 0.35 +
    (df_up["trailer_views_m"] / max_trailer_views).clip(0, 1) * 0.35 +
    (df_up["reddit_mentions_k"] / max_reddit_mentions).clip(0, 1) * 0.30
).round(4)
df_up["hype_sentiment_gap"] = (df_up["hype_score"] / 10 - df_up["reddit_sentiment_pre"]).round(4)
df_up["wishlist_per_day"] = (df_up["steam_wishlist_k"] / (df_up["marketing_days"] + 1)).round(3)
df_up["is_premium"] = (df_up["price_usd"] >= 60).astype(int)
df_up["is_aaa"] = (df_up["publisher_tier"] == "AAA").astype(int)
df_up["years_ago"] = 2025 - df_up["year"]
df_up["genre_avg_delta"] = df_up["genre"].map(genre_delta_baseline).fillna(0.0)

# ─── Predicción ─────────────────────────────────────────────
X_up = df_up[FEATURES]
probas = model.predict_proba(X_up)[:, 1]
preds  = model.predict(X_up)

df_up["proba_disappointment"] = np.round(probas, 4)
df_up["prediction"] = np.where(preds == 1, "⚠️ Riesgo de decepción", "✅ Probablemente bueno")

# Nivel de riesgo
def risk_level(p):
    if p >= 0.75: return "ALTO"
    if p >= 0.50: return "MODERADO"
    if p >= 0.30: return "BAJO"
    return "MUY BAJO"

df_up["risk_level"] = df_up["proba_disappointment"].apply(risk_level)

# ─── Reporte ─────────────────────────────────────────────────
print("=" * 60)
print("PREDICCIONES — Juegos Próximos 2025-2026")
print("=" * 60)

show = df_up[["name", "year", "publisher_tier", "hype_score",
              "proba_disappointment", "risk_level", "prediction"]].sort_values(
    "proba_disappointment", ascending=False
)
print(show.to_string(index=False))

# ─── Exportar ────────────────────────────────────────────────
predictions_out = df_up[[
    "name", "year", "genre", "publisher_tier", "hype_score",
    "google_trends_peak", "trailer_views_m", "reddit_sentiment_pre",
    "steam_wishlist_k", "marketing_days", "had_beta", "price_usd",
    "proba_disappointment", "risk_level", "prediction"
]].to_dict("records")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "models", "hype_model.pkl"), "rb") as f:
    json.dump(predictions_out, f, indent=2)

print("\n✅ Predicciones exportadas.")
