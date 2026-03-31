"""
FASE 3 — Modelo Predictivo ML
Entrena un Random Forest + Gradient Boosting para predecir
si un juego va a decepcionar ANTES de su lanzamiento.
"""
import os
import pandas as pd
import numpy as np
import json
import pickle
import math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "games_engineered.csv"))

print("=" * 60)
print("HYPE DETECTOR — Model Training")
print("=" * 60)

# ─── Features para el modelo ────────────────────────────────
FEATURES = [
    "hype_score",
    "google_trends_peak",
    "trailer_views_m",
    "like_ratio",
    "reddit_mentions_k",
    "reddit_sentiment_pre",
    "steam_wishlist_k",
    "press_coverage_score",
    "dev_track_record",
    "marketing_days",
    "had_beta",
    "price_usd",
    "hype_vs_genre_avg",
    "marketing_intensity",
    "hype_sentiment_gap",
    "wishlist_per_day",
    "is_premium",
    "is_aaa",
    "years_ago",
    "genre_avg_delta",
]

TARGET = "label_binary"  # 1 = decepcionó, 0 = no decepcionó

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\nDataset: {len(df)} juegos | Features: {len(FEATURES)}")
print(f"Positivos (decepcionó): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Negativos (no decepcionó): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

# ─── Cross-validation temporal ──────────────────────────────
print("\n\n📊 Cross-Validation (5-fold estratificado):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    ),
}

results = {}
for name, model in models.items():
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1")
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    results[name] = {
        "auc_mean": auc_scores.mean(),
        "auc_std": auc_scores.std(),
        "f1_mean": f1_scores.mean(),
        "acc_mean": acc_scores.mean(),
    }
    print(f"\n  {name}:")
    print(f"    AUC-ROC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"    F1:      {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    print(f"    Acc:     {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

# ─── Elegir mejor modelo y entrenar full dataset ─────────────
best_model_name = max(results, key=lambda k: results[k]["auc_mean"])
print(f"\n\n🏆 Mejor modelo: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X, y)

# ─── Feature Importance ─────────────────────────────────────
print("\n\n📌 Feature Importance (permutation-based):")

# Para modelos con feature_importances_ nativo
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "named_steps"):
    clf = best_model.named_steps.get("clf")
    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        # rescalar para que sumen 1
        importances = importances / importances.sum()
    else:
        importances = np.ones(len(FEATURES)) / len(FEATURES)
else:
    importances = np.ones(len(FEATURES)) / len(FEATURES)

feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
for feat, imp in feat_imp:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f} {bar}")

# ─── Evaluación detallada en train (para reportar) ──────────
y_pred = best_model.predict(X)
y_proba = best_model.predict_proba(X)[:, 1]

print("\n\n📋 Reporte de clasificación (train):")
print(classification_report(y, y_pred, target_names=["No decepciona", "Decepciona"]))

final_auc = roc_auc_score(y, y_proba)
print(f"AUC-ROC final (train): {final_auc:.3f}")

# ─── Matriz de confusión ────────────────────────────────────
cm = confusion_matrix(y, y_pred)
print("\nMatriz de confusión:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

# ─── Predicciones con probabilidad en todos los juegos reales
real = df[~df["name"].str.startswith("Game_")].copy()
X_real = real[FEATURES]
real["proba_disappointment"] = best_model.predict_proba(X_real)[:, 1].round(4)
real["predicted_label"] = best_model.predict(X_real)

print("\n\n🎯 Predicciones en juegos reales conocidos:")
show_cols = ["name", "year", "hype_score", "review_score_combined",
             "delta", "label", "proba_disappointment"]
print(real[show_cols].sort_values("proba_disappointment", ascending=False)
      .head(20).to_string(index=False))

# ─── Guardar modelo y artefactos ─────────────────────────────
import pickle

model_data = {
    "model": best_model,
    "features": FEATURES,
    "model_name": best_model_name,
    "cv_results": results,
    "feature_importance": dict(feat_imp),
    "train_auc": final_auc,
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "games_engineered.csv")) 
    pickle.dump(model_data, f)

# Exportar resultados para el dashboard
model_results = {
    "model_name": best_model_name,
    "cv_results": {
        k: {kk: round(vv, 4) for kk, vv in v.items()}
        for k, v in results.items()
    },
    "feature_importance": {k: round(float(v), 4) for k, v in feat_imp},
    "confusion_matrix": cm.tolist(),
    "train_auc": round(final_auc, 4),
    "real_game_predictions": real[show_cols + ["publisher_tier", "genre"]].to_dict("records"),
    "all_model_scores": {
        name: {
            "auc": round(r["auc_mean"], 3),
            "f1":  round(r["f1_mean"], 3),
            "acc": round(r["acc_mean"], 3),
        }
        for name, r in results.items()
    }
}

with openBASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "games_engineered.csv")) as f:
    json.dump(model_results, f, indent=2)

print("\n\n💾 Modelo guardado en models/hype_model.pkl")
print("📊 Resultados exportados a data/model_results.json")
print("\n✅ Training completo.")
