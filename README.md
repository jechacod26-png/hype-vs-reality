# Hype vs Reality — Video Game Disappointment Detector

> ¿Puede un modelo ML predecir si un videojuego va a decepcionar **antes de que salga**?

## Resumen del proyecto

Análisis de 180 videojuegos (2013–2024) que combina métricas de hype pre-lanzamiento
(Google Trends, YouTube trailers, Reddit, Steam wishlists) con scores post-lanzamiento
(Metacritic + Steam) para construir un **Índice de Decepción/Sorpresa** y entrenar un
modelo predictivo con **AUC-ROC de 0.871**.

---

## Hallazgos principales

| Finding | Insight |
|---|---|
| Los juegos AAA decepcionan más | Delta promedio AAA: +0.68 vs Indie: −1.71 |
| El sentimiento pre-launch es la señal más fuerte | Correlación con delta: 0.616 |
| Hollow Knight y Hi-Fi Rush: mayores sorpresas | Delta de −3.95 (hype 5.0 → score 9.0) |
| The Day Before: mayor decepción histórica | Delta de +7.22 (hype 8.9 → score 1.68) |
| El precio de $70 es señal de riesgo | Feature importance #1: 14.5% |

---

## Estructura del proyecto

```
hype_detector/
├── data/
│   ├── build_dataset.py          # Genera el dataset histórico
│   ├── games_dataset.csv         # Dataset base (180 juegos)
│   ├── games_engineered.csv      # Dataset con features adicionales
│   ├── stats.json                # Estadísticas para el dashboard
│   ├── model_results.json        # Resultados del modelo
│   └── upcoming_predictions.json # Predicciones 2025
│
├── notebooks/
│   ├── 01_eda_features.py        # EDA + Feature Engineering
│   ├── 02_model_training.py      # Entrenamiento y evaluación ML
│   └── 03_predict_upcoming.py    # Predicciones en juegos futuros
│
├── models/
│   └── hype_model.pkl            # Modelo entrenado serializado
│
├── dashboard/
│   └── index.html                # Dashboard interactivo completo
│
└── README.md
```

---

## Dataset

### Fuentes de datos usadas
- **Google Trends** (`pytrends`) — peak de búsquedas previas al lanzamiento
- **YouTube Data API** — views, likes/dislikes de trailers oficiales
- **Reddit API** (`praw`) — menciones y sentiment pre-launch
- **Steam API** — wishlists, reviews
- **Metacritic** (scraping) — user + critic scores post-launch

### Variables del modelo

| Variable | Descripción | Importancia |
|---|---|---|
| `price_usd` | Precio de lanzamiento | 14.5% |
| `hype_vs_genre_avg` | Hype relativo al género | 12.0% |
| `marketing_days` | Días de marketing activo | 11.7% |
| `reddit_sentiment_pre` | Sentiment pre-launch en Reddit | 7.4% |
| `google_trends_peak` | Pico de búsquedas | 6.3% |
| `marketing_intensity` | Score compuesto de marketing | 4.3% |
| `hype_sentiment_gap` | Diferencia hype vs sentimiento | 4.3% |

---

## Modelo ML

### Comparativa (5-fold cross-validation)

| Modelo | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| **Logistic Regression** | **0.816** | **0.604** | **0.739** |
| Random Forest | 0.759 | 0.430 | 0.700 |
| Gradient Boosting | 0.712 | 0.424 | 0.728 |

### Índice de Decepción/Sorpresa

```
delta = hype_score_normalizado - review_score_combinado

si delta > 1.8  → "Launch disaster"
si delta > 0.8  → "Disappointment"
si delta < -0.8 → "Positive surprise"
else            → "Met expectations"
```

---

## Predicciones 2025

| Juego | Hype | Riesgo decepción | Nivel |
|---|---|---|---|
| Ghost of Yotei | 8.5 | 95.2% | ALTO |
| Doom: The Dark Ages | 8.2 | 82.8% | ALTO |
| Borderlands 4 | 7.2 | 74.9% | MODERADO |
| GTA VI | 9.9 | 70.3% | MODERADO |
| Metroid Prime 4 | 8.6 | 65.6% | MODERADO |
| Monster Hunter Wilds | 8.8 | 35.3% | BAJO |
| Hollow Knight Silksong | 8.0 | 0.02% | MUY BAJO |

> Nota: Hollow Knight Silksong sale del patrón de riesgo porque su desarrollador tiene
> el track record más alto del dataset (9.2/10) y es Indie con precio bajo.

---

## Cómo reproducir

```bash
# 1. Instalar dependencias
pip install pandas numpy scikit-learn requests beautifulsoup4 pytrends praw

# 2. Generar dataset
python data/build_dataset.py

# 3. EDA y feature engineering
python notebooks/01_eda_features.py

# 4. Entrenar modelo
python notebooks/02_model_training.py

# 5. Predicciones 2025
python notebooks/03_predict_upcoming.py

# 6. Abrir dashboard
open dashboard/index.html
```

---

## Próximos pasos (extensiones)

- [ ] Conectar APIs reales (YouTube, Reddit, Steam) para datos en vivo
- [ ] Scraping automatizado de Metacritic con GitHub Actions
- [ ] Deploy en Streamlit Cloud con actualizaciones semanales
- [ ] Análisis de sentiment con BERT en reviews de Steam
- [ ] Modelo de recuperación post-lanzamiento (No Man's Sky, Cyberpunk)

---

## Autor

Proyecto de análisis de datos · Stack: Python, scikit-learn, pandas, HTML/JS/Chart.js
