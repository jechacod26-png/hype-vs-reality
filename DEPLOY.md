# Guía de Deploy — Hype vs Reality

## Opción A: Streamlit Community Cloud (gratis, recomendado)

### 1. Subir a GitHub
```bash
git init
git add .
git commit -m "feat: initial hype detector project"
git remote add origin https://github.com/TU_USUARIO/hype-vs-reality.git
git push -u origin main
```

### 2. Conectar Streamlit Cloud
1. Ve a https://share.streamlit.io
2. "New app" → conecta tu repo GitHub
3. En "Main file path" escribe: `streamlit_app/app.py`
4. Click "Deploy"

URL pública gratis: `https://TU_USUARIO-hype-vs-reality-streamlit-appapp-XXXXX.streamlit.app`

### 3. Agregar secrets (API keys)
En Streamlit Cloud → Settings → Secrets:
```toml
YOUTUBE_API_KEY = "AIza..."
REDDIT_CLIENT_ID = "..."
REDDIT_SECRET = "..."
REDDIT_USER_AGENT = "HypeDetector/1.0"
```

---

## Opción B: GitHub Actions (automatización semanal)

### 1. Agregar secrets en GitHub
Settings → Secrets and variables → Actions → New repository secret:
- `YOUTUBE_API_KEY`
- `REDDIT_CLIENT_ID`
- `REDDIT_SECRET`

### 2. Activar el workflow
El archivo `.github/workflows/weekly_refresh.yml` ya está listo.
Se ejecutará automáticamente cada lunes a las 08:00 UTC.
También puedes correrlo manualmente: Actions → Weekly Hype Data Refresh → Run workflow.

---

## Cómo obtener las API Keys

### YouTube Data API v3
1. https://console.cloud.google.com
2. Crear proyecto → Enable "YouTube Data API v3"
3. Credentials → Create API Key
4. Cuota gratuita: 10,000 unidades/día (suficiente para ~100 búsquedas)

### Reddit API
1. https://www.reddit.com/prefs/apps
2. "Create App" → tipo: "script"
3. Anota el `client_id` (bajo el nombre de la app) y el `secret`
4. Sin límite de rate para uso personal moderado

### Google Trends
No requiere API key — usa `pytrends` que hace scraping directamente.
Límite: ~1 request/segundo para evitar bloqueos.

### Steam API
No requiere key para los endpoints públicos usados en este proyecto.

---

## Correr localmente
```bash
# Instalar dependencias
pip install -r requirements.txt

# Correr app
streamlit run streamlit_app/app.py

# Recolectar datos frescos (requiere API keys en .env)
export YOUTUBE_API_KEY="..."
export REDDIT_CLIENT_ID="..."
export REDDIT_SECRET="..."
python api_collectors/collector.py --batch api_collectors/upcoming_games.json

# Pipeline completo
python data/build_dataset.py
python notebooks/01_eda_features.py
python notebooks/02_model_training.py
python notebooks/03_predict_upcoming.py
python notebooks/05_export_for_streamlit.py
```

---

## Estructura final del proyecto

```
hype-vs-reality/
├── .github/
│   └── workflows/
│       └── weekly_refresh.yml    ← Automatización semanal
├── .streamlit/
│   └── config.toml               ← Tema dark personalizado
├── api_collectors/
│   ├── collector.py              ← Integración YouTube/Reddit/Steam/Trends
│   └── upcoming_games.json       ← Lista de juegos a monitorear
├── data/
│   ├── games_engineered.csv      ← Dataset histórico (180 juegos)
│   ├── stats.json                ← Estadísticas agregadas
│   ├── model_results.json        ← Métricas del modelo
│   └── upcoming_predictions.json ← Predicciones 2025 (se actualiza semanalmente)
├── models/
│   └── hype_model.pkl            ← Modelo serializado
├── notebooks/
│   ├── 01_eda_features.py
│   ├── 02_model_training.py
│   ├── 03_predict_upcoming.py
│   ├── 04_merge_real_data.py
│   └── 05_export_for_streamlit.py
├── streamlit_app/
│   ├── app.py                    ← App principal (5 tabs)
│   ├── data/                     ← Copia de data/ para Streamlit
│   └── models/                   ← Copia del modelo
├── requirements.txt
└── README.md
```
