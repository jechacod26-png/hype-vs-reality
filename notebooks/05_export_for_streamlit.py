"""
FASE 5 — Export para Streamlit
Copia y prepara los archivos de datos que necesita la app de Streamlit.
"""
import os, shutil, json, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "..", "data")
DST  = os.path.join(BASE, "..", "streamlit_app", "data")
os.makedirs(DST, exist_ok=True)

FILES = [
    "games_engineered.csv",
    "stats.json",
    "upcoming_predictions.json",
    "model_results.json",
]

for f in FILES:
    src = os.path.join(SRC, f)
    dst = os.path.join(DST, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copiado: {f}")
    else:
        print(f"  No encontrado: {f}")

# Copiar el modelo
model_src = os.path.join(BASE, "..", "models", "hype_model.pkl")
model_dst = os.path.join(BASE, "..", "streamlit_app", "models")
os.makedirs(model_dst, exist_ok=True)
if os.path.exists(model_src):
    shutil.copy2(model_src, os.path.join(model_dst, "hype_model.pkl"))
    print("  Copiado: hype_model.pkl")

print("\nExport completo. La app de Streamlit está lista.")
print("Para correr localmente: streamlit run streamlit_app/app.py")
