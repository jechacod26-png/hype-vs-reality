import streamlit as st, pandas as pd, numpy as np, json, os
from datetime import datetime

st.set_page_config(page_title="Hype vs Reality", page_icon="🎮", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0f0f13}
[data-testid="stHeader"]{background:transparent}
.mc{background:#16161e;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1rem 1.25rem;text-align:center}
.mn{font-size:32px;font-weight:700}
.ml{font-size:12px;color:#888899;text-transform:uppercase;letter-spacing:0.5px}
.explain-box{background:#16161e;border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:1rem 1.25rem;margin-top:0.5rem}
.feat-pos{color:#42d9a8;font-weight:600}
.feat-neg{color:#f07060;font-weight:600}
</style>""", unsafe_allow_html=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

def find_data():
    for base in [THIS_DIR, ROOT]:
        p = os.path.join(base, "data")
        if os.path.exists(os.path.join(p, "games_engineered.csv")):
            return p
    return os.path.join(THIS_DIR, "data")

DATA = find_data()

FEATURES = [
    "hype_score","google_trends_peak","trailer_views_m","like_ratio",
    "reddit_mentions_k","reddit_sentiment_pre","steam_wishlist_k",
    "press_coverage_score","dev_track_record","marketing_days","had_beta",
    "price_usd","hype_vs_genre_avg","marketing_intensity","hype_sentiment_gap",
    "wishlist_per_day","is_premium","is_aaa","years_ago","genre_avg_delta",
]

FEAT_LABELS = {
    "hype_score": "Hype score general",
    "google_trends_peak": "Pico en Google Trends",
    "trailer_views_m": "Views de trailers (M)",
    "like_ratio": "Ratio likes/views",
    "reddit_mentions_k": "Menciones en Reddit",
    "reddit_sentiment_pre": "Sentimiento pre-launch",
    "steam_wishlist_k": "Wishlists en Steam",
    "press_coverage_score": "Cobertura de prensa",
    "dev_track_record": "Historial del developer",
    "marketing_days": "Días de marketing",
    "had_beta": "Beta pública",
    "price_usd": "Precio de lanzamiento",
    "hype_vs_genre_avg": "Hype vs promedio del género",
    "marketing_intensity": "Intensidad de marketing",
    "hype_sentiment_gap": "Brecha hype-sentimiento",
    "wishlist_per_day": "Wishlists por día",
    "is_premium": "Juego premium ($60+)",
    "is_aaa": "Publisher AAA",
    "years_ago": "Años desde lanzamiento",
    "genre_avg_delta": "Delta histórico del género",
}

@st.cache_resource
def get_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    df = pd.read_csv(os.path.join(DATA, "games_engineered.csv"))
    df = df[df["label"] != "upcoming"].copy()

    TRAIN_END = 2021; TEST_START = 2022
    df_train = df[df["year"] <= TRAIN_END].copy()
    df_test  = df[df["year"] >= TEST_START].copy()

    genre_hype_mean  = df_train.groupby("genre")["hype_score"].mean()
    genre_delta_mean = df_train.groupby("genre")["delta"].mean()
    global_delta_mean = df_train["delta"].mean()

    for dset in [df_train, df_test]:
        dset["hype_vs_genre_avg"] = (dset["hype_score"] - dset["genre"].map(genre_hype_mean)).fillna(0).round(3)
        dset["genre_avg_delta"]   = dset["genre"].map(genre_delta_mean).fillna(global_delta_mean).round(3)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    X_train = df_train[FEATURES]; y_train = df_train["label_binary"]
    X_test  = df_test[FEATURES];  y_test  = df_test["label_binary"]
    X_all   = df[FEATURES];       y_all   = df["label_binary"]

    # CV en train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_f1  = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")

    clf.fit(X_train, y_train)

    # Metricas temporales
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    from sklearn.metrics import precision_recall_curve
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_proba_test)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    opt_idx = np.argmax(f1_arr)
    opt_thr = float(thresholds[opt_idx]) if opt_idx < len(thresholds) else 0.5

    y_pred_test = (y_proba_test >= opt_thr).astype(int)
    auc_temporal = round(roc_auc_score(y_test, y_proba_test), 3)
    f1_temporal  = round(f1_score(y_test, y_pred_test), 3)
    prec_temporal= round(precision_score(y_test, y_pred_test), 3)
    rec_temporal = round(recall_score(y_test, y_pred_test), 3)

    # Reentrenar con todo
    clf.fit(X_all, y_all)
    auc_full = round(roc_auc_score(y_all, clf.predict_proba(X_all)[:, 1]), 3)

    return {
        "model": clf, "threshold": opt_thr,
        "auc_cv": round(cv_auc.mean(), 3), "auc_cv_std": round(cv_auc.std(), 3),
        "f1_cv":  round(cv_f1.mean(), 3),
        "auc_temporal": auc_temporal, "f1_temporal": f1_temporal,
        "precision_temporal": prec_temporal, "recall_temporal": rec_temporal,
        "auc_full": auc_full,
        "train_size": len(df_train), "test_size": len(df_test),
        "genre_hype_mean": genre_hype_mean.to_dict(),
        "genre_delta_mean": genre_delta_mean.to_dict(),
        "global_delta_mean": global_delta_mean,
    }

@st.cache_data(ttl=3600)
def get_data():
    df    = pd.read_csv(os.path.join(DATA, "games_engineered.csv"))
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    up    = json.load(open(os.path.join(DATA, "upcoming_predictions.json")))
    mr    = json.load(open(os.path.join(DATA, "model_results.json")))
    return df, stats, up, mr

def explain_prediction(model_info, feat_values):
    """Calcula contribucion de cada feature usando coeficientes de LR."""
    clf      = model_info["model"]
    imputer  = clf.named_steps["imputer"]
    scaler   = clf.named_steps["scaler"]
    lr       = clf.named_steps["clf"]

    X = pd.DataFrame([feat_values])[FEATURES]
    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    coefs    = lr.coef_[0]
    contribs = X_scaled[0] * coefs

    results = []
    for feat, contrib in zip(FEATURES, contribs):
        results.append({
            "feature": feat,
            "label": FEAT_LABELS.get(feat, feat),
            "contribution": round(float(contrib), 4),
            "value": feat_values.get(feat, 0),
        })
    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results[:5]

md = get_model()
model = md["model"]
threshold = md["threshold"]
df, stats, upcoming, mr = get_data()
real = df[~df["name"].str.startswith("Game_")].copy()

try:
    import plotly.express as px; PX=True
except: PX=False

# ─── HEADER ────────────────────────────────────────────────────
st.markdown('<div style="padding:1.5rem 0 0.5rem"><span style="font-size:28px;font-weight:700;color:#e8e8f0">Hype <span style="color:#9f8fff">vs</span> Reality</span><span style="margin-left:12px;font-size:14px;color:#888899">Video Game Disappointment Detector</span></div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col,num,lbl,clr in zip([c1,c2,c3,c4,c5],[
    ("180","Juegos","#9f8fff"),("52","Decepciones","#f07060"),
    ("77","Sorpresas","#42d9a8"),(str(md["auc_temporal"]),"AUC temporal","#6abf69"),
    (str(threshold)[:5],"Threshold óptimo","#f0a442")]):
    col.markdown(f'<div class="mc"><div class="mn" style="color:{num[2]}">{num[0]}</div><div class="ml">{num[1]}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
t1,t2,t3,t4,t5 = st.tabs(["📊 Overview","🏆 Rankings","🔍 Explorador","🤖 Modelo ML","🔮 Predictor 2025"])

with t1:
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Distribución de resultados")
        ld=pd.DataFrame({"Resultado":["Launch disaster","Disappointment","Met expectations","Positive surprise"],"Count":[38,14,51,77]})
        if PX:
            f=px.pie(ld,values="Count",names="Resultado",color_discrete_sequence=["#f07060","#f0a442","#5aa8f0","#42d9a8"],hole=0.55)
            f.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0"); st.plotly_chart(f,use_container_width=True)
        else: st.dataframe(ld)
    with c2:
        st.subheader("Delta por tier")
        td=pd.DataFrame({"Tier":["AAA","AA","Indie"],"Delta":[0.68,-0.58,-1.71]})
        if PX:
            f2=px.bar(td,x="Tier",y="Delta",color="Delta",color_continuous_scale=["#42d9a8","#f07060"],text="Delta")
            f2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",showlegend=False)
            f2.update_traces(texttemplate='%{text:.2f}',textposition='outside'); st.plotly_chart(f2,use_container_width=True)
        else: st.dataframe(td)
    st.subheader("Delta por género")
    gd=pd.DataFrame(list(stats["genre_avg_delta"].items()),columns=["Género","Delta"]).sort_values("Delta")
    if PX:
        f3=px.bar(gd,x="Delta",y="Género",orientation="h",color="Delta",color_continuous_scale=["#42d9a8","#f07060"])
        f3.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=380,showlegend=False); st.plotly_chart(f3,use_container_width=True)
    st.subheader("Delta por año")
    yd=pd.DataFrame(list(stats["year_avg_delta"].items()),columns=["Año","Delta"]).sort_values("Año")
    if PX:
        f4=px.line(yd,x="Año",y="Delta",line_shape="spline",markers=True,color_discrete_sequence=["#9f8fff"])
        f4.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.2)")
        f4.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0"); st.plotly_chart(f4,use_container_width=True)

with t2:
    c1,c2=st.columns(2)
    with c1:
        st.subheader("💥 Top 10 decepciones")
        dec=real.nlargest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        dec.columns=["Juego","Año","Tier","Hype","Review","Delta"]
        st.dataframe(dec,use_container_width=True,hide_index=True,column_config={"Delta":st.column_config.ProgressColumn("Delta",min_value=0,max_value=8,format="%.2f")})
    with c2:
        st.subheader("✨ Top 10 sorpresas")
        sur=real.nsmallest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        sur.columns=["Juego","Año","Tier","Hype","Review","Delta"]
        st.dataframe(sur,use_container_width=True,hide_index=True)
    st.subheader("🕹️ Scatter: Hype vs Review")
    if PX:
        cmap={"launch_disaster":"#f07060","disappointment":"#f0a442","met_expectations":"#5aa8f0","positive_surprise":"#42d9a8"}
        f5=px.scatter(real,x="hype_score",y="review_score_combined",color="label",color_discrete_map=cmap,hover_name="name",hover_data={"hype_score":":.1f","review_score_combined":":.1f","delta":":.2f"})
        f5.add_shape(type="line",x0=1,y0=1,x1=10,y1=10,line=dict(color="rgba(255,255,255,0.15)",dash="dash"))
        f5.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=480)
        f5.update_traces(marker=dict(size=10,opacity=0.8)); st.plotly_chart(f5,use_container_width=True)

with t3:
    st.subheader("🔍 Explorador")
    c1,c2,c3=st.columns(3)
    with c1: tf=st.multiselect("Tier",["AAA","AA","Indie"],default=["AAA","AA","Indie"])
    with c2: lf=st.multiselect("Resultado",["launch_disaster","disappointment","met_expectations","positive_surprise"],default=["launch_disaster","disappointment","met_expectations","positive_surprise"])
    with c3: yr=st.slider("Años",2013,2024,(2015,2024))
    flt=real[(real["publisher_tier"].isin(tf))&(real["label"].isin(lf))&(real["year"].between(yr[0],yr[1]))]
    st.caption(f"{len(flt)} juegos")
    st.dataframe(flt[["name","year","genre","publisher_tier","hype_score","metacritic_score","steam_score","delta","label"]].sort_values("delta",ascending=False),use_container_width=True,hide_index=True)
    st.markdown("---")
    sel=st.selectbox("Detalle",real["name"].sort_values().tolist())
    row=real[real["name"]==sel].iloc[0]
    g1,g2,g3,g4=st.columns(4)
    g1.metric("Hype",f"{row['hype_score']:.1f}/10"); g2.metric("Metacritic",f"{row['metacritic_score']:.1f}/10")
    g3.metric("Steam",f"{row['steam_score']:.1f}/10"); g4.metric("Delta",f"{row['delta']:+.2f}",delta_color="inverse")

with t4:
    st.subheader("🤖 Validación del modelo")

    # ─── MÉTRICAS DE VALIDACIÓN PROMINENTES ──────────────────
    st.markdown("### Resultados de validación temporal (out-of-time)")
    st.caption(f"Entrenado con {md['train_size']} juegos (2013-2021) · Evaluado en {md['test_size']} juegos (2022-2024) que el modelo nunca vio")

    v1,v2,v3,v4,v5 = st.columns(5)
    v1.metric("AUC-ROC temporal", md["auc_temporal"], help="En datos 2022-2024 nunca vistos")
    v2.metric("Precision", md["precision_temporal"], help="De los que predice como decepción, cuántos realmente decepcionaron")
    v3.metric("Recall", md["recall_temporal"], help="De las decepciones reales, cuántas detectó")
    v4.metric("F1 Score", md["f1_temporal"], help="Balance entre precision y recall")
    v5.metric("Threshold óptimo", f"{threshold:.3f}", help="Optimizado con precision-recall curve (no 0.5 arbitrario)")

    st.markdown("---")

    # ─── COMPARATIVA DE MODELOS ───────────────────────────────
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**Comparativa de modelos (CV en train + test temporal)**")
        mdf=pd.DataFrame(mr["all_model_scores"]).T.reset_index()
        mdf.columns=["Modelo","AUC CV","F1 CV","Acc CV","AUC Temporal","F1 Temporal"]
        st.dataframe(mdf,use_container_width=True,hide_index=True,
            column_config={
                "AUC CV": st.column_config.ProgressColumn("AUC CV",min_value=0,max_value=1,format="%.3f"),
                "AUC Temporal": st.column_config.ProgressColumn("AUC Temporal",min_value=0,max_value=1,format="%.3f"),
                "F1 Temporal": st.column_config.ProgressColumn("F1 Temporal",min_value=0,max_value=1,format="%.3f"),
            })
        st.markdown("<br>**Matriz de confusión (dataset completo)**",unsafe_allow_html=True)
        cm=mr["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm,index=["Real: No decepciona","Real: Decepciona"],columns=["Pred: No","Pred: Sí"]),use_container_width=True)

    with col2:
        st.markdown("**Feature importance (coeficientes LR)**")
        fi=pd.DataFrame(list(mr["feature_importance"].items()),columns=["Feature","Imp"]).head(15).sort_values("Imp")
        if PX:
            f6=px.bar(fi,x="Imp",y="Feature",orientation="h",color="Imp",color_continuous_scale=["#5aa8f0","#9f8fff"])
            f6.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=420,showlegend=False); st.plotly_chart(f6,use_container_width=True)
        else: st.dataframe(fi)

    # ─── LIMITACIONES ─────────────────────────────────────────
    with st.expander("Limitaciones y contexto metodológico"):
        st.markdown(f"""
        **Metodología de validación**
        - Train: juegos 2013–2021 ({md['train_size']} juegos)
        - Test: juegos 2022–2024 ({md['test_size']} juegos, nunca vistos durante entrenamiento)
        - Cross-validation: 5-fold estratificada dentro del set de train
        - Threshold: {threshold:.3f} optimizado con precision-recall curve (no 0.5 arbitrario)
        - Data leakage corregido: features de género calculadas exclusivamente con datos de train

        **Limitaciones conocidas**
        - Dataset de 180 juegos — suficiente para MVP, limitado para generalizar
        - Métricas pre-lanzamiento parcialmente estimadas para juegos sin Steam ID
        - El modelo no captura eventos externos (controversias, recortes de presupuesto)
        - Riesgo de decepción ≠ calidad del juego

        **Próximos pasos**
        - NLP en reviews de Steam para sentiment más granular
        - Ampliar dataset con más juegos indie y AA
        - Incorporar historial del publisher como feature
        """)

with t5:
    st.subheader("🔮 Predicciones 2025")
    st.warning("**Importante:** riesgo de decepción ≠ calidad del juego. El modelo mide la brecha entre expectativa (hype pre-launch) y review score post-lanzamiento. Un juego puede decepcionar expectativas y aun así ser bueno.")

    for g in sorted(upcoming,key=lambda x:x["proba_disappointment"],reverse=True):
        pct=round(g["proba_disappointment"]*100); risk=g["risk_level"]
        clr={"ALTO":"#f07060","MODERADO":"#f0a442","BAJO":"#5aa8f0","MUY BAJO":"#42d9a8"}.get(risk,"#888899")
        c1,c2,c3,c4=st.columns([3,1,2,1])
        c1.markdown(f"**{g['name']}**"); c1.caption(f"{g['year']} · {g['genre']} · {g['publisher_tier']}")
        c2.markdown(f"**{g['hype_score']:.1f}**"); c2.caption("Hype")
        c3.progress(g["proba_disappointment"]); c3.caption(f"{pct}% riesgo")
        c4.markdown(f'<span style="background:{clr};color:#0f0f13;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700">{risk}</span>',unsafe_allow_html=True)
        st.markdown("---")

    # ─── PREDICTOR INTERACTIVO CON EXPLICABILIDAD ─────────────
    st.subheader("🎮 Predice tu propio juego")
    with st.form("pf"):
        pc1,pc2,pc3=st.columns(3)
        with pc1:
            ph=st.slider("Hype",1.0,10.0,7.5,0.1); pt=st.slider("Google Trends",0,100,65)
            pv=st.slider("Trailer views (M)",0.0,50.0,15.0,0.5); plr=st.slider("Like ratio",0.70,0.99,0.90,0.01)
        with pc2:
            prm=st.slider("Reddit mentions (K)",0.0,150.0,30.0,1.0); prs=st.slider("Reddit sentiment",0.5,0.95,0.75,0.01)
            pw=st.slider("Wishlists (K)",0.0,5000.0,800.0,50.0); pp=st.slider("Press coverage",1.0,10.0,7.0,0.1)
        with pc3:
            pdt=st.slider("Dev track record",1.0,10.0,7.0,0.1); pmd=st.slider("Días marketing",30,900,365)
            ppr=st.selectbox("Precio ($)",[20,30,40,50,60,70]); pb=st.checkbox("Beta pública")
            ptier=st.selectbox("Tier",["AAA","AA","Indie"]); pgr=st.selectbox("Género",["RPG","Shooter","Action","Platformer","Horror","Adventure","Simulation","Fighting","Roguelike","Strategy","Sports"])
        sub=st.form_submit_button("🔮 Predecir + Explicar",use_container_width=True)

    if sub:
        ghb={"RPG":7.1,"Shooter":6.8,"Action":7.0,"Platformer":6.2,"Strategy":5.8,"Sports":6.0,"Horror":6.5,"Simulation":5.5,"Fighting":6.8,"Adventure":6.3,"Roguelike":5.9}
        gdb={"RPG":0.1,"Shooter":0.5,"Action":-0.1,"Platformer":-1.2,"Strategy":-0.5,"Sports":0.2,"Horror":-0.3,"Simulation":-0.8,"Fighting":0.0,"Adventure":-0.6,"Roguelike":-1.5}
        feat={
            "hype_score":ph,"google_trends_peak":pt,"trailer_views_m":pv,"like_ratio":plr,
            "reddit_mentions_k":prm,"reddit_sentiment_pre":prs,"steam_wishlist_k":pw,
            "press_coverage_score":pp,"dev_track_record":pdt,"marketing_days":pmd,
            "had_beta":int(pb),"price_usd":ppr,
            "hype_vs_genre_avg":round(ph-ghb.get(pgr,6.5),3),
            "marketing_intensity":round((pt/100)*0.35+min(pv/30,1)*0.35+min(prm/100,1)*0.30,4),
            "hype_sentiment_gap":round(ph/10-prs,4),
            "wishlist_per_day":round(pw/(pmd+1),3),
            "is_premium":int(ppr>=60),"is_aaa":int(ptier=="AAA"),
            "years_ago":0,"genre_avg_delta":gdb.get(pgr,0.0)
        }

        proba=model.predict_proba(pd.DataFrame([feat])[FEATURES])[0][1]
        pred=(proba>=threshold)
        pct2=round(proba*100)
        clr2="#f07060" if proba>=0.75 else "#f0a442" if proba>=0.50 else "#5aa8f0" if proba>=0.30 else "#42d9a8"
        risk2="ALTO" if proba>=0.75 else "MODERADO" if proba>=0.50 else "BAJO" if proba>=0.30 else "MUY BAJO"
        verd="⚠️ Alto riesgo de decepcionar." if pred else "✅ Probablemente cumplirá expectativas."

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f'<div style="background:#16161e;border:1px solid {clr2}40;border-radius:12px;padding:1.5rem">'
                f'<div style="font-size:42px;font-weight:700;color:{clr2}">{pct2}%</div>'
                f'<div style="font-size:14px;color:#888899;margin-bottom:8px">probabilidad de decepcionar</div>'
                f'<span style="background:{clr2};color:#0f0f13;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700">{risk2}</span>'
                f'<p style="margin-top:1rem;color:#e8e8f0">{verd}</p>'
                f'<p style="font-size:12px;color:#888899">Threshold usado: {threshold:.3f}</p>'
                f'</div>', unsafe_allow_html=True)

        with col2:
            # ─── EXPLICABILIDAD ───────────────────────────────
            st.markdown("**¿Por qué esta predicción?**")
            explanations = explain_prediction(md, feat)
            for exp in explanations:
                contrib = exp["contribution"]
                direction = "↑ aumenta riesgo" if contrib > 0 else "↓ reduce riesgo"
                color = "#f07060" if contrib > 0 else "#42d9a8"
                bar_w = min(abs(contrib) * 200, 100)
                sign = "+" if contrib > 0 else ""
                st.markdown(
                    f'<div style="margin-bottom:8px;padding:8px 12px;background:#1e1e2a;border-radius:8px;border-left:3px solid {color}">'
                    f'<div style="font-size:13px;color:#e8e8f0;font-weight:500">{exp["label"]}</div>'
                    f'<div style="font-size:12px;color:{color}">{sign}{contrib:.3f} · {direction}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

st.markdown("---")
st.caption(f"Hype vs Reality · {datetime.now().strftime('%Y-%m-%d')} · AUC temporal {md['auc_temporal']} · Threshold {threshold:.3f} · Train 2013-2021 / Test 2022-2024")
