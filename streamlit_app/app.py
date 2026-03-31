"""
Hype vs Reality — Streamlit App
================================
Deploy: streamlit run streamlit_app/app.py
"""
import streamlit as st, pandas as pd, numpy as np, json, os, pickle
from datetime import datetime

st.set_page_config(page_title="Hype vs Reality", page_icon="🎮", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0f0f13}
[data-testid="stHeader"]{background:transparent}
.metric-card{background:#16161e;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1rem 1.25rem;text-align:center}
.metric-num{font-size:32px;font-weight:700}
.metric-label{font-size:12px;color:#888899;text-transform:uppercase;letter-spacing:0.5px}
</style>""", unsafe_allow_html=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(THIS_DIR)

def find_dir(rel_paths, sentinel):
    for p in rel_paths:
        full = os.path.join(THIS_DIR, p) if not os.path.isabs(p) else p
        if os.path.exists(os.path.join(full, sentinel)):
            return full
    return os.path.join(THIS_DIR, rel_paths[0])

DATA_DIR   = find_dir(["data", "../data"], "games_engineered.csv")
MODEL_PATH = next((p for p in [
    os.path.join(THIS_DIR, "models", "hype_model.pkl"),
    os.path.join(ROOT, "models", "hype_model.pkl"),
] if os.path.exists(p)), "")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "games_engineered.csv"))
    stats        = json.load(open(os.path.join(DATA_DIR, "stats.json")))
    upcoming     = json.load(open(os.path.join(DATA_DIR, "upcoming_predictions.json")))
    model_results= json.load(open(os.path.join(DATA_DIR, "model_results.json")))
    return df, stats, upcoming, model_results

@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))

df, stats, upcoming, mr = load_data()
md    = load_model()
model = md["model"]
FEAT  = md["features"]
real  = df[~df["name"].str.startswith("Game_")].copy()

try:
    import plotly.express as px
    PX = True
except ImportError:
    PX = False

# Header
st.markdown('<div style="padding:1.5rem 0 0.5rem"><span style="font-size:28px;font-weight:700;color:#e8e8f0">Hype <span style="color:#9f8fff">vs</span> Reality</span><span style="margin-left:12px;font-size:14px;color:#888899">Video Game Disappointment Detector</span></div>', unsafe_allow_html=True)

# Metrics
c1,c2,c3,c4,c5 = st.columns(5)
for col, num, lbl, clr in [(c1,"180","Juegos","#9f8fff"),(c2,"52","Decepciones","#f07060"),(c3,"77","Sorpresas","#42d9a8"),(c4,str(mr["train_auc"]),"AUC-ROC","#6abf69"),(c5,"10","Predicciones 2025","#f0a442")]:
    col.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{clr}">{num}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Overview","🏆 Rankings","🔍 Explorador","🤖 Modelo ML","🔮 Predictor 2025"])

with tab1:
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de resultados")
        ld = pd.DataFrame({"Resultado":["Launch disaster","Disappointment","Met expectations","Positive surprise"],"Count":[38,14,51,77]})
        if PX:
            fig=px.pie(ld,values="Count",names="Resultado",color_discrete_sequence=["#f07060","#f0a442","#5aa8f0","#42d9a8"],hole=0.55)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0")
            st.plotly_chart(fig,use_container_width=True)
        else: st.dataframe(ld)
    with col2:
        st.subheader("Delta por tier")
        td = pd.DataFrame({"Tier":["AAA","AA","Indie"],"Delta":[0.68,-0.58,-1.71]})
        if PX:
            fig2=px.bar(td,x="Tier",y="Delta",color="Delta",color_continuous_scale=["#42d9a8","#f07060"],text="Delta")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",showlegend=False)
            fig2.update_traces(texttemplate='%{text:.2f}',textposition='outside')
            st.plotly_chart(fig2,use_container_width=True)
        else: st.dataframe(td)
    st.subheader("Delta por género")
    gd = pd.DataFrame(list(stats["genre_avg_delta"].items()),columns=["Género","Delta"]).sort_values("Delta")
    if PX:
        fig3=px.bar(gd,x="Delta",y="Género",orientation="h",color="Delta",color_continuous_scale=["#42d9a8","#f07060"])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=380,showlegend=False)
        st.plotly_chart(fig3,use_container_width=True)
    st.subheader("Delta por año")
    yd = pd.DataFrame(list(stats["year_avg_delta"].items()),columns=["Año","Delta"]).sort_values("Año")
    if PX:
        fig4=px.line(yd,x="Año",y="Delta",line_shape="spline",markers=True,color_discrete_sequence=["#9f8fff"])
        fig4.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.2)")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0")
        st.plotly_chart(fig4,use_container_width=True)

with tab2:
    col1,col2=st.columns(2)
    with col1:
        st.subheader("💥 Top 10 decepciones")
        dec=real.nlargest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        dec.columns=["Juego","Año","Tier","Hype","Review","Delta"]
        st.dataframe(dec,use_container_width=True,hide_index=True,column_config={"Delta":st.column_config.ProgressColumn("Delta",min_value=0,max_value=8,format="%.2f")})
    with col2:
        st.subheader("✨ Top 10 sorpresas")
        sur=real.nsmallest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        sur.columns=["Juego","Año","Tier","Hype","Review","Delta"]
        st.dataframe(sur,use_container_width=True,hide_index=True)
    st.subheader("🕹️ Scatter: Hype vs Review")
    if PX:
        cmap={"launch_disaster":"#f07060","disappointment":"#f0a442","met_expectations":"#5aa8f0","positive_surprise":"#42d9a8"}
        fig5=px.scatter(real,x="hype_score",y="review_score_combined",color="label",color_discrete_map=cmap,hover_name="name",hover_data={"hype_score":":.1f","review_score_combined":":.1f","delta":":.2f"})
        fig5.add_shape(type="line",x0=1,y0=1,x1=10,y1=10,line=dict(color="rgba(255,255,255,0.15)",dash="dash"))
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=480)
        fig5.update_traces(marker=dict(size=10,opacity=0.8))
        st.plotly_chart(fig5,use_container_width=True)

with tab3:
    st.subheader("🔍 Explorador")
    col1,col2,col3=st.columns(3)
    with col1: tf=st.multiselect("Tier",["AAA","AA","Indie"],default=["AAA","AA","Indie"])
    with col2: lf=st.multiselect("Resultado",["launch_disaster","disappointment","met_expectations","positive_surprise"],default=["launch_disaster","disappointment","met_expectations","positive_surprise"])
    with col3: yr=st.slider("Años",2013,2024,(2015,2024))
    flt=real[(real["publisher_tier"].isin(tf))&(real["label"].isin(lf))&(real["year"].between(yr[0],yr[1]))]
    st.caption(f"{len(flt)} juegos")
    st.dataframe(flt[["name","year","genre","publisher_tier","hype_score","metacritic_score","steam_score","delta","label"]].sort_values("delta",ascending=False),use_container_width=True,hide_index=True,column_config={"delta":st.column_config.NumberColumn("Delta",format="%.2f")})
    st.markdown("---")
    sel=st.selectbox("Detalle de juego",real["name"].sort_values().tolist())
    row=real[real["name"]==sel].iloc[0]
    g1,g2,g3,g4=st.columns(4)
    g1.metric("Hype",f"{row['hype_score']:.1f}/10"); g2.metric("Metacritic",f"{row['metacritic_score']:.1f}/10")
    g3.metric("Steam",f"{row['steam_score']:.1f}/10"); g4.metric("Delta",f"{row['delta']:+.2f}",delta_color="inverse")
    col1,col2=st.columns(2)
    col1.markdown(f"**Género:** {row['genre']}  \n**Tier:** {row['publisher_tier']}  \n**Año:** {int(row['year'])}  \n**Label:** `{row['label']}`")
    col2.markdown(f"**Google Trends:** {row['google_trends_peak']}/100  \n**Trailer Views:** {row['trailer_views_m']:.1f}M  \n**Reddit Sentiment:** {row['reddit_sentiment_pre']:.3f}  \n**Wishlists:** {row['steam_wishlist_k']:.0f}K")

with tab4:
    st.subheader("🤖 Modelo ML")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**Comparativa modelos (5-fold CV)**")
        mdf=pd.DataFrame(mr["all_model_scores"]).T.reset_index(); mdf.columns=["Modelo","AUC-ROC","F1","Accuracy"]
        st.dataframe(mdf,use_container_width=True,hide_index=True,column_config={"AUC-ROC":st.column_config.ProgressColumn("AUC-ROC",min_value=0,max_value=1,format="%.3f"),"F1":st.column_config.ProgressColumn("F1",min_value=0,max_value=1,format="%.3f"),"Accuracy":st.column_config.ProgressColumn("Accuracy",min_value=0,max_value=1,format="%.3f")})
        st.markdown("<br>**Matriz de confusión**",unsafe_allow_html=True)
        cm=mr["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm,index=["Real: No","Real: Sí"],columns=["Pred: No","Pred: Sí"]),use_container_width=True)
    with col2:
        st.markdown("**Feature importance**")
        fi=pd.DataFrame(list(mr["feature_importance"].items()),columns=["Feature","Imp"]).head(15).sort_values("Imp")
        if PX:
            fig6=px.bar(fi,x="Imp",y="Feature",orientation="h",color="Imp",color_continuous_scale=["#5aa8f0","#9f8fff"])
            fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=420,showlegend=False)
            st.plotly_chart(fig6,use_container_width=True)
        else: st.dataframe(fi)

with tab5:
    st.subheader("🔮 Predicciones 2025")
    st.info("Probabilidad de decepción estimada por el modelo según métricas pre-lanzamiento.")
    for g in sorted(upcoming,key=lambda x:x["proba_disappointment"],reverse=True):
        pct=round(g["proba_disappointment"]*100); risk=g["risk_level"]
        clr={"ALTO":"#f07060","MODERADO":"#f0a442","BAJO":"#5aa8f0","MUY BAJO":"#42d9a8"}.get(risk,"#888899")
        c1,c2,c3,c4=st.columns([3,1,2,1])
        c1.markdown(f"**{g['name']}**"); c1.caption(f"{g['year']} · {g['genre']} · {g['publisher_tier']}")
        c2.markdown(f"**{g['hype_score']:.1f}**"); c2.caption("Hype")
        c3.progress(g["proba_disappointment"]); c3.caption(f"{pct}% riesgo")
        c4.markdown(f'<span style="background:{clr};color:#0f0f13;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700">{risk}</span>',unsafe_allow_html=True)
        st.markdown("---")

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
        sub=st.form_submit_button("🔮 Predecir",use_container_width=True)

    if sub:
        ghb={"RPG":7.1,"Shooter":6.8,"Action":7.0,"Platformer":6.2,"Strategy":5.8,"Sports":6.0,"Horror":6.5,"Simulation":5.5,"Fighting":6.8,"Adventure":6.3,"Roguelike":5.9}
        gdb={"RPG":0.1,"Shooter":0.5,"Action":-0.1,"Platformer":-1.2,"Strategy":-0.5,"Sports":0.2,"Horror":-0.3,"Simulation":-0.8,"Fighting":0.0,"Adventure":-0.6,"Roguelike":-1.5}
        feat={"hype_score":ph,"google_trends_peak":pt,"trailer_views_m":pv,"like_ratio":plr,"reddit_mentions_k":prm,"reddit_sentiment_pre":prs,"steam_wishlist_k":pw,"press_coverage_score":pp,"dev_track_record":pdt,"marketing_days":pmd,"had_beta":int(pb),"price_usd":ppr,"hype_vs_genre_avg":round(ph-ghb.get(pgr,6.5),3),"marketing_intensity":round((pt/100)*0.35+min(pv/30,1)*0.35+min(prm/100,1)*0.30,4),"hype_sentiment_gap":round(ph/10-prs,4),"wishlist_per_day":round(pw/(pmd+1),3),"is_premium":int(ppr>=60),"is_aaa":int(ptier=="AAA"),"years_ago":0,"genre_avg_delta":gdb.get(pgr,0.0)}
        proba=model.predict_proba(pd.DataFrame([feat])[FEAT])[0][1]; pred=model.predict(pd.DataFrame([feat])[FEAT])[0]
        pct2=round(proba*100); clr2="#f07060" if proba>=0.75 else "#f0a442" if proba>=0.50 else "#5aa8f0" if proba>=0.30 else "#42d9a8"
        risk2="ALTO" if proba>=0.75 else "MODERADO" if proba>=0.50 else "BAJO" if proba>=0.30 else "MUY BAJO"
        verd="⚠️ Alto riesgo de decepcionar." if pred==1 else "✅ Probablemente cumplirá (o superará) expectativas."
        st.markdown(f'<div style="background:#16161e;border:1px solid {clr2}40;border-radius:12px;padding:1.5rem;margin-top:1rem"><div style="font-size:42px;font-weight:700;color:{clr2}">{pct2}%</div><div style="font-size:14px;color:#888899;margin-bottom:8px">probabilidad de decepcionar</div><span style="background:{clr2};color:#0f0f13;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700">{risk2}</span><p style="margin-top:1rem;color:#e8e8f0">{verd}</p></div>',unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Hype vs Reality · {datetime.now().strftime('%Y-%m-%d')} · {md['model_name']} AUC {md['train_auc']:.3f}")
