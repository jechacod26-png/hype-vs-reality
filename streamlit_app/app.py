import streamlit as st, pandas as pd, numpy as np, json, os
from datetime import datetime

st.set_page_config(page_title="Hype vs Reality", page_icon="🎮", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0f0f13}
[data-testid="stHeader"]{background:transparent}
.mc{background:#16161e;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1rem 1.25rem;text-align:center}
.mn{font-size:32px;font-weight:700}
.ml{font-size:12px;color:#888899;text-transform:uppercase;letter-spacing:0.5px}
</style>""", unsafe_allow_html=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)

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

@st.cache_resource
def get_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(os.path.join(DATA, "games_engineered.csv"))
    clf = Pipeline([("s", StandardScaler()),("m", LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"))])
    clf.fit(df[FEATURES], df["label_binary"])
    auc = round(roc_auc_score(df["label_binary"], clf.predict_proba(df[FEATURES])[:,1]), 3)
    return clf, auc

@st.cache_data(ttl=3600)
def get_data():
    df    = pd.read_csv(os.path.join(DATA, "games_engineered.csv"))
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    up    = json.load(open(os.path.join(DATA, "upcoming_predictions.json")))
    mr    = json.load(open(os.path.join(DATA, "model_results.json")))
    return df, stats, up, mr

model, auc = get_model()
df, stats, upcoming, mr = get_data()
real = df[~df["name"].str.startswith("Game_")].copy()

try:
    import plotly.express as px; PX=True
except: PX=False

st.markdown('<div style="padding:1.5rem 0 0.5rem"><span style="font-size:28px;font-weight:700;color:#e8e8f0">Hype <span style="color:#9f8fff">vs</span> Reality</span><span style="margin-left:12px;font-size:14px;color:#888899">Video Game Disappointment Detector</span></div>',unsafe_allow_html=True)

cols=st.columns(5)
data_metrics=[("180","Juegos","#9f8fff"),("52","Decepciones","#f07060"),("77","Sorpresas","#42d9a8"),(str(auc),"AUC-ROC","#6abf69"),("10","Pred. 2025","#f0a442")]
for col,m in zip(cols,data_metrics):
    col.markdown(f'<div class="mc"><div class="mn" style="color:{m[2]}">{m[0]}</div><div class="ml">{m[1]}</div></div>',unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
t1,t2,t3,t4,t5=st.tabs(["📊 Overview","🏆 Rankings","🔍 Explorador","🤖 Modelo ML","🔮 Predictor 2025"])

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
            f2.update_traces(texttemplate="%{text:.2f}",textposition="outside"); st.plotly_chart(f2,use_container_width=True)
        else: st.dataframe(td)
    st.subheader("Delta por genero")
    gd=pd.DataFrame(list(stats["genre_avg_delta"].items()),columns=["Genero","Delta"]).sort_values("Delta")
    if PX:
        f3=px.bar(gd,x="Delta",y="Genero",orientation="h",color="Delta",color_continuous_scale=["#42d9a8","#f07060"])
        f3.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=380,showlegend=False); st.plotly_chart(f3,use_container_width=True)
    st.subheader("Delta por anio")
    yd=pd.DataFrame(list(stats["year_avg_delta"].items()),columns=["Anio","Delta"]).sort_values("Anio")
    if PX:
        f4=px.line(yd,x="Anio",y="Delta",line_shape="spline",markers=True,color_discrete_sequence=["#9f8fff"])
        f4.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.2)")
        f4.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0"); st.plotly_chart(f4,use_container_width=True)

with t2:
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Top 10 decepciones")
        dec=real.nlargest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        dec.columns=["Juego","Anio","Tier","Hype","Review","Delta"]
        st.dataframe(dec,use_container_width=True,hide_index=True,column_config={"Delta":st.column_config.ProgressColumn("Delta",min_value=0,max_value=8,format="%.2f")})
    with c2:
        st.subheader("Top 10 sorpresas")
        sur=real.nsmallest(10,"delta")[["name","year","publisher_tier","hype_score","review_score_combined","delta"]].copy()
        sur.columns=["Juego","Anio","Tier","Hype","Review","Delta"]
        st.dataframe(sur,use_container_width=True,hide_index=True)
    st.subheader("Scatter: Hype vs Review")
    if PX:
        cm2={"launch_disaster":"#f07060","disappointment":"#f0a442","met_expectations":"#5aa8f0","positive_surprise":"#42d9a8"}
        f5=px.scatter(real,x="hype_score",y="review_score_combined",color="label",color_discrete_map=cm2,hover_name="name")
        f5.add_shape(type="line",x0=1,y0=1,x1=10,y1=10,line=dict(color="rgba(255,255,255,0.15)",dash="dash"))
        f5.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=480)
        f5.update_traces(marker=dict(size=10,opacity=0.8)); st.plotly_chart(f5,use_container_width=True)

with t3:
    st.subheader("Explorador")
    c1,c2,c3=st.columns(3)
    with c1: tf=st.multiselect("Tier",["AAA","AA","Indie"],default=["AAA","AA","Indie"])
    with c2: lf=st.multiselect("Resultado",["launch_disaster","disappointment","met_expectations","positive_surprise"],default=["launch_disaster","disappointment","met_expectations","positive_surprise"])
    with c3: yr=st.slider("Anios",2013,2024,(2015,2024))
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
    st.subheader("Modelo ML")
    c1,c2=st.columns(2)
    with c1:
        mdf=pd.DataFrame(mr["all_model_scores"]).T.reset_index(); mdf.columns=["Modelo","AUC","F1","Acc"]
        st.dataframe(mdf,use_container_width=True,hide_index=True)
        cm3=mr["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm3,index=["Real: No","Real: Si"],columns=["Pred: No","Pred: Si"]),use_container_width=True)
    with c2:
        fi=pd.DataFrame(list(mr["feature_importance"].items()),columns=["Feature","Imp"]).head(15).sort_values("Imp")
        if PX:
            f6=px.bar(fi,x="Imp",y="Feature",orientation="h",color="Imp",color_continuous_scale=["#5aa8f0","#9f8fff"])
            f6.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8e8f0",height=420,showlegend=False); st.plotly_chart(f6,use_container_width=True)

with t5:
    st.subheader("Predicciones 2025")
    for g in sorted(upcoming,key=lambda x:x["proba_disappointment"],reverse=True):
        pct=round(g["proba_disappointment"]*100); risk=g["risk_level"]
        clr={"ALTO":"#f07060","MODERADO":"#f0a442","BAJO":"#5aa8f0","MUY BAJO":"#42d9a8"}.get(risk,"#888899")
        c1,c2,c3,c4=st.columns([3,1,2,1])
        c1.markdown(f"**{g['name']}**"); c1.caption(f"{g['year']} - {g['publisher_tier']}")
        c2.markdown(f"**{g['hype_score']:.1f}**"); c2.caption("Hype")
        c3.progress(g["proba_disappointment"]); c3.caption(f"{pct}% riesgo")
        c4.markdown(f'<span style="background:{clr};color:#0f0f13;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700">{risk}</span>',unsafe_allow_html=True)
        st.markdown("---")
    st.subheader("Predice tu propio juego")
    with st.form("pf"):
        pc1,pc2,pc3=st.columns(3)
        with pc1:
            ph=st.slider("Hype",1.0,10.0,7.5,0.1); pt=st.slider("Google Trends",0,100,65)
            pv=st.slider("Trailer views M",0.0,50.0,15.0,0.5); plr=st.slider("Like ratio",0.70,0.99,0.90,0.01)
        with pc2:
            prm=st.slider("Reddit mentions K",0.0,150.0,30.0,1.0); prs=st.slider("Reddit sentiment",0.5,0.95,0.75,0.01)
            pw=st.slider("Wishlists K",0.0,5000.0,800.0,50.0); pp=st.slider("Press coverage",1.0,10.0,7.0,0.1)
        with pc3:
            pdt=st.slider("Dev track record",1.0,10.0,7.0,0.1); pmd=st.slider("Dias marketing",30,900,365)
            ppr=st.selectbox("Precio USD",[20,30,40,50,60,70]); pb=st.checkbox("Beta publica")
            ptier=st.selectbox("Tier",["AAA","AA","Indie"]); pgr=st.selectbox("Genero",["RPG","Shooter","Action","Platformer","Horror","Adventure","Simulation","Fighting","Roguelike","Strategy","Sports"])
        sub=st.form_submit_button("Predecir",use_container_width=True)
    if sub:
        ghb={"RPG":7.1,"Shooter":6.8,"Action":7.0,"Platformer":6.2,"Strategy":5.8,"Sports":6.0,"Horror":6.5,"Simulation":5.5,"Fighting":6.8,"Adventure":6.3,"Roguelike":5.9}
        gdb={"RPG":0.1,"Shooter":0.5,"Action":-0.1,"Platformer":-1.2,"Strategy":-0.5,"Sports":0.2,"Horror":-0.3,"Simulation":-0.8,"Fighting":0.0,"Adventure":-0.6,"Roguelike":-1.5}
        feat={"hype_score":ph,"google_trends_peak":pt,"trailer_views_m":pv,"like_ratio":plr,"reddit_mentions_k":prm,"reddit_sentiment_pre":prs,"steam_wishlist_k":pw,"press_coverage_score":pp,"dev_track_record":pdt,"marketing_days":pmd,"had_beta":int(pb),"price_usd":ppr,"hype_vs_genre_avg":round(ph-ghb.get(pgr,6.5),3),"marketing_intensity":round((pt/100)*0.35+min(pv/30,1)*0.35+min(prm/100,1)*0.30,4),"hype_sentiment_gap":round(ph/10-prs,4),"wishlist_per_day":round(pw/(pmd+1),3),"is_premium":int(ppr>=60),"is_aaa":int(ptier=="AAA"),"years_ago":0,"genre_avg_delta":gdb.get(pgr,0.0)}
        proba=model.predict_proba(pd.DataFrame([feat])[FEATURES])[0][1]
        pred=model.predict(pd.DataFrame([feat])[FEATURES])[0]
        pct2=round(proba*100)
        clr2="#f07060" if proba>=0.75 else "#f0a442" if proba>=0.50 else "#5aa8f0" if proba>=0.30 else "#42d9a8"
        risk2="ALTO" if proba>=0.75 else "MODERADO" if proba>=0.50 else "BAJO" if proba>=0.30 else "MUY BAJO"
        verd="Alto riesgo de decepcionar." if pred==1 else "Probablemente cumplira expectativas."
        st.markdown(f'<div style="background:#16161e;border:1px solid {clr2}40;border-radius:12px;padding:1.5rem;margin-top:1rem"><div style="font-size:42px;font-weight:700;color:{clr2}">{pct2}%</div><div style="font-size:14px;color:#888899;margin-bottom:8px">probabilidad de decepcionar</div><span style="background:{clr2};color:#0f0f13;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:700">{risk2}</span><p style="margin-top:1rem;color:#e8e8f0">{verd}</p></div>',unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Hype vs Reality - {datetime.now().strftime('%Y-%m-%d')} - Logistic Regression AUC {auc}")
