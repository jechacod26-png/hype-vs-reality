import os, json, time
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

YOUTUBE_KEY = os.environ.get("YOUTUBE_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_FILE = os.path.join(DATA_DIR, "games_engineered.csv")

GAMES_2025 = [
    {"name": "Monster Hunter Wilds",   "steam_id": 2246340, "genre": "RPG",        "tier": "AAA",   "year": 2025},
    {"name": "Hollow Knight Silksong", "steam_id": 1030300, "genre": "Platformer", "tier": "Indie", "year": 2025},
    {"name": "Doom The Dark Ages",     "steam_id": 2835570, "genre": "Shooter",    "tier": "AAA",   "year": 2025},
    {"name": "Borderlands 4",          "steam_id": None,    "genre": "Shooter",    "tier": "AAA",   "year": 2025},
    {"name": "Ghost of Yotei",         "steam_id": None,    "genre": "Action",     "tier": "AAA",   "year": 2025},
    {"name": "GTA VI",                 "steam_id": None,    "genre": "Action",     "tier": "AAA",   "year": 2025},
    {"name": "Death Stranding 2",      "steam_id": None,    "genre": "Adventure",  "tier": "AA",    "year": 2025},
    {"name": "Fable",                  "steam_id": None,    "genre": "RPG",        "tier": "AAA",   "year": 2025},
    {"name": "Metroid Prime 4",        "steam_id": None,    "genre": "Action",     "tier": "AAA",   "year": 2025},
    {"name": "Mafia The Old Country",  "steam_id": None,    "genre": "Action",     "tier": "AA",    "year": 2025},
]

def fetch_json(url):
    try:
        req = Request(url, headers={"User-Agent": "HypeDetector/1.0"})
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"    Error: {e}")
        return None

def get_youtube_data(name):
    if not YOUTUBE_KEY:
        return {}
    print(f"  [YouTube] {name}...")
    q = quote_plus(f"{name} official trailer")
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={q}&type=video&maxResults=5&key={YOUTUBE_KEY}"
    data = fetch_json(url)
    if not data or "items" not in data:
        return {}
    ids = [i["id"]["videoId"] for i in data["items"] if i.get("id",{}).get("kind")=="youtube#video"]
    if not ids:
        return {}
    stats = fetch_json(f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={','.join(ids)}&key={YOUTUBE_KEY}")
    if not stats:
        return {}
    views, likes = 0, 0
    for item in stats.get("items", []):
        s = item.get("statistics", {})
        views += int(s.get("viewCount", 0))
        likes += int(s.get("likeCount", 0))
    views_m = round(views / 1_000_000, 2)
    ratio = round(min(likes / max(views * 0.05, 1), 0.99), 3) if views > 0 else 0.85
    print(f"    Views: {views_m}M | Ratio: {ratio}")
    return {"trailer_views_m": views_m, "like_ratio": ratio}

def get_steam_data(app_id):
    if not app_id:
        return {}
    print(f"  [Steam] {app_id}...")
    data = fetch_json(f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us")
    if not data or str(app_id) not in data:
        return {}
    d = data[str(app_id)]
    if not d.get("success"):
        return {}
    price = round(d["data"].get("price_overview", {}).get("final", 6000) / 100, 2)
    rv = fetch_json(f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all") or {}
    qs = rv.get("query_summary", {})
    total = qs.get("total_reviews", 0)
    pos   = qs.get("total_positive", 0)
    score = round(pos / total * 10, 2) if total > 0 else 0.0
    print(f"    Precio:  | Score: {score}")
    return {"price_usd": price, "steam_score": score}

def run():
    import pandas as pd
    print("="*50)
    print(f"Collector — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50)
    df = pd.read_csv(OUT_FILE)
    rows = []
    for g in GAMES_2025:
        print(f"\n[{g['name']}]")
        yt = get_youtube_data(g["name"]); time.sleep(1)
        st = get_steam_data(g.get("steam_id")); time.sleep(1)
        views = yt.get("trailer_views_m", 5.0)
        hype  = round(min(5.0 + views * 0.3, 9.9), 2) if yt else {"AAA":8.2,"AA":7.0,"Indie":6.5}.get(g["tier"],7.0)
        tier  = g["tier"]; genre = g["genre"]
        tm    = {"AAA":1.3,"AA":1.0,"Indie":0.5}.get(tier,1.0)
        rows.append({
            "name": g["name"], "year": g["year"], "genre": genre, "publisher_tier": tier,
            "hype_score": hype, "metacritic_score": 0.0,
            "steam_score": st.get("steam_score", 0.0), "review_score_combined": 0.0,
            "delta": 0.0, "label": "upcoming", "label_binary": 0,
            "google_trends_peak": round(min(hype*8.5*tm,100),1),
            "trailer_views_m": yt.get("trailer_views_m",10.0),
            "like_ratio": yt.get("like_ratio",0.85),
            "reddit_mentions_k": round(hype*4.5*tm,1),
            "reddit_sentiment_pre": round(0.65+hype*0.02,3),
            "steam_wishlist_k": round(hype*200*tm,0),
            "press_coverage_score": round(hype*0.85,2),
            "dev_track_record": {"AAA":7.2,"AA":6.8,"Indie":7.5}.get(tier,7.0),
            "marketing_days": {"AAA":540,"AA":360,"Indie":180}.get(tier,365),
            "had_beta": 0, "price_usd": st.get("price_usd",60.0),
            "hype_vs_genre_avg": round(hype-{"RPG":7.1,"Shooter":6.8,"Action":7.0,"Platformer":6.2,"Adventure":6.3}.get(genre,6.5),3),
            "marketing_intensity": round((min(hype*8.5*tm,100)/100)*0.35+min(yt.get("trailer_views_m",10)/30,1)*0.35+min(hype*4.5*tm/100,1)*0.30,4),
            "hype_sentiment_gap": round(hype/10-(0.65+hype*0.02),4),
            "wishlist_per_day": round(hype*200*tm/({"AAA":540,"AA":360,"Indie":180}.get(tier,365)+1),3),
            "is_premium": int(st.get("price_usd",60)>=60), "is_aaa": int(tier=="AAA"),
            "years_ago": 2025-g["year"],
            "genre_avg_delta": {"RPG":0.1,"Shooter":0.5,"Action":-0.1,"Platformer":-1.2,"Adventure":-0.6}.get(genre,0.0),
        })
        print(f"  Hype: {hype}")
    df_new = pd.DataFrame(rows)
    names  = set(df_new["name"].str.lower())
    df_old = df[~df["name"].str.lower().isin(names)]
    for col in df_old.columns:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[[c for c in df_old.columns if c in df_new.columns]]
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final.to_csv(OUT_FILE, index=False)
    st_path = os.path.join(BASE_DIR, "streamlit_app", "data", "games_engineered.csv")
    if os.path.exists(os.path.dirname(st_path)):
        df_final.to_csv(st_path, index=False)
    print(f"\nDataset actualizado: {len(df_final)} juegos")

if __name__ == "__main__":
    run()
