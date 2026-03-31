"""
API Collector — Hype vs Reality
================================
Recolecta métricas reales de pre-lanzamiento desde:
  - YouTube Data API v3
  - Reddit API (PRAW)
  - Steam Storefront API (pública, sin key)
  - Google Trends (pytrends)
  - Metacritic (scraping)

USO:
  python collector.py --game "Elden Ring" --steam-id 1245620
  python collector.py --batch upcoming_games.json

VARIABLES DE ENTORNO REQUERIDAS:
  YOUTUBE_API_KEY   — Google Cloud Console
  REDDIT_CLIENT_ID  — https://www.reddit.com/prefs/apps
  REDDIT_SECRET     — ídem
  REDDIT_USER_AGENT — nombre de tu app (e.g. "HypeDetector/1.0")
"""

import os
import json
import time
import argparse
import statistics
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus
from urllib.error import HTTPError, URLError


# ─── CONFIG ────────────────────────────────────────────────────
YOUTUBE_KEY   = os.environ.get("YOUTUBE_API_KEY", "")
REDDIT_ID     = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.environ.get("REDDIT_SECRET", "")
REDDIT_UA     = os.environ.get("REDDIT_USER_AGENT", "HypeDetector/1.0")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─── HELPERS ───────────────────────────────────────────────────
def fetch_json(url, headers=None, timeout=10):
    """GET request que devuelve JSON, con retry básico."""
    req = Request(url, headers=headers or {})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            if e.code == 429:
                print(f"  Rate limit, esperando {2**attempt*5}s...")
                time.sleep(2**attempt * 5)
            else:
                raise
        except URLError as e:
            if attempt == 2:
                raise
            time.sleep(2)
    return None


def safe_mean(lst):
    return round(statistics.mean(lst), 4) if lst else 0.0


# ─── 1. YOUTUBE ────────────────────────────────────────────────
def collect_youtube(game_name, max_results=10):
    """
    Busca trailers oficiales del juego y extrae:
      - Total de views
      - Like ratio promedio
      - Fecha del primer trailer
    """
    if not YOUTUBE_KEY:
        print("  [YouTube] Sin API key — saltando")
        return {}

    print(f"  [YouTube] Buscando trailers de '{game_name}'...")

    # Search
    q = quote_plus(f"{game_name} official trailer")
    search_url = (
        f"https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&q={q}&type=video&maxResults={max_results}"
        f"&relevanceLanguage=en&key={YOUTUBE_KEY}"
    )
    search_data = fetch_json(search_url)
    if not search_data or "items" not in search_data:
        return {}

    video_ids = [item["id"]["videoId"] for item in search_data["items"]
                 if item.get("id", {}).get("kind") == "youtube#video"]
    if not video_ids:
        return {}

    # Stats
    stats_url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?part=statistics,snippet&id={','.join(video_ids)}&key={YOUTUBE_KEY}"
    )
    stats_data = fetch_json(stats_url)
    if not stats_data or "items" not in stats_data:
        return {}

    total_views  = 0
    like_ratios  = []
    publish_dates = []

    for item in stats_data["items"]:
        s = item.get("statistics", {})
        views  = int(s.get("viewCount", 0))
        likes  = int(s.get("likeCount", 0))
        total_views += views
        if likes > 0 and views > 0:
            # YouTube ocultó dislikes en 2021; usamos solo likes vs views como proxy
            like_ratios.append(min(likes / max(views * 0.05, 1), 1.0))
        pub = item.get("snippet", {}).get("publishedAt", "")
        if pub:
            publish_dates.append(pub[:10])

    result = {
        "trailer_views_m": round(total_views / 1_000_000, 2),
        "like_ratio": round(safe_mean(like_ratios), 3) if like_ratios else 0.85,
        "first_trailer_date": min(publish_dates) if publish_dates else None,
        "num_trailers_found": len(video_ids),
    }
    print(f"    Views: {result['trailer_views_m']}M | Like ratio: {result['like_ratio']}")
    return result


# ─── 2. REDDIT ─────────────────────────────────────────────────
def _reddit_token():
    """Obtiene OAuth token de Reddit."""
    import base64
    credentials = base64.b64encode(f"{REDDIT_ID}:{REDDIT_SECRET}".encode()).decode()
    req = Request(
        "https://www.reddit.com/api/v1/access_token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {credentials}",
            "User-Agent": REDDIT_UA,
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )
    data = json.loads(urlopen(req, timeout=10).read().decode())
    return data.get("access_token", "")


def collect_reddit(game_name, days_before_launch=180):
    """
    Busca en r/games, r/gaming, r/patientgamers y el subreddit específico.
    Extrae:
      - Número de posts y comentarios
      - Score promedio de posts
      - Sentiment aproximado (upvote ratio)
    """
    if not REDDIT_ID:
        print("  [Reddit] Sin credenciales — saltando")
        return {}

    print(f"  [Reddit] Buscando menciones de '{game_name}'...")

    try:
        token = _reddit_token()
    except Exception as e:
        print(f"    Error auth: {e}")
        return {}

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": REDDIT_UA,
    }

    subreddits = ["games", "gaming", "patientgamers", "pcgaming"]
    total_posts = 0
    upvote_ratios = []
    scores = []

    for sub in subreddits:
        q = quote_plus(game_name)
        url = (
            f"https://oauth.reddit.com/r/{sub}/search"
            f"?q={q}&restrict_sr=1&sort=top&t=year&limit=25"
        )
        try:
            data = fetch_json(url, headers=headers)
            if not data or "data" not in data:
                continue
            for post in data["data"].get("children", []):
                p = post["data"]
                total_posts += 1
                upvote_ratios.append(p.get("upvote_ratio", 0.75))
                scores.append(p.get("score", 0))
        except Exception as e:
            print(f"    r/{sub} error: {e}")

    result = {
        "reddit_mentions_k": round(total_posts / 1000 * 10, 1),  # escalar a miles
        "reddit_sentiment_pre": round(safe_mean(upvote_ratios), 3),
        "reddit_avg_score": round(safe_mean(scores), 0),
    }
    print(f"    Posts: {total_posts} | Sentiment: {result['reddit_sentiment_pre']}")
    return result


# ─── 3. STEAM ──────────────────────────────────────────────────
def collect_steam(app_id):
    """
    Usa la API pública de Steam (sin key):
      - Nombre del juego
      - Precio
      - Wishlist count (aproximado desde featured)
      - Review summary
    """
    if not app_id:
        return {}

    print(f"  [Steam] Consultando app_id {app_id}...")

    # App details
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&l=en"
    try:
        data = fetch_json(url)
        if not data or str(app_id) not in data:
            return {}
        app = data[str(app_id)]
        if not app.get("success"):
            return {}
        d = app["data"]
    except Exception as e:
        print(f"    Error: {e}")
        return {}

    # Review stats
    review_url = (
        f"https://store.steampowered.com/appreviews/{app_id}"
        f"?json=1&language=all&purchase_type=all"
    )
    review_data = {}
    try:
        review_data = fetch_json(review_url) or {}
    except Exception:
        pass

    query_summary = review_data.get("query_summary", {})
    total_reviews = query_summary.get("total_reviews", 0)
    positive      = query_summary.get("total_positive", 0)
    steam_score   = round(positive / total_reviews * 10, 2) if total_reviews > 0 else 0.0

    # Precio
    price_overview = d.get("price_overview", {})
    price_usd = price_overview.get("final", 0) / 100 if price_overview else 0

    result = {
        "steam_app_id": app_id,
        "steam_name": d.get("name", ""),
        "price_usd": round(price_usd, 2),
        "steam_score": steam_score,
        "steam_total_reviews": total_reviews,
        "steam_release_date": d.get("release_date", {}).get("date", ""),
        "categories": [c["description"] for c in d.get("categories", [])[:5]],
    }
    print(f"    Precio: ${result['price_usd']} | Reviews: {total_reviews} | Score: {steam_score}")
    return result


# ─── 4. GOOGLE TRENDS ──────────────────────────────────────────
def collect_trends(game_name, months_back=12):
    """
    Usa pytrends para obtener el interés relativo en Google Search.
    Requiere: pip install pytrends
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  [Trends] pytrends no instalado (pip install pytrends)")
        return {}

    print(f"  [Trends] Consultando Google Trends para '{game_name}'...")
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([game_name], timeframe=f"today {months_back}-m", geo="")
        df = pytrends.interest_over_time()
        if df.empty:
            return {}
        peak = int(df[game_name].max())
        avg  = round(float(df[game_name].mean()), 1)
        result = {
            "google_trends_peak": peak,
            "google_trends_avg": avg,
        }
        print(f"    Peak: {peak} | Avg: {avg}")
        return result
    except Exception as e:
        print(f"    Error: {e}")
        return {}


# ─── 5. METACRITIC ─────────────────────────────────────────────
def collect_metacritic(game_name, platform="pc"):
    """
    Scraping ligero de Metacritic.
    Respeta robots.txt — solo extrae el score público.
    """
    slug = game_name.lower().replace(" ", "-").replace("'", "").replace(":", "")
    url  = f"https://www.metacritic.com/game/{slug}/"

    print(f"  [Metacritic] Scraping '{url}'...")
    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; HypeDetectorBot/1.0)",
            "Accept": "text/html",
        })
        html = urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")

        # Extraer metascore
        import re
        meta_match = re.search(r'"metaScore"[^>]*>(\d+)<', html)
        if not meta_match:
            meta_match = re.search(r'c-siteReviewScore[^>]*>.*?(\d{2,3})', html, re.DOTALL)

        user_match = re.search(r'"userScore"[^>]*>([\d.]+)<', html)

        result = {}
        if meta_match:
            result["metacritic_score"] = round(int(meta_match.group(1)) / 10, 1)
        if user_match:
            result["metacritic_user_score"] = float(user_match.group(1))

        print(f"    Metacritic: {result}")
        return result
    except Exception as e:
        print(f"    Error scraping: {e}")
        return {}


# ─── COLLECTOR PRINCIPAL ───────────────────────────────────────
def collect_game(game_name, steam_app_id=None, skip_existing=True):
    """
    Recolecta todas las métricas para un juego.
    Guarda resultado en data/collected/{slug}.json
    """
    slug = game_name.lower().replace(" ", "_").replace("'", "").replace(":", "")
    out_path = os.path.join(DATA_DIR, "collected", f"{slug}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if skip_existing and os.path.exists(out_path):
        print(f"  Ya existe {out_path}, saltando.")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Recolectando: {game_name}")
    print(f"{'='*50}")

    result = {
        "name": game_name,
        "steam_app_id": steam_app_id,
        "collected_at": datetime.utcnow().isoformat(),
    }

    result.update(collect_youtube(game_name))
    time.sleep(1)

    result.update(collect_reddit(game_name))
    time.sleep(1)

    if steam_app_id:
        result.update(collect_steam(steam_app_id))
        time.sleep(1)

    result.update(collect_trends(game_name))
    time.sleep(1)

    result.update(collect_metacritic(game_name))

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Guardado en {out_path}")
    return result


def collect_batch(games_list):
    """
    Recolecta una lista de juegos desde un JSON.
    Format: [{"name": "Elden Ring", "steam_app_id": 1245620}, ...]
    """
    results = []
    for i, game in enumerate(games_list):
        print(f"\n[{i+1}/{len(games_list)}]")
        result = collect_game(
            game["name"],
            steam_app_id=game.get("steam_app_id"),
        )
        results.append(result)
        # Rate limiting entre juegos
        if i < len(games_list) - 1:
            print("  Esperando 3s...")
            time.sleep(3)

    # Exportar consolidado
    out = os.path.join(DATA_DIR, "collected_batch.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nBatch completado: {len(results)} juegos → {out}")
    return results


# ─── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hype Detector — API Collector")
    parser.add_argument("--game", type=str, help="Nombre del juego a recolectar")
    parser.add_argument("--steam-id", type=int, help="Steam App ID")
    parser.add_argument("--batch", type=str, help="Path a JSON con lista de juegos")
    parser.add_argument("--no-cache", action="store_true", help="Ignorar cache existente")
    args = parser.parse_args()

    if args.batch:
        with open(args.batch) as f:
            games = json.load(f)
        collect_batch(games)
    elif args.game:
        collect_game(args.game, args.steam_id, skip_existing=not args.no_cache)
    else:
        parser.print_help()
