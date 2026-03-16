import os
import pickle

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================
# CONFIG
# =============================
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_BASE = "https://api.themoviedb.org/3"

# API key: prefer Streamlit secrets (cloud), fall back to env var (local dev)
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    from dotenv import load_dotenv

    load_dotenv()
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

if not TMDB_API_KEY:
    st.error("⚠️ TMDB_API_KEY not found. Set it in Streamlit secrets or .env file.")
    st.stop()

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES (minimal modern)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# LOAD PICKLE DATA (cached — runs once)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Loading movie database…")
def load_data():
    with open(os.path.join(BASE_DIR, "df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(BASE_DIR, "indices.pkl"), "rb") as f:
        indices = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Build a normalized title → index mapping
    title_to_idx = {}
    for k, v in indices.items():
        title_to_idx[k.strip().lower()] = int(v)

    return df, tfidf_matrix, title_to_idx


df, tfidf_matrix, TITLE_TO_IDX = load_data()


# =============================
# TMDB API HELPERS (direct calls, no FastAPI)
# =============================
def tmdb_get(path: str, params: dict | None = None) -> dict | None:
    """Call TMDB API directly. Returns parsed JSON or None."""
    q = dict(params or {})
    q["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=q, timeout=20)
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def make_img_url(path: str | None) -> str | None:
    return f"{TMDB_IMG}{path}" if path else None


def tmdb_cards_from_results(results: list, limit: int = 20) -> list[dict]:
    out = []
    for r in (results or [])[:limit]:
        tmdb_id = r.get("id")
        title = r.get("title") or r.get("name") or "Unknown"
        poster = make_img_url(r.get("poster_path"))
        if tmdb_id:
            out.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster,
                    "release_date": r.get("release_date", ""),
                    "vote_average": r.get("vote_average"),
                }
            )
    return out


def tmdb_movie_details(tmdb_id: int) -> dict | None:
    data = tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    if not data:
        return None
    return {
        "tmdb_id": int(data["id"]),
        "title": data.get("title"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "poster_url": make_img_url(data.get("poster_path")),
        "backdrop_url": make_img_url(data.get("backdrop_path")),
        "genres": data.get("genres", []),
    }


def tmdb_search_movies(query: str, page: int = 1) -> dict | None:
    return tmdb_get(
        "/search/movie",
        {"query": query, "page": page, "include_adult": "false"},
    )


def tmdb_search_first(query: str) -> dict | None:
    data = tmdb_search_movies(query)
    if data:
        results = data.get("results", [])
        if results:
            return results[0]
    return None


def tmdb_home_feed(category: str = "popular", limit: int = 20) -> list[dict]:
    if category == "trending":
        data = tmdb_get("/trending/movie/day", {})
    else:
        data = tmdb_get(f"/movie/{category}", {})
    if not data:
        return []
    return tmdb_cards_from_results(data.get("results"), limit)


# =============================
# TF-IDF RECOMMENDATION (inline)
# =============================
def tfidf_recommend(query_title: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Return list of (title, score) for similar movies."""
    key = query_title.strip().lower()
    if key not in TITLE_TO_IDX:
        return []
    idx = TITLE_TO_IDX[key]
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out = []
    for i in order:
        if i == idx:
            continue
        out.append((df.iloc[i]["title"], float(scores[i])))
        if len(out) >= top_n:
            break
    return out


def tfidf_recs_with_posters(query_title: str, top_n: int = 12) -> list[dict]:
    """Get TF-IDF recs and attach TMDB poster info."""
    recs = tfidf_recommend(query_title, top_n)
    cards = []
    for title, score in recs:
        first = tmdb_search_first(title)
        if first:
            cards.append(
                {
                    "tmdb_id": int(first["id"]),
                    "title": first.get("title") or title,
                    "poster_url": make_img_url(first.get("poster_path")),
                }
            )
    return cards


# =============================
# STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# UI HELPERS
# =============================
def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return
    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1
            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")
            with colset[c]:
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.write("🖼️ No poster")
                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)
                st.markdown(
                    f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True
                )


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed (only home)")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )
    st.divider()

    # SEARCH MODE
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data = tmdb_search_movies(typed.strip())

            if data is None:
                st.error("Search failed. Please try again.")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)
                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # HOME FEED MODE
    st.markdown(f"### 🏠 Home — {home_category.replace('_', ' ').title()}")
    home_cards = tmdb_home_feed(home_category, limit=24)
    if not home_cards:
        st.error("Home feed failed. Check your API key or try again.")
        st.stop()
    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # Top bar
    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    # Details
    data = tmdb_movie_details(tmdb_id)
    if not data:
        st.error("Could not load details. Please try again.")
        st.stop()

    # Layout: Poster LEFT, Details RIGHT
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_column_width=True)
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title', '')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_column_width=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    # TF-IDF recommendations (computed locally)
    title = (data.get("title") or "").strip()
    if title:
        with st.spinner("Computing recommendations…"):
            tfidf_cards = tfidf_recs_with_posters(title, top_n=12)

        if tfidf_cards:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(tfidf_cards, cols=grid_cols, key_prefix="details_tfidf")
        else:
            st.info("No TF-IDF recommendations available for this movie.")
    else:
        st.warning("No title available to compute recommendations.")