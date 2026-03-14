import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# ENV VARIABLES
# -----------------------------
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not found in environment variables")

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Movie Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# -----------------------------
# MODELS
# -----------------------------
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    release_date: Optional[str] = None
    overview: Optional[str] = None
    poster_url: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetail(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: Optional[List[Dict[str, Any]]] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetail
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

# -----------------------------
# HELPERS
# -----------------------------
def _norm_title(t: str) -> str:
    return t.strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if path:
        return f"{TMDB_IMG_BASE}{path}"
    return None


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error {repr(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB API error {r.text}")

    return r.json()


async def tmdb_cards_from_results(results: List[Dict], limit: int = 20):

    out: List[TMDBMovieCard] = []

    for r in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(r["id"]),
                title=r.get("title") or r.get("name") or "Unknown Title",
                release_date=r.get("release_date") or r.get("first_air_date"),
                poster_url=make_img_url(r.get("poster_path")),
                vote_average=r.get("vote_average"),
            )
        )

    return out


async def tmdb_movie_details(tmdb_id: int):

    data = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})

    return TMDBMovieDetail(
        tmdb_id=int(data["id"]),
        title=data.get("title"),
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []),
    )


async def tmdb_search_movies(query: str, page: int = 1):

    return await tmdb_get(
        "/search/movie",
        {"query": query, "page": page, "include_adult": False},
    )


async def tmdb_search_first(query: str):

    data = await tmdb_search_movies(query)

    results = data.get("results", [])

    if results:
        return results[0]

    return None


# -----------------------------
# TF-IDF FUNCTIONS
# -----------------------------
def build_title_to_idx_map(indices: Any):

    title_to_idx = {}

    for k, v in indices.items():
        title_to_idx[_norm_title(k)] = int(v)

    return title_to_idx


def get_local_idx_by_title(title: str):

    key = _norm_title(title)

    if key in TITLE_TO_IDX:
        return TITLE_TO_IDX[key]

    raise HTTPException(status_code=404, detail="Movie not found in dataset")


def tfidf_recommend_titles(query_title: str, top_n: int = 10):

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]

    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out = []

    for i in order:

        if i == idx:
            continue

        title_i = df.iloc[i]["title"]

        out.append((title_i, float(scores[i])))

        if len(out) >= top_n:
            break

    return out


async def attach_tmdb_card_by_title(title: str):

    try:

        m = await tmdb_search_first(title)

        if not m:
            return None

        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title"),
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )

    except Exception:
        return None


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def load_pickles():

    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = "popular", limit: int = 20):

    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {})

    else:
        data = await tmdb_get(f"/movie/{category}", {})

    return await tmdb_cards_from_results(data.get("results"), limit)


@app.get("/tmdb/search")
async def tmdb_search(query: str):

    return await tmdb_search_movies(query)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetail)
async def movie_details_route(tmdb_id: int):

    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/tfidf")
async def recommend_tfidf(title: str, top_n: int = 10):

    recs = tfidf_recommend_titles(title, top_n)

    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(query: str):

    best = await tmdb_search_first(query)

    if not best:
        raise HTTPException(status_code=404, detail="Movie not found")

    tmdb_id = best["id"]

    details = await tmdb_movie_details(tmdb_id)

    recs = tfidf_recommend_titles(details.title)

    tfidf_items = []

    for title, score in recs:

        card = await attach_tmdb_card_by_title(title)

        tfidf_items.append(
            TFIDFRecItem(title=title, score=score, tmdb=card)
        )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=[],
    )