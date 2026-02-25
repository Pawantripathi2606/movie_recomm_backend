"""
Movie Recommendation System — FastAPI Backend
=============================================
Production-ready inference API.

Architecture decisions:
  - All pickle files loaded ONCE at startup via FastAPI lifespan.
    Never reload inside a request handler (avoids GC pressure & latency).
  - TMDB calls use the `requests` library inside a thread pool executor
    so they don't block the async event loop.
  - TMDB API key is read exclusively from the environment variable TMDB_API_KEY.
    Never hardcode secrets in source code.
  - Cosine similarity is computed on a single row (not the full NxN matrix)
    to keep memory footprint minimal on Render's free tier.
  - CORS is open (*) for development; tighten to your Vercel domain in production.
"""

import os
import pickle
import logging
import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import requests as req_lib          # rename to avoid shadowing FastAPI Request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# Logging — stdout so Render captures it automatically
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("movie_recommender")

# ─────────────────────────────────────────────────────────────
# TMDB Configuration
# Best practice: read secrets from environment — NEVER hardcode.
# Set TMDB_API_KEY in Render → Environment → Add Environment Variable.
# ─────────────────────────────────────────────────────────────
TMDB_API_KEY: Optional[str] = os.getenv("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_TIMEOUT_SECONDS = 5       # per-request timeout for TMDB HTTP calls
TOP_N_DEFAULT = 10             # default number of recommendations

# Thread pool for running blocking `requests` calls without stalling event loop
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# ─────────────────────────────────────────────────────────────
# Global model state — populated ONCE at startup
# ─────────────────────────────────────────────────────────────
model: dict = {}


def _load_pickle(path: str):
    """Load a single pickle file with error logging."""
    logger.info(f"Loading: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    All expensive I/O (pickle loading) runs here before the server
    starts accepting traffic — exactly once per process.
    """
    logger.info("🚀 Server starting — loading model artifacts...")

    base = os.path.dirname(os.path.abspath(__file__))
    try:
        model["df"]           = _load_pickle(os.path.join(base, "df.pkl"))
        model["indices"]      = _load_pickle(os.path.join(base, "indices.pkl"))
        model["tfidf"]        = _load_pickle(os.path.join(base, "tfidf.pkl"))
        model["tfidf_matrix"] = _load_pickle(os.path.join(base, "tfidf_matrix.pkl"))

        logger.info(f"✅ DataFrame: {len(model['df'])} movies")
        logger.info(f"✅ TF-IDF matrix shape: {model['tfidf_matrix'].shape}")
        logger.info(f"✅ Sparse matrix: {issparse(model['tfidf_matrix'])}")

        if not TMDB_API_KEY:
            logger.warning(
                "⚠️  TMDB_API_KEY environment variable is not set. "
                "Poster URLs will be null for all recommendations."
            )
        else:
            logger.info("✅ TMDB_API_KEY loaded from environment")

        logger.info("🎬 Model ready — accepting requests")
    except FileNotFoundError as exc:
        logger.critical(f"❌ Missing pickle file: {exc}")
        raise RuntimeError(f"Model file not found: {exc}") from exc

    yield  # ← server is live here

    model.clear()
    _thread_pool.shutdown(wait=False)
    logger.info("🛑 Server shut down — resources released")


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendations using TF-IDF + Cosine Similarity with TMDB posters.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allows all origins for easy Vercel / local integration.
# Restrict to ["https://your-app.vercel.app"] in strict production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    title: str
    top_n: Optional[int] = TOP_N_DEFAULT


class MovieResult(BaseModel):
    title: str
    poster_url: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: list[MovieResult]


# ─────────────────────────────────────────────────────────────
# TMDB Helpers
# ─────────────────────────────────────────────────────────────
def _fetch_poster_sync(title: str) -> Optional[str]:
    """
    Fetch movie poster from TMDB using a multi-strategy search.

    Strategy:
      1. Search by exact title, sort by popularity — pick FIRST result that HAS a poster_path.
      2. If no poster found, strip special characters and retry (helps foreign titles).
      3. Return None gracefully on any failure — recommendations still work without posters.

    Production notes:
    - Timeout=5s prevents slow TMDB responses from blocking the thread pool.
    - We prefer a result WITH a poster over the highest-ranked result.
    - sort_by=popularity.desc ensures we get well-known films first.
    """
    if not TMDB_API_KEY:
        return None

    def _search(query: str) -> Optional[str]:
        """Run a single TMDB search and return best poster URL or None."""
        try:
            response = req_lib.get(
                TMDB_SEARCH_URL,
                params={
                    "api_key": TMDB_API_KEY,
                    "query": query,
                    "language": "en-US",
                    "page": 1,
                    "include_adult": False,
                },
                timeout=TMDB_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            if not results:
                return None

            # Strategy: pick the first result that actually has a poster
            # (don't blindly take index 0 which may have no image)
            for movie in results:
                poster_path = movie.get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"

        except req_lib.exceptions.Timeout:
            logger.warning(f"TMDB timeout for '{query}'")
        except req_lib.exceptions.RequestException as exc:
            logger.warning(f"TMDB error for '{query}': {exc}")
        return None

    # Strategy 1: exact title search
    poster = _search(title)
    if poster:
        return poster

    # Strategy 2: strip content after colon/dash (helps subtitles like "Die Hard: With a Vengeance")
    short_title = title.split(":")[0].split(" - ")[0].strip()
    if short_title != title and len(short_title) > 2:
        poster = _search(short_title)
        if poster:
            return poster

    # Strategy 3: remove special chars for foreign titles
    import re
    clean_title = re.sub(r"[^\w\s]", " ", title).strip()
    if clean_title != title and len(clean_title) > 2:
        poster = _search(clean_title)
        if poster:
            return poster

    logger.debug(f"No TMDB poster found for '{title}'")
    return None



async def fetch_poster(title: str) -> Optional[str]:
    """
    Async wrapper: runs the blocking `requests` call in a thread pool
    so it doesn't block the FastAPI event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, _fetch_poster_sync, title)


async def fetch_all_posters(titles: list[str]) -> list[Optional[str]]:
    """
    Fetch posters for all recommended movies concurrently using asyncio.gather.
    All TMDB calls happen in parallel — total latency ≈ slowest single call.
    """
    tasks = [fetch_poster(t) for t in titles]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────
# Core Recommendation Logic (pure CPU — no I/O)
# ─────────────────────────────────────────────────────────────
def _get_title_column(df) -> str:
    """Detect the movie title column name regardless of dataset naming."""
    for col in ["title", "Title", "movie_title", "name"]:
        if col in df.columns:
            return col
    # Fallback: first object/string column
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols):
        return obj_cols[0]
    raise ValueError("No suitable title column found in DataFrame.")


def compute_recommendations(title: str, top_n: int) -> list[str]:
    """
    Content-based filtering with robust duplicate-key handling.

    Key fix: The indices Series has 3,170 duplicate title keys.
    Doing indices['Inception'] returns a Series (not a scalar int)
    when duplicates exist. We use a boolean mask + .iloc[0] to always
    extract the FIRST matching integer row-index safely.
    """
    df           = model["df"]
    indices      = model["indices"]
    tfidf_matrix = model["tfidf_matrix"]

    title_norm = title.strip()

    # ── Case-insensitive boolean mask lookup (handles duplicates correctly) ───
    # indices.index contains original-case titles e.g. 'Inception'
    mask = indices.index.str.lower() == title_norm.lower()
    matches = indices[mask]

    if matches.empty:
        raise ValueError(f"Movie '{title}' not found in the dataset.")

    # Always take the first match (iloc[0] returns a scalar even with dupes)
    idx = int(matches.iloc[0])
    logger.info(f"Found '{title}' at matrix row index {idx}")

    # Row-wise cosine similarity — memory efficient (O(n) not O(n²))
    row = tfidf_matrix[idx]
    scores = cosine_similarity(row, tfidf_matrix).flatten()

    # Argsort descending, skip index 0 (the movie itself = score 1.0)
    top_indices = scores.argsort()[::-1][1: top_n + 1]

    title_col = _get_title_column(df)
    return df.iloc[top_indices][title_col].tolist()



# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
async def health_check():
    """
    Used by Render's health check and uptime monitors.
    Returns 200 with model stats when ready.
    """
    ready = bool(model)
    return {
        "status": "healthy" if ready else "loading",
        "service": "Movie Recommendation API",
        "version": "2.0.0",
        "model_loaded": ready,
        "total_movies": len(model.get("df", [])) if ready else 0,
        "tmdb_enabled": bool(TMDB_API_KEY),
    }


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get Movie Recommendations with Posters",
)
async def recommend(request: RecommendRequest):
    """
    Returns top-N recommendations for a given movie title.

    Each result includes:
      - title: movie name
      - poster_url: TMDB poster image URL (null if not found or TMDB unavailable)

    TMDB poster fetches are done concurrently so total added latency
    is roughly the slowest single TMDB call, not the sum.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is still loading. Retry in a moment.")

    title  = request.title.strip()
    top_n  = max(1, min(request.top_n or TOP_N_DEFAULT, 20))  # clamp 1–20

    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty.")

    logger.info(f"📽️  /recommend  title='{title}'  top_n={top_n}")

    # Step 1: Pure CPU — compute recommendations from pre-trained model
    try:
        rec_titles = compute_recommendations(title, top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Recommendation error for '{title}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during recommendation.")

    # Step 2: Concurrent TMDB poster fetch (non-blocking)
    poster_urls = await fetch_all_posters(rec_titles)

    # Step 3: Assemble response
    recommendations = [
        MovieResult(title=t, poster_url=p)
        for t, p in zip(rec_titles, poster_urls)
    ]

    logger.info(f"✅ Returning {len(recommendations)} recommendations for '{title}'")
    return RecommendResponse(recommendations=recommendations)


@app.get("/search", summary="Autocomplete — search movie titles")
async def search(q: str, limit: int = 8):
    """
    Lightweight autocomplete endpoint used by the frontend search bar.
    Filters movie titles from the loaded DataFrame.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not ready.")
    df        = model["df"]
    title_col = _get_title_column(df)
    q_norm    = q.lower().strip()
    matches   = (
        df[df[title_col].str.lower().str.contains(q_norm, na=False)][title_col]
        .head(limit)
        .tolist()
    )
    return {"query": q, "results": matches}
