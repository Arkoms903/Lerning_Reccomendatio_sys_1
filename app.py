from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.recommender import (
    encode_topics,
    encode_interactions,
    hybrid_recommend,
    get_completed_topics,
    prerequisites_met_for_user,
    cold_start_recommend,
)
from schemas import (
    RecommendRequest,
    NewUserRecommendRequest,
    RecommendResponse,
    TopicRecommendation,
    CompletedTopicsResponse,
    HealthResponse,
)


# -----------------------------------------
# Global State (loaded once at startup)
# -----------------------------------------

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads and encodes all data once at startup."""
    print("Loading and encoding data...")

    topics_raw = pd.read_csv("data/topics.csv")
    interactions_raw = pd.read_csv("data/interactions.csv")
    prerequisites_df = pd.read_csv("data/prerequisites.csv")

    topics_df, ohe_encoder, topic_vectors, vectorizer = encode_topics(topics_raw)
    interactions_df = encode_interactions(interactions_raw)

    app_state["topics_df"] = topics_df
    app_state["interactions_df"] = interactions_df
    app_state["prerequisites_df"] = prerequisites_df
    app_state["topic_vectors"] = topic_vectors
    app_state["vectorizer"] = vectorizer

    print("✅ Data loaded successfully!")
    print(f"   Topics: {len(topics_df)} | Interactions: {len(interactions_df)} | Prerequisites: {len(prerequisites_df)}")

    yield

    app_state.clear()
    print("App shut down.")


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    lifespan=lifespan,
    title="Learning Path Recommender API",
    description="""
Hybrid ML-based learning path recommendation system.

**Two types of users supported:**
- **Existing users** → `/recommend` (POST) or `/recommend/{user_id}` (GET)
- **New users with prior knowledge** → `/recommend/new-user` (POST) — provide completed topic IDs
- **Brand new users (zero history)** → `/recommend/cold-start` (GET)
    """,
    version="1.0.0",
)


# -----------------------------------------
# Helper: build recommendations list from DataFrame
# -----------------------------------------

def build_recommendation_list(result_df: pd.DataFrame) -> list[TopicRecommendation]:
    recs = []
    for _, row in result_df.iterrows():
        recs.append(
            TopicRecommendation(
                topic_id=str(row["topic_id"]),
                topic_name=str(row["topic_name"]),
                difficulty=str(row.get("difficulty", "Unknown")),
                estimated_hours=str(row.get("estimated_hours", "")),
                score=float(row["score"]) if "score" in row and pd.notna(row.get("score")) else None,
            )
        )
    return recs


# -----------------------------------------
# Routes
# -----------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    return HealthResponse(status="ok", message="Learning Path Recommender API is running")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(status="ok", message="Learning Path Recommender API is running")


# ── Existing Users ──────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["Existing Users"])
def recommend(request: RecommendRequest):
    """
    Get personalized recommendations for an **existing user** in the system.

    - Looks up the user's completed topics from the interactions data
    - Uses TF-IDF cosine similarity + prerequisite filtering
    - Falls back to cold-start (beginner topics) if user has no completed topics
    """
    interactions_df = app_state["interactions_df"]
    topics_df = app_state["topics_df"]
    prerequisites_df = app_state["prerequisites_df"]
    topic_vectors = app_state["topic_vectors"]

    if request.user_id not in interactions_df["user_id"].values:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Use /recommend/new-user for new users."
        )

    completed = get_completed_topics(request.user_id, interactions_df)
    is_cold_start = len(completed) == 0

    result_df = hybrid_recommend(
        user_id=request.user_id,
        interactions_df=interactions_df,
        prerequisites_df=prerequisites_df,
        topics_df=topics_df,
        topic_vectors=topic_vectors,
        top_k=request.top_k,
    )

    if result_df.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations found for '{request.user_id}'.")

    return RecommendResponse(
        user_id=request.user_id,
        total=len(result_df),
        is_cold_start=is_cold_start,
        is_new_user=False,
        recommendations=build_recommendation_list(result_df),
    )


@app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["Existing Users"])
def recommend_by_id(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    GET version for existing users — easy to test in browser.

    Example: `/recommend/U050?top_k=5`
    """
    return recommend(RecommendRequest(user_id=user_id, top_k=top_k))


@app.get("/user/{user_id}/completed", response_model=CompletedTopicsResponse, tags=["Existing Users"])
def get_user_completed(user_id: str):
    """Returns the list of topics an existing user has already completed."""
    interactions_df = app_state["interactions_df"]

    if user_id not in interactions_df["user_id"].values:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    completed = get_completed_topics(user_id, interactions_df)

    return CompletedTopicsResponse(
        user_id=user_id,
        total_completed=len(completed),
        completed_topic_ids=sorted(list(completed)),
    )


# ── New Users ────────────────────────────────────────────────────────────────

@app.post("/recommend/new-user", response_model=RecommendResponse, tags=["New Users"])
def recommend_new_user(request: NewUserRecommendRequest):
    """
    Get recommendations for a **brand new user** who is not in the system yet.

    The user provides a list of **topic IDs they have already completed**
    (e.g. from a previous course or self-assessment). The system will:

    1. Validate that all provided topic IDs actually exist
    2. Build a TF-IDF user profile from their completed topics
    3. Find similar topics via cosine similarity
    4. Filter out topics whose prerequisites are not met
    5. Return the top recommendations

    **Example request body:**
    ```json
    {
        "user_id": "new_user_john",
        "completed_topic_ids": ["T001", "T002", "T003"],
        "top_k": 5
    }
    ```
    """
    topics_df = app_state["topics_df"]
    prerequisites_df = app_state["prerequisites_df"]
    topic_vectors = app_state["topic_vectors"]

    # ── Validate provided topic IDs ──
    valid_topic_ids = set(topics_df["topic_id"].values)
    invalid = [t for t in request.completed_topic_ids if t not in valid_topic_ids]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown topic IDs: {invalid}. Check /topics for valid IDs."
        )

    completed = set(request.completed_topic_ids)

    # ── Build user profile vector from completed topics ──
    indices = topics_df[topics_df["topic_id"].isin(completed)].index
    user_vec = np.asarray(topic_vectors[indices].mean(axis=0))

    # ── Cosine similarity against all topics ──
    sims = cosine_similarity(user_vec, topic_vectors).flatten()
    topics_scored = topics_df.copy()
    topics_scored["score"] = sims
    topics_scored = topics_scored.sort_values("score", ascending=False)

    # ── Build a temporary interactions DataFrame just for this new user ──
    # This lets us reuse prerequisites_met_for_user() without changes
    temp_interactions = pd.DataFrame({
        "user_id": [request.user_id] * len(completed),
        "topic_id": list(completed),
        "time_spent": [0.0] * len(completed),
        "rating": [0] * len(completed),
        "completion_encoded": [3.0] * len(completed),  # 3.0 = Completed
    })

    # ── Filter: skip completed, check prerequisites ──
    difficulty_map = {0.0: "Beginner", 1.0: "Intermediate", 2.0: "Advanced"}
    final = []

    for _, row in topics_scored.iterrows():
        topic_id = row["topic_id"]

        if topic_id in completed:
            continue

        if prerequisites_met_for_user(request.user_id, topic_id, temp_interactions, prerequisites_df):
            row = row.copy()
            row["difficulty"] = difficulty_map.get(float(row["difficulty_encoded"]), "Unknown")
            final.append(row)

        if len(final) == request.top_k:
            break

    if not final:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found. Try providing more completed topics or check prerequisites."
        )

    result_df = pd.DataFrame(final)

    return RecommendResponse(
        user_id=request.user_id,
        total=len(result_df),
        is_cold_start=False,
        is_new_user=True,
        recommendations=build_recommendation_list(result_df),
    )


@app.get("/recommend/cold-start", response_model=RecommendResponse, tags=["New Users"])
def recommend_cold_start(
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    Get beginner-level recommendations for a **completely new user** with zero prior knowledge.

    No input required — just returns the top beginner topics to start with.
    """
    topics_df = app_state["topics_df"]
    result_df = cold_start_recommend(topics_df, top_k=top_k)

    difficulty_map = {0.0: "Beginner", 1.0: "Intermediate", 2.0: "Advanced"}
    result_df = result_df.copy()
    result_df["difficulty"] = result_df["difficulty_encoded"].map(difficulty_map)

    return RecommendResponse(
        user_id="anonymous",
        total=len(result_df),
        is_cold_start=True,
        is_new_user=True,
        recommendations=build_recommendation_list(result_df),
    )


# ── Topics List ──────────────────────────────────────────────────────────────

@app.get("/topics", tags=["Topics"])
def list_topics():
    """
    Returns all available topic IDs and names.
    Useful for new users to know which topic IDs to pass to /recommend/new-user.
    """
    topics_df = app_state["topics_df"]
    difficulty_map = {"0.0": "Beginner", "1.0": "Intermediate", "2.0": "Advanced"}

    topics = []
    for _, row in topics_df.iterrows():
        topics.append({
            "topic_id": row["topic_id"],
            "topic_name": row["topic_name"],
            "difficulty": difficulty_map.get(str(row["difficulty_encoded"]), "Unknown"),
            "estimated_hours": row["estimated_hours"],
        })

    return {"total": len(topics), "topics": topics}