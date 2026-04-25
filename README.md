# Attention Gravity Engine (AGE) — GitHub Repo Structure

A working starter repository for a closed-loop recommendation / attention-state system.

## Repository tree

```text
attention-gravity-engine/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ docker-compose.yml
├─ Dockerfile
├─ Makefile
├─ .gitignore
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sample_events.jsonl
├─ models/
│  ├─ embeddings/
│  └─ ranking/
├─ notebooks/
│  └─ exploration.ipynb
├─ scripts/
│  ├─ ingest_sample_data.py
│  ├─ build_embeddings.py
│  ├─ train_ranker.py
│  └─ run_api.sh
├─ src/
│  └─ age/
│     ├─ __init__.py
│     ├─ api/
│     │  ├─ __init__.py
│     │  └─ app.py
│     ├─ config/
│     │  ├─ __init__.py
│     │  └─ settings.py
│     ├─ core/
│     │  ├─ __init__.py
│     │  ├─ schemas.py
│     │  ├─ state.py
│     │  └─ scoring.py
│     ├─ ingestion/
│     │  ├─ __init__.py
│     │  └─ events.py
│     ├─ retrieval/
│     │  ├─ __init__.py
│     │  └─ vector_store.py
│     ├─ ranking/
│     │  ├─ __init__.py
│     │  ├─ ranker.py
│     │  └─ rerank.py
│     ├─ feedback/
│     │  ├─ __init__.py
│     │  └─ processor.py
│     ├─ training/
│     │  ├─ __init__.py
│     │  ├─ dataset.py
│     │  ├─ embeddings.py
│     │  └─ train.py
│     └─ utils/
│        ├─ __init__.py
│        ├─ logging.py
│        └─ time.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_api.py
│  ├─ test_state.py
│  ├─ test_scoring.py
│  └─ test_ranking.py
└─ infra/
   ├─ postgres/
   │  └─ init.sql
   ├─ kafka/
   │  └─ topics.md
   └─ deployment/
      └─ k8s/
         ├─ api-deployment.yaml
         └─ worker-deployment.yaml
```

---

## What each part does

### `src/age/api/`

FastAPI service exposing event ingestion and recommendation endpoints.

### `src/age/core/`

Shared domain logic: schemas, state representation, scoring functions.

### `src/age/ingestion/`

Normalizes raw behavior events and prepares them for storage or streaming.

### `src/age/retrieval/`

Vector-search retrieval layer for candidate generation.

### `src/age/ranking/`

Ranking and re-ranking logic, including novelty and diversity control.

### `src/age/feedback/`

Converts behavior into reward signals and state updates.

### `src/age/training/`

Offline jobs for embeddings, datasets, and model training.

### `infra/`

Infrastructure resources for local setup and deployment.

---

## Minimal file contents to start with

### `src/age/api/app.py`

```python
from fastapi import FastAPI
from age.core.schemas import EventIn, RecommendationResponse
from age.ingestion.events import ingest_event
from age.core.state import get_user_state
from age.retrieval.vector_store import retrieve_candidates
from age.ranking.ranker import rank_candidates

app = FastAPI(title="Attention Gravity Engine")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/event")
def post_event(event: EventIn):
    ingest_event(event)
    return {"status": "accepted"}


@app.get("/recommendations", response_model=RecommendationResponse)
def recommendations(user_id: str, k: int = 10):
    state = get_user_state(user_id)
    candidates = retrieve_candidates(state.vector, top_k=100)
    ranked = rank_candidates(state, candidates, top_k=k)
    return RecommendationResponse(items=ranked)
```

### `src/age/core/schemas.py`

```python
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class EventIn(BaseModel):
    user_id: str
    item_id: Optional[str] = None
    event_type: str
    value: float = 1.0
    timestamp: Optional[str] = None
    context: Dict[str, Any] = {}


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    metadata: Dict[str, Any] = {}


class RecommendationResponse(BaseModel):
    items: List[RecommendationItem]
```

### `src/age/core/state.py`

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class UserState:
    user_id: str
    vector: List[float] = field(default_factory=list)
    session_vector: List[float] = field(default_factory=list)
    long_term_vector: List[float] = field(default_factory=list)


def get_user_state(user_id: str) -> UserState:
    return UserState(user_id=user_id, vector=[0.0] * 128)
```

### `src/age/ranking/ranker.py`

```python
from age.core.schemas import RecommendationItem


def rank_candidates(user_state, candidates, top_k: int = 10):
    ranked = []
    for c in candidates:
        score = c.get("similarity", 0.0) * 0.6 + c.get("recency", 0.0) * 0.2 + c.get("diversity", 0.0) * 0.2
        ranked.append(RecommendationItem(item_id=c["item_id"], score=score, metadata=c.get("metadata", {})))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
```

### `src/age/retrieval/vector_store.py`

```python
def retrieve_candidates(user_vector, top_k: int = 100):
    # Replace with FAISS / pgvector / Pinecone later
    return [
        {"item_id": f"item_{i}", "similarity": 1.0 - i * 0.01, "recency": 0.5, "diversity": 0.5, "metadata": {}}
        for i in range(top_k)
    ]
```

### `src/age/ingestion/events.py`

```python
from age.core.schemas import EventIn


def ingest_event(event: EventIn):
    # Write to DB / stream / log file here
    print(f"ingested: {event.user_id} {event.event_type}")
```

### `src/age/feedback/processor.py`

```python
def reward_from_event(event_type: str, value: float) -> float:
    mapping = {
        "click": 1.0,
        "dwell": 0.7,
        "save": 1.2,
        "skip": -0.8,
        "hide": -1.2,
    }
    return mapping.get(event_type, 0.0) * value
```

---

## Suggested dependencies

### `pyproject.toml`

```toml
[project]
name = "attention-gravity-engine"
version = "0.1.0"
description = "Closed-loop attention-state recommendation engine"
requires-python = ">=3.11"
dependencies = [
  "fastapi",
  "uvicorn",
  "pydantic",
  "numpy",
  "pandas",
  "scikit-learn",
  "torch",
  "sentence-transformers",
  "sqlalchemy",
  "psycopg2-binary",
  "redis",
  "python-dotenv",
  "pytest",
]
```

---

## Local development flow

1. Start services with Docker Compose.
2. Run the API.
3. Send events to `/event`.
4. Request recommendations from `/recommendations`.
5. Train offline models in `src/age/training/`.
6. Swap mock retrieval for FAISS / pgvector.

---

## First milestone

A good first milestone is:

* log events
* maintain a simple user vector
* retrieve mock candidates
* rank them
* return a feed through FastAPI

That gives you a working closed loop before adding heavy ML.

---

## Next layer to add

After the scaffold works, add:

* PostgreSQL persistence
* vector search
* bandit exploration
* A/B testing
* dashboard for state evolution
