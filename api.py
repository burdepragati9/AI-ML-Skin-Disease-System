import json
import os
import time
from collections import defaultdict, deque

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from analytics.admin_analytics import ai_recognized_images, newly_learned_images_count, training_status
from database.db import init_db


RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_REQUESTS = int(os.getenv("API_RATE_LIMIT_REQUESTS", "60"))
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")
_REQUESTS = defaultdict(deque)


def _rows(rows):
    return [dict(row) for row in rows]


def _guard(request):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    entries = _REQUESTS[client]
    while entries and now - entries[0] > RATE_LIMIT_WINDOW_SECONDS:
        entries.popleft()
    if len(entries) >= RATE_LIMIT_REQUESTS:
        return JSONResponse({"error": "Rate limit exceeded."}, status_code=429)
    entries.append(now)

    if ADMIN_API_TOKEN:
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {ADMIN_API_TOKEN}"
        if auth_header != expected:
            return JSONResponse({"error": "Unauthorized."}, status_code=401)
    return None


async def get_training_status(request):
    guarded = _guard(request)
    if guarded:
        return guarded
    return JSONResponse(training_status())


async def get_newly_learned_count(request):
    guarded = _guard(request)
    if guarded:
        return guarded
    return JSONResponse({"count": newly_learned_images_count()})


async def get_ai_images(request):
    guarded = _guard(request)
    if guarded:
        return guarded
    limit = int(request.query_params.get("limit", "100"))
    return JSONResponse({"items": _rows(ai_recognized_images(limit=min(limit, 500)))})


init_db()
app = Starlette(
    routes=[
        Route("/api/training/status", get_training_status, methods=["GET"]),
        Route("/api/training/newly-learned-count", get_newly_learned_count, methods=["GET"]),
        Route("/api/ai-recognized-images", get_ai_images, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
