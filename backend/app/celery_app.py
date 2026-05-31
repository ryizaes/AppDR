import os
from typing import Any


try:
    from celery import Celery
except Exception:
    Celery = None  # type: ignore[assignment]


REDIS_URL = os.getenv("APPDR_REDIS_URL", "redis://localhost:6379/0")

celery_app: Any | None = None

if Celery is not None:
    celery_app = Celery("appdr", broker=REDIS_URL, backend=REDIS_URL)
    celery_app.conf.update(
        accept_content=["json"],
        result_serializer="json",
        task_serializer="json",
        task_track_started=True,
        timezone="UTC",
    )
