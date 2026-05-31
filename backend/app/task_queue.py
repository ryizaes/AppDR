import base64
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any
from uuid import uuid4

from app.celery_app import celery_app
from app.pipeline import analyze_image, build_analyze_response


LOCAL_WORKERS = max(1, int(os.getenv("APPDR_LOCAL_WORKERS", "2")))
USE_CELERY = os.getenv("APPDR_USE_CELERY", "0").lower() in {"1", "true", "yes"}

_LOCAL_EXECUTOR = ThreadPoolExecutor(max_workers=LOCAL_WORKERS)
_LOCAL_TASKS: dict[str, dict[str, Any]] = {}
_LOCAL_TASK_LOCK = Lock()


def submit_analysis(
    filename: str,
    image_bytes: bytes,
    calibration: dict[str, float] | None = None,
) -> str:
    if celery_enabled():
        from app.tasks import analyze_image_task

        encoded = base64.b64encode(image_bytes).decode("ascii")
        async_result = analyze_image_task.delay(filename, encoded, calibration or {})
        return str(async_result.id)

    task_id = uuid4().hex

    with _LOCAL_TASK_LOCK:
        _LOCAL_TASKS[task_id] = {
            "task_id": task_id,
            "state": "PENDING",
            "message": "Queued for local background processing.",
            "result": None,
            "error": None,
        }

    _LOCAL_EXECUTOR.submit(
        run_local_analysis,
        task_id,
        filename,
        image_bytes,
        calibration or {},
    )
    return task_id


def get_task_status(task_id: str) -> dict[str, Any] | None:
    with _LOCAL_TASK_LOCK:
        local_record = _LOCAL_TASKS.get(task_id)

    if local_record is not None:
        return dict(local_record)

    if not celery_enabled():
        return None

    from app.tasks import analyze_image_task

    async_result = analyze_image_task.AsyncResult(task_id)
    state = str(async_result.state)

    if state == "PENDING":
        return pending_status(task_id, "Queued or waiting for a Celery worker.")
    if state == "STARTED":
        return pending_status(task_id, "Classical retinal feature extraction is running.")
    if state == "SUCCESS":
        return {
            "task_id": task_id,
            "state": "SUCCESS",
            "message": "Analysis completed.",
            "result": async_result.result,
            "error": None,
        }
    if state == "FAILURE":
        return {
            "task_id": task_id,
            "state": "FAILURE",
            "message": "Analysis failed.",
            "result": None,
            "error": str(async_result.result),
        }

    return pending_status(task_id, f"Task state: {state}.")


def celery_enabled() -> bool:
    return USE_CELERY and celery_app is not None


def run_local_analysis(
    task_id: str,
    filename: str,
    image_bytes: bytes,
    calibration: dict[str, float],
) -> None:
    update_local_task(
        task_id,
        state="STARTED",
        message="Classical retinal feature extraction is running.",
    )

    try:
        output = analyze_image(image_bytes, calibration=calibration)
        response = build_analyze_response(filename, output)
        update_local_task(
            task_id,
            state="SUCCESS",
            message="Analysis completed.",
            result=response.model_dump(mode="json"),
            error=None,
        )
    except Exception as error:
        update_local_task(
            task_id,
            state="FAILURE",
            message="Analysis failed.",
            result=None,
            error=f"{error}\n{traceback.format_exc(limit=3)}",
        )


def update_local_task(task_id: str, **updates: Any) -> None:
    with _LOCAL_TASK_LOCK:
        current = _LOCAL_TASKS.get(task_id, {"task_id": task_id})
        current.update(updates)
        _LOCAL_TASKS[task_id] = current


def pending_status(task_id: str, message: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "state": "PENDING",
        "message": message,
        "result": None,
        "error": None,
    }
