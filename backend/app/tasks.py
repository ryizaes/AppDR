import base64
from typing import Any

from app.celery_app import celery_app
from app.pipeline import analyze_image, build_analyze_response


if celery_app is not None:

    @celery_app.task(name="appdr.analyze_image")
    def analyze_image_task(
        filename: str,
        image_base64: str,
    ) -> dict[str, Any]:
        image_bytes = base64.b64decode(image_base64.encode("ascii"))
        output = analyze_image(image_bytes)
        return build_analyze_response(filename, output).model_dump(mode="json")

else:

    def analyze_image_task(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Celery is not installed or configured.")
