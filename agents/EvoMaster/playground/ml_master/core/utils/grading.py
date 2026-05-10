from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Iterable, Tuple

from .grading_server import ensure_grading_server, stop_grading_server


try:
    import requests

    _HAS_REQUESTS = True
    _REQ_ERR: str | None = None
except ImportError as exc:
    _HAS_REQUESTS = False
    _REQ_ERR = str(exc)

logger = logging.getLogger(__name__)


def _post_validate(
    *,
    server_url: str,
    exp_id: str,
    submission_path: Path,
    timeout: int,
) -> tuple[bool, dict | str]:
    """Execute post validate.

    Args:
        server_url: Value for server url.
        exp_id: Identifier string.
        submission_path: Path to the submission CSV file.
        timeout: Numeric control parameter.

    Returns:
        tuple[bool, dict | str]: Result of this function.
    """
    import requests

    with submission_path.open("rb") as file_obj:
        files = {"file": file_obj}
        response = requests.post(
            f"{server_url}/validate",
            files=files,
            headers={"exp-id": exp_id},
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "error" in data:
        return False, data.get("details", data["error"])
    return True, data


def is_server_online(
    server_urls: Iterable[str],
    timeout: int = 60,
    max_retries: int | None = None,
) -> Tuple[bool, str]:
    """Check grading server health status."""
    if not _HAS_REQUESTS:
        logger.warning("requests not installed; skip grading health check. missing=%s", _REQ_ERR)
        return False, ""

    urls = [u.rstrip("/") for u in server_urls if u]
    if not urls:
        return False, ""

    retries = max_retries or len(urls)
    start_index = random.randrange(len(urls))

    for attempt in range(retries):
        url = urls[(start_index + attempt) % len(urls)]
        try:
            import requests

            response = requests.get(f"{url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info("grading server online: %s", url)
                return True, url
            logger.warning("grading health non-200: %s (%s)", response.status_code, url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("grading health check failed (%s): %s", url, exc)
        time.sleep(1)
    return False, ""


def validate_submission(
    exp_id: str,
    submission_path: Path,
    *,
    server_urls: Iterable[str],
    dataset_root: str | Path | None = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> Tuple[bool, dict | str]:
    """Validate submission.csv via grading service."""
    if not _HAS_REQUESTS:
        msg = f"requests not installed; cannot call grading server ({_REQ_ERR})"
        logger.warning(msg)
        return False, msg

    submission_path = Path(submission_path)
    if not submission_path.exists():
        return False, f"submission file not found: {submission_path}"

    urls = [u.rstrip("/") for u in server_urls if u]
    if dataset_root:
        started_url = ensure_grading_server(dataset_root, server_urls=urls)
        if started_url and started_url not in urls:
            urls.append(started_url)

    online, server_url = is_server_online(urls, timeout=timeout, max_retries=max_retries)
    if not online:
        return False, "grading server unavailable"

    for attempt in range(1, max_retries + 1):
        try:
            return _post_validate(
                server_url=server_url,
                exp_id=exp_id,
                submission_path=submission_path,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "grading validate failed (%s), attempt %s/%s: %s",
                server_url,
                attempt,
                max_retries,
                exc,
            )
            time.sleep(1)

    return False, "grading server call failed"


def shutdown_embedded_grading_server(timeout: int = 5) -> bool:
    """Shutdown embedded grading server started by current process."""
    return stop_grading_server(timeout=timeout)

shutdown_embedded_grading_server(5)
