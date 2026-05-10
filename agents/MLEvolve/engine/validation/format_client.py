"""Validation server client: health check and submission validate."""

import logging
import os
import time

import requests

logger = logging.getLogger("MLEvolve")


def get_server_url_list():
    """Return server URL list (env GRADING_SERVER_PORT or default)."""
    server_port = os.getenv("GRADING_SERVER_PORT", "5005")
    return [f"http://127.0.0.1:{server_port}"]


server_url_list = get_server_url_list()


def is_server_online(max_retries=3, timeout=300):
    server_url_list = get_server_url_list()
    retry = 0
    index = 0
    server_url = server_url_list[index]
    while retry < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is online, status code: {response.status_code}")
                return True, server_url
            else:
                logger.warning(f"Server returned non-200 status code: {response.status_code}")
                logger.warning(f"Response body: {response.text[:500]}")
                logger.warning(f"Response headers: {dict(response.headers)}")

        except requests.exceptions.Timeout:
            timeout += 20
            logger.error(f"Connection to {server_url} timed out.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            index += 1
            index = index%(len(server_url_list))
            server_url = server_url_list[index]
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
    logger.error(f"Server is not online after {max_retries} retries.")
    return False, server_url


def call_validate(exp_id, submission_path, timeout=300, max_retries=3):
    online, server_url = is_server_online()
    retry=0
    while retry < max_retries:
        try:
            if online:
                with open(submission_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(f"{server_url}/validate", files=files, headers={"exp-id": exp_id}, timeout=timeout)
                response_json = response.json()
                if "error" in response_json:
                    logger.error(f"Server returned error: {response.text}")
                    return False, response_json['details']
                else:
                    return True, response_json
            else:
                return False, f"Server at {server_url} is not online"
        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
        else:
            return False, ""
