
import time
import requests


def get_json(url, params=None, timeout=30, retries=3):
    """
    Download JSON from an API with automatic retries.
    """

    last_error = None

    for attempt in range(retries):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )

            response.raise_for_status()

            return response.json()

        except Exception as e:

            last_error = e

            print(
                f"Retry {attempt+1}/{retries}..."
            )

            time.sleep(1)

    raise last_error


def safe_get(obj, keys, default=None):
    """
    Safely retrieve nested dictionary/list values.
    """

    current = obj

    for key in keys:

        if isinstance(current, dict):

            current = current.get(key, default)

        elif isinstance(current, list):

            if isinstance(key, int) and key < len(current):
                current = current[key]
            else:
                return default

        else:

            return default

    return current
