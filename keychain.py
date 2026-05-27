"""Fetch the OpenRouter API key from macOS Keychain.

The key is stored under service "openrouter" using:
    security add-generic-password -s openrouter -a openrouter -w <key>

This avoids ever placing the key in source files or shell history.
"""
from __future__ import annotations

import subprocess


def get_openrouter_key() -> str:
    """Return the OpenRouter API key from macOS Keychain.

    Raises RuntimeError if the key is not found.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "openrouter", "-w"],
            capture_output=True,
            text=True,
            check=True,
        )
        key = result.stdout.strip()
        if not key:
            raise RuntimeError("Keychain returned empty key for service 'openrouter'.")
        return key
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Could not retrieve 'openrouter' key from macOS Keychain. "
            "Add it with: security add-generic-password -s openrouter -a openrouter -w <key>"
        ) from e


if __name__ == "__main__":
    key = get_openrouter_key()
    print(f"Key loaded from Keychain (starts with {key[:11]}..., length {len(key)}).")
