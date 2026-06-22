"""Resolve Hugging Face Hub token from machine-specific config or environment variables."""

import os

from nicetoolbox.configs.schemas.machine_specific_paths import MachineSpecificConfig


def effective_hf_hub_token(machine: MachineSpecificConfig | None = None) -> str | None:
    """
    Return a non-empty token string from `hugging_face_token` in machine_specific_paths.toml,
    falling back to the environment variable 'HF_TOKEN'.
    """

    # 1. Machine-specific config (used by local development)
    if machine is not None and machine.hugging_face_token.strip():
        return machine.hugging_face_token.strip()

    # 2. Fallback to environment variable (used by Docker CI/CD)
    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    return None
