"""Read one environment variable (from .env or the shell) and check its value.

Every reader takes the variable name and a default written as text, exactly as
it would appear in .env, and raises ValueError with the variable name when the
value is not usable.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path_from_project_root(value: str) -> Path:
    """Absolute paths stay as they are; relative paths start at the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_path(variable_name: str, default_value: str) -> Path:
    return resolve_path_from_project_root(os.environ.get(variable_name, default_value))


def read_optional_path(variable_name: str, default_value: str) -> Path | None:
    """Like read_path, but an explicitly empty value disables the file (returns None)."""
    value = os.environ.get(variable_name, default_value)
    if not value.strip():
        return None
    return resolve_path_from_project_root(value)


def read_integer_at_least(variable_name: str, default_value: str, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(variable_name, default_value))
    except ValueError as error:
        raise ValueError(f"{variable_name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{variable_name} must be {minimum} or greater")
    return value


def read_decimal_number(variable_name: str, default_value: str) -> float:
    try:
        return float(os.environ.get(variable_name, default_value))
    except ValueError as error:
        raise ValueError(f"{variable_name} must be a number") from error


def read_percentage(variable_name: str, default_value: str = "0") -> float:
    value = read_decimal_number(variable_name, default_value)
    if not 0 <= value <= 100:
        raise ValueError(f"{variable_name} must be between 0 and 100")
    return value


def read_true_or_false(variable_name: str, default_value: str) -> bool:
    """"1", "true" and "yes" (any case) mean True; anything else means False."""
    value = os.environ.get(variable_name, default_value)
    return value.strip().lower() in {"1", "true", "yes"}


def read_text(variable_name: str, default_value: str) -> str:
    return os.environ.get(variable_name, default_value)
