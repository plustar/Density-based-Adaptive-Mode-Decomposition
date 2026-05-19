# damd/core/exceptions.py
"""Package-specific exceptions."""
from __future__ import annotations


class DAMDError(Exception):
    """Base class for all damd errors."""


class ConfigError(DAMDError):
    """Invalid configuration."""


class NotFittedError(DAMDError):
    """fit() has not been called."""


class ConvergenceWarning(UserWarning):
    """ADMM / meanshift did not meet tolerance."""
