"""ISOFire airborne fire-temperature retrieval.

AVIRIS-3 two-component Gaussian fire-temperature retrieval. The retrieval
fits a hot + smoldering Gaussian mixture and reports the analytic
moment-matched single Gaussian (mu, sigma) as its public output.

Quickstart
----------
After ``pip install isofire-science`` (or ``uv sync`` from a checkout):

    import isofire

    # The retrieval algorithm
    from isofire import (
        run_moment_matched_retrieval,
        MixtureRetrievalConfig,
        estimate_scene_water_vapor,
    )

    # The physics layer
    from isofire.rt import radiative_transfer_engine, emissive_radiance

    # The CLI / scene runner (also: ``isofire-science --help``)
    from isofire import airborne

The CLI is registered as a console script named ``isofire-science``.
Equivalently:  ``python -m isofire.airborne ...``
"""

from __future__ import annotations

# ---- Version --------------------------------------------------------------
# Single source of truth: pyproject.toml [project].version. We try to read
# the installed distribution metadata first; if the package isn't installed
# (e.g. running from a checkout without `pip install -e .`), fall back to a
# placeholder so imports still work.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("isofire-science")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
except Exception:
    __version__ = "0.0.0+local"


# ---- Public retrieval API -------------------------------------------------
from .retrieval import (
    DEFAULT_ALPHA_TUNING,
    DEFAULT_SIGMA_MIN_K,
    MixtureRetrievalConfig,
    estimate_scene_water_vapor,
    load_fire_library_resampled,
    run_moment_matched_retrieval,
)

# ---- Public physics API ---------------------------------------------------
from .rt import (
    emissive_radiance,
    radiative_transfer_engine,
    resample_spectrum,
)

# ---- CLI module is importable as `isofire.airborne` -----------------------
from . import airborne  # noqa: F401

__all__ = [
    "__version__",
    # retrieval
    "DEFAULT_ALPHA_TUNING",
    "DEFAULT_SIGMA_MIN_K",
    "MixtureRetrievalConfig",
    "estimate_scene_water_vapor",
    "load_fire_library_resampled",
    "run_moment_matched_retrieval",
    # rt
    "emissive_radiance",
    "radiative_transfer_engine",
    "resample_spectrum",
    # CLI module
    "airborne",
]
