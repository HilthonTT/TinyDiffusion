"""The single source of truth for the package version.

Both the installed distribution metadata (via ``[tool.hatch.version]`` in
``pyproject.toml``) and :data:`tinydiffusion.__version__` read this file, so a
release only ever needs the number changed in one place — and a checkout run
straight from the source tree reports the same version as an installed wheel.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
