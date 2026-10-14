"""Entry point for ``python -m tinydiffusion.cli``.

The same CLI as ``python -m tinydiffusion`` and the ``tinydiffusion`` console
script. The wrappers in ``scripts/`` launch it by this name, so it has to keep
working now that ``cli`` is a package rather than a module.
"""

import sys

from tinydiffusion.cli import main

if __name__ == "__main__":
    sys.exit(main())
