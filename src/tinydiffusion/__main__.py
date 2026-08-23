"""Entry point for ``python -m tinydiffusion``.

The ``tinydiffusion`` console script is the usual way in, and this runs the
same CLI. It exists because a launcher that starts the processes itself needs a
module to hand them rather than a script on the PATH — which is what a
multi-GPU run is::

    torchrun --nproc_per_node=4 -m tinydiffusion train --config configs/cifar10.toml

See :mod:`tinydiffusion.training.distributed` for what the extra processes then
do differently.
"""

import sys

from tinydiffusion.cli import main

if __name__ == "__main__":
    sys.exit(main())
