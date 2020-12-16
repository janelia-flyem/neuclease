#!/groups/flyem/proj/cluster/miniforge/envs/flyem/bin/python
"""
Copy a directory tree, using multiple processes.

LSF Exmaple:

    bsub -o copy.log -N -n 32 {prog_name} -n 32 /path/to/foo /path/to/bar

This is advantageous on network filesystems with parallel throughput.
On a single machine with a single hard drive, this will likely be
worse than a single-threaded copy.
"""
import sys
__doc__ = __doc__.format(prog_name=sys.argv[0])

import os
import logging
import argparse
import subprocess

from os.path import dirname, abspath, exists
from shutil import copyfile

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', type=int, default=8)
    parser.add_argument('src')
    parser.add_argument('dest')
    args = parser.parse_args()

    args.src = abspath(args.src)
    args.dest = abspath(args.dest)

    if exists(args.dest):
        sys.exit(f"Error: Destination already exists: {args.dest}")

    from neuclease.util import compute_parallel, tqdm_proxy
    from neuclease import configure_default_logging
    configure_default_logging()

    os.chdir(args.src)
    logger.info("Listing source files")
    r = subprocess.run('find . -type f', shell=True, capture_output=True)
    src_paths = r.stdout.decode('utf-8').strip().split('\n')
    dest_paths = [f'{args.dest}/{p}' for p in src_paths]
    dest_dirs = sorted(set([*map(dirname, dest_paths)]))

    logger.info("Initializing directory tree")
    for d in tqdm_proxy(dest_dirs):
        os.makedirs(d, exist_ok=True)

    logger.info(f"Copying {len(src_paths)} files")
    compute_parallel(copyfile, [*zip(src_paths, dest_paths)], 10, starmap=True, ordered=False, processes=args.n)

    logger.info("DONE")


if __name__ == "__main__":
    main()
