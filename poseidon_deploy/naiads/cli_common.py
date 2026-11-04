"""Shared command-line helpers for the POSEIDON HPC driver scripts.

Every ``run_*.py`` script in this folder imports this module, which is
found because Python puts a script's own folder on ``sys.path``.
"""

import argparse
import logging
import os
import sys

import numpy as np

from poseidon_utils.logging_utils import setup_logging
from poseidon_utils.site_config import load_site_config

logger = logging.getLogger("poseidon.naiads")


def make_parser(description):
    """
    Create an argument parser with consistent help formatting.

    Parameters
    ----------
    description : str
        Script description shown by ``--help``.

    Returns
    -------
    argparse.ArgumentParser
        Parser that also prints argument defaults.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_site_argument(parser):
    """
    Add the ``--site`` option selecting a site configuration.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    """
    parser.add_argument(
        "--site", required=True,
        help="Site name in poseidon_deploy/configs/sites (e.g. "
             "carolina_beach) or a path to a site YAML file.")


def add_gpu_argument(parser):
    """
    Add the ``--disable_gpu`` flag (GPU is used when available).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    """
    parser.add_argument("--disable_gpu", action="store_true",
                        help="Run on the CPU even if a GPU is available.")


def gpu_choice(args):
    """
    Translate ``--disable_gpu`` into a ``use_gpu`` argument.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``disable_gpu``.

    Returns
    -------
    bool or None
        False when disabled, otherwise None (use a GPU if available).
    """
    return False if args.disable_gpu else None


def start(parser, argv=None):
    """
    Configure logging, parse arguments, and log them.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Fully configured parser.
    argv : list of str, optional
        Arguments to parse. Default reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    setup_logging()
    args = parser.parse_args(argv)
    logger.info("%s %s", os.path.basename(sys.argv[0]), vars(args))
    return args


def load_site(args):
    """
    Load the site configuration named by ``--site``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``site``.

    Returns
    -------
    poseidon_utils.site_config.SiteConfig
        The site configuration.
    """
    site = load_site_config(args.site)
    logger.info("Loaded site '%s' (camera %s)", site.name, site.camera_name)
    return site


def shared_grid_path(grid_dir, site):
    """
    Return the path of a site's precomputed grid archive.

    Parameters
    ----------
    grid_dir : str
        Folder of shared grids.
    site : poseidon_utils.site_config.SiteConfig
        Site configuration.

    Returns
    -------
    str
        ``<grid_dir>/shared_grid_<site name>.npz``.
    """
    return os.path.join(grid_dir, f"shared_grid_{site.name}.npz")


def load_shared_grid(grid_dir, site):
    """
    Load the grid written by ``run_prep_grid.py``, exiting if missing.

    Parameters
    ----------
    grid_dir : str
        Folder of shared grids.
    site : poseidon_utils.site_config.SiteConfig
        Site configuration.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        Archive with ``x``, ``y`` and ``z`` grids.
    """
    path = shared_grid_path(grid_dir, site)
    if not os.path.exists(path):
        fail(f"Grid not found: {path}. Run run_prep_grid.py first.")
    logger.info("Loading grid %s", path)
    return np.load(path)


def require_dir(path, what):
    """
    Exit with an error if a folder does not exist.

    Parameters
    ----------
    path : str
        Folder to check.
    what : str
        Description used in the error message.
    """
    if not os.path.isdir(path):
        fail(f"{what} not found: {path}")


def fail(message):
    """
    Log an error and exit with status 1.

    Parameters
    ----------
    message : str
        Error message.
    """
    logger.error(message)
    sys.exit(1)
