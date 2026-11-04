"""Convert a conda-lock file into a conda explicit spec for linux-64.

``conda create --file explicit-spec.txt`` installs exact package builds
without solving, which is faster and reproducible on HPC systems. Pip
packages cannot be expressed in an explicit spec; the script prints
the ``pip install`` command for them instead.

Example
-------
``python parse_lock.py poseidon-linux-64-lock.yml explicit-spec.txt``
"""

import argparse
import sys
import urllib.parse

import yaml

#: Required first line of a conda explicit spec.
EXPLICIT_HEADER = "@EXPLICIT"


def explicit_spec(lock, platform="linux-64"):
    """
    Build explicit spec lines and pip requirements from a lock file.

    Parameters
    ----------
    lock : dict
        Parsed conda-lock YAML.
    platform : str, optional
        Platform to extract. Default is ``"linux-64"``.

    Returns
    -------
    conda_lines : list of str
        Explicit spec lines, starting with :data:`EXPLICIT_HEADER`.
    pip_requirements : list of str
        ``name==version`` pins for pip-managed packages.
    """
    conda_lines, pip_requirements = [EXPLICIT_HEADER], []
    for package in lock.get("package", []):
        if package.get("platform") != platform:
            continue
        if package.get("manager") == "conda":
            # Decode escapes such as %21 so conda accepts the version
            url = urllib.parse.unquote(package["url"])
            md5 = package.get("hash", {}).get("md5")
            conda_lines.append(f"{url}#{md5}" if md5 else url)
        elif package.get("manager") == "pip":
            pip_requirements.append(f"{package['name']}=={package['version']}")
    return conda_lines, pip_requirements


def main(argv=None):
    """
    Write the explicit spec and report any pip requirements.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if the lock file is missing, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lock_file", nargs="?",
                        default="poseidon-linux-64-lock.yml",
                        help="conda-lock YAML to read.")
    parser.add_argument("output_file", nargs="?",
                        default="explicit-spec.txt",
                        help="Explicit spec to write.")
    args = parser.parse_args(argv)

    try:
        with open(args.lock_file) as handle:
            lock = yaml.safe_load(handle)
    except FileNotFoundError:
        print(f"Lock file not found: {args.lock_file}", file=sys.stderr)
        return 1

    conda_lines, pip_requirements = explicit_spec(lock)
    with open(args.output_file, "w") as handle:
        handle.write("\n".join(conda_lines))
    print(f"Wrote {len(conda_lines) - 1} conda packages to "
          f"{args.output_file}.")
    if pip_requirements:
        print("Pip packages are not part of the explicit spec; after "
              "activating the environment run:")
        print(f"pip install {' '.join(pip_requirements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
