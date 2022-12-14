#! /usr/bin/env python

import logging
import os
import sys

from functools import partial

import toml

from packaging.markers import InvalidMarker
from packaging.markers import Marker
from packaging.markers import UndefinedEnvironmentName


package_repos = {
    "cuda11": [
        "https://download.pytorch.org/whl/cu113",
    ],
    "cpu": [
        "https://download.pytorch.org/whl/cpu",
    ],
}


def should_ignore(name, dep):
    if isinstance(dep, dict):
        text = dep.get("markers", "")
        if text:
            marker = Marker(text)
            return not marker.evaluate(environment={"extra": None})

    return False


# https://github.com/python-poetry/poetry/issues/1301#issue-481070812
# Sometimes implicit packages that work in poetry don't work in pip (eg. torchvision==0.12 and torch==1.9.1)
# If we only allow explicit packages the problem is resolved, but we loose control of implicit versions.
# You can use only_exlicit_packages=True to solve the problem temporarily.
def _poetry_lock_to_requirements_txt(device, only_exlicit_packages=False):
    if only_exlicit_packages:
        with open("pyproject.toml", encoding="utf-8") as fd:
            pyproject = toml.load(fd)

        explicit_packages = pyproject["tool"]["poetry"]["dependencies"].keys()
        install_package = lambda package: package["name"] in explicit_packages

    else:
        install_package = lambda package: package["category"] == "main"

    with open("poetry.lock", encoding="utf-8") as fd:
        poetry_lock = toml.load(fd)

    dont_ignore_packages = {}
    for package in poetry_lock["package"]:
        for name, dep in package.get("dependencies", {}).items():
            dont_ignore_packages[name] = not should_ignore(name, dep)

    package_versions = {
        package["name"]: package["version"]
        for package in poetry_lock["package"]
        if install_package(package) and dont_ignore_packages.get(package["name"], True)
    }

    # Replace intentsify packages with its module version
    lines = []

    for repo in package_repos.get(device, []):
        lines.append(f"--extra-index-url {repo}")

    for package, version in package_versions.items():
        lines.append(f"{package}=={version}")

    return lines


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Generating requirements.txt...")

    device = sys.argv[1]

    lines = _poetry_lock_to_requirements_txt(device, only_exlicit_packages=False)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
