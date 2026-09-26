"""
Requirements for installing nanochat on top of an NGC PyTorch image (docker/Dockerfile, TORCH=base).

Writes nanochat's direct dependencies without torch, since the image's PyTorch is kept, and constraints that
keep the image's stack as NVIDIA built it while every package that does change lands on its uv.lock version:
- NVIDIA's own builds stay exactly as installed: distributions with a local version label
  (torch 2.9.0a0+145a3a7, ...) cannot come from PyPI, which rejects those labels;
- a package the image lacks is pinned to its uv.lock version;
- a package the image has is kept if it satisfies nanochat, and otherwise moves to its uv.lock version: it is
  capped at the newer of the two, so it can neither pass the locked version nor fall below the image's;
- the image's own pip constraints ($PIP_CONSTRAINT) apply too, as uv does not read them by itself.

python docker/ngc_requirements.py pyproject.toml lock-export.txt requirements.txt constraints.txt
"""

import os
import re
import sys
import tomllib
from importlib.metadata import distributions

from packaging.version import Version


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement):
    return canonical(re.split(r"[<>=!~;@\[\s]", requirement, maxsplit=1)[0])


def requirement_lines(path):
    with open(path) as f:
        return [line for line in map(str.strip, f) if line and not line.startswith(("#", "-"))]


pyproject, lock_export, requirements_path, constraints_path = sys.argv[1:]
with open(pyproject, "rb") as f:
    project = tomllib.load(f)["project"]
installed = {canonical(d.metadata["Name"]): d.version for d in distributions() if d.metadata["Name"]}

direct = project["dependencies"]
with open(requirements_path, "w") as f:
    f.writelines(r + "\n" for r in direct if requirement_name(r) != "torch")

constraints = [f"{name}=={version}" for name, version in sorted(installed.items()) if "+" in version]
for line in requirement_lines(lock_export):
    name = requirement_name(line)
    pin = re.fullmatch(r"[^=;\s]+==([^;\s]+)\s*(;.*)?", line)
    if name not in installed:
        constraints.append(line)
    elif pin and "+" not in installed[name]:
        locked, marker = pin.groups()
        cap = max(Version(installed[name]), Version(locked))
        constraints.append(f"{name}<={cap}" + (f" {marker}" if marker else ""))
for path in os.environ.get("PIP_CONSTRAINT", "").split():
    constraints += requirement_lines(path)
with open(constraints_path, "w") as f:
    f.writelines(c + "\n" for c in constraints)
