# MIT License
#
# Copyright (c) 2025 ericsmacedo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""invoke tasks.py file."""

import os
import platform
from pathlib import Path

from colorama import init
from invoke import task

ENV = "uv run --frozen --"
PYTEST_OPTIONS = ""


# Initialize colorama so ANSI codes work on Windows too
init(autoreset=True)

ENV = "uv run --frozen --"
PYTEST_OPTIONS = ""
is_windows = platform.system() == "Windows"


def run_cmd(c, cmd, force_color=False):
    """Run a command with cross-platform color handling.

    On Linux/macOS, uses pty=True for proper color passthrough.
    On Windows, avoids pty and optionally forces --color flags.
    """
    if force_color:
        # Try to enforce colors if the tool supports it
        if "pytest" in cmd and "--color" not in cmd:
            cmd += " --color=yes"
        elif "ruff" in cmd and "--color" not in cmd:
            cmd += " --color always"

    if is_windows:
        c.run(cmd)
    else:
        c.run(cmd, pty=True)


@task
def pre_commit(c):
    """[All] Run 'pre-commit' on all files."""
    run_cmd(c, f"{ENV} pre-commit install --install-hooks", force_color=True)
    run_cmd(c, f"{ENV} pre-commit run --all-files", force_color=True)


@task
def test(c):
    """[All] Run Unittests via pytest."""
    run_cmd(c, f"{ENV} pytest -vv {PYTEST_OPTIONS}", force_color=True)
    print(f"See coverage report:\n\n    file://{Path.cwd()}/htmlcov/index.html\n")


@task
def test2ref(c):
    """Run tests and update refdata."""
    run_cmd(c, "touch .test2ref", force_color=True)
    run_cmd(c, "rm -rf tests/refdata", force_color=True)
    run_cmd(c, f"{ENV} pytest -vv {PYTEST_OPTIONS}", force_color=True)
    print(f"See coverage report:\n\n    file://{Path.cwd()}/htmlcov/index.html\n")
    run_cmd(c, "rm .test2ref", force_color=True)


@task
def checktypes(c):
    """[All] Run Type-Checking via mypy."""
    run_cmd(c, f"{ENV} mypy .", force_color=True)


@task
def doc(c):
    """[All] Build Documentation via mkdocs."""
    run_cmd(c, f"{ENV} mkdocs build --strict", force_color=True)


@task
def doc_serve(c):
    """Start Local Documentation Server via mkdocs."""
    run_cmd(c, f"{ENV} mkdocs serve --no-strict", force_color=True)


@task
def clean(c):
    """Remove everything mentioned by .gitignore file."""
    run_cmd(c, "git clean -xdf", force_color=True)


@task
def distclean(c):
    """Remove everything mentioned by .gitignore file and UNTRACKED files."""
    run_cmd(c, "git clean -xdf", force_color=True)


@task
def shell(c):
    """Open a project specific shell."""
    run_cmd(c, f"{ENV} {os.environ.get('SHELL', '/bin/bash')}", force_color=True)


@task(pre=[pre_commit, test, checktypes, doc])
def all(c):  # noqa: ARG001
    """Do everything tagged with [ALL]."""
    print("\n    PASS\n")
