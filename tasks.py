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
from pathlib import Path

from invoke import task

ENV = "uv run --frozen --"
PYTEST_OPTIONS = ""


@task
def pre_commit(c):
    """[All] Run 'pre-commit' on all files."""
    c.run(f"{ENV} pre-commit install --install-hooks", pty=True)
    c.run(f"{ENV} pre-commit run --all-files", pty=True)


@task
def test(c):
    """[All] Run Unittests via pytest."""
    c.run(f"{ENV} pytest -vv {PYTEST_OPTIONS}", pty=True)
    print(f"See coverage report:\n\n    file://{Path.cwd()}/htmlcov/index.html\n")


@task
def test2ref(c):
    """Run tests and update refdata."""
    c.run("touch .test2ref")
    c.run("rm -rf tests/refdata")
    c.run(f"{ENV} pytest -vv {PYTEST_OPTIONS}", pty=True)
    print(f"See coverage report:\n\n    file://{Path.cwd()}/htmlcov/index.html\n")
    c.run("rm .test2ref")


@task
def checktypes(c):
    """[All] Run Type-Checking via mypy."""
    c.run(f"{ENV} mypy .", pty=True)


@task
def doc(c):
    """[All] Build Documentation via mkdocs."""
    c.run(f"{ENV} mkdocs build --strict")


@task
def doc_serve(c):
    """Start Local Documentation Server via mkdocs."""
    c.run(f"{ENV} mkdocs serve --no-strict", pty=True)


@task
def clean(c):
    """Remove everything mentioned by .gitignore file."""
    c.run("git clean -xdf")


@task
def distclean(c):
    """Remove everything mentioned by .gitignore file and UNTRACKED files."""
    c.run("git clean -xdf")


@task
def shell(c):
    """Open a project specific shell."""
    c.run(f"{ENV} {os.environ.get('SHELL', '/bin/bash')}", pty=True)


@task(pre=[pre_commit, test, checktypes, doc])
def all(c):  # noqa: ARG001
    """Do everything tagged with [ALL]."""
    print("\n    PASS\n")
