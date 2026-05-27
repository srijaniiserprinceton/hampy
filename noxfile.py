# /// script
# requires-python = ">=3.11"
# dependencies = ["nox", "uv"]
# ///

"""
Nox is an automation tool used to perform checks and run tasks.

To run one of the "sessions" defined in noxfile.py, run:

   nox -s '<session_name>'

where <session_name> is replaced with the name of the session. Running
tests on Python 3.14 can be done with `nox -s 'tests-3.14'`.

To see available Nox sessions, run `nox -l`.

Nox documentation: https://nox.thea.codes
"""

import nox

# Use uv to create the Python environment if available, with backup options.
nox.options.default_venv_backend = "uv|micromamba|mamba|conda|virtualenv"

SUPPORTED_PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")

# When running `nox` without specifying a session, run the tests session
# on the most recent supported version of Python.
nox.options.sessions = [f"tests-{SUPPORTED_PYTHON_VERSIONS[-1]}"]


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Create an environment and run pytest."""
    session.install("pytest", "jax", "numpy", "scipy", "matplotlib", ".")
    session.run("pytest")
