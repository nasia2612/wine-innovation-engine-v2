"""Makes pytest put the repository root on sys.path.

This file is intentionally empty of code. pytest adds the directory containing the
rootdir conftest to sys.path under the default "prepend" import mode, which is what lets
`tests/` do `from src.process.simulator import ...` without the package being installed.
Deleting it breaks the test suite.
"""
