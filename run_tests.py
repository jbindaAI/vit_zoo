#!/usr/bin/env python
"""Test runner script for CI/CD pipelines.

This script runs the test suite using pytest with appropriate settings
for continuous integration environments.
"""

import sys
import subprocess


def main():
    """Run the test suite."""
    # Pytest arguments for CI
    pytest_args = [
        "pytest",
        "tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings to reduce noise in CI
    ]
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: pytest not found. Please install test dependencies.", file=sys.stderr)
        print("Run: pip install -e '.[dev]'", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user.", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
