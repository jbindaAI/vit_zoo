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
    result = subprocess.run(pytest_args)
    
    # Exit with pytest's exit code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
