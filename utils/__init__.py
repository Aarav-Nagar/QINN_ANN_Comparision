"""
Utils package for QINN Research project.

This package contains utility functions and configuration:
- CONFIG: Global configuration object
- setup_logging: Logging configuration function
"""

from .config import CONFIG, setup_logging

__all__ = ['CONFIG', 'setup_logging']
