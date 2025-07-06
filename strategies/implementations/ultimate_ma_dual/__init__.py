#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MA Dual Entry Strategy Package

This package implements a dual Ultimate MA strategy that uses:
- Short-term Ultimate MA for entry timing detection
- Long-term Ultimate MA for trend confirmation and exit timing
- Individual cycle detectors for each MA with independent parameters
"""

from .strategy import UltimateMADualStrategy
from .signal_generator import UltimateMADualSignalGenerator

__all__ = [
    'UltimateMADualStrategy',
    'UltimateMADualSignalGenerator'
]

__version__ = "1.0.0"
__author__ = "Ultimate MA Dual Strategy Team" 