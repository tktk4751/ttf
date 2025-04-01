#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaMAシグナルモジュール
"""

from .direction import (
    AlphaMATrendFollowingSignal,
    AlphaMACirculationSignal,
    AlphaMADirectionSignal2,
    AlphaMATripleDirectionSignal
)

from .entry import (
    AlphaMACrossoverEntrySignal,
    AlphaMATripleCrossoverEntrySignal,
    AlphaMACDCrossoverEntrySignal
)

__all__ = [
    'AlphaMATrendFollowingStrategy',
    'AlphaMACirculationSignal',
    'AlphaMADirectionSignal2',
    'AlphaMATripleDirectionSignal',
    'AlphaMACrossoverEntrySignal',
    'AlphaMATripleCrossoverEntrySignal',
    'AlphaMACDCrossoverEntrySignal'
] 