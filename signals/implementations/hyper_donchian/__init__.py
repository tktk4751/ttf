#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperドンチャンブレイクアウトシグナル

Min/Max範囲内80-20%位置ベースのHyperドンチャンチャネルを使用したブレイクアウトシグナル。
従来のドンチャンチャネルよりも安定したブレイクアウト検出が可能。
"""

from .entry import HyperDonchianBreakoutEntrySignal

__all__ = [
    'HyperDonchianBreakoutEntrySignal'
]