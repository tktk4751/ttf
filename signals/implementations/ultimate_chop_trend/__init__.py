#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Chop Trend V3 シグナル実装

Ultimate Chop Trend V3指標によるエントリー・決済シグナル生成
トレンド方向の変化を検出してタイミングよくエントリー/エグジット
"""

from .ultimate_chop_trend_entry import UltimateChopTrendEntrySignal

__all__ = [
    'UltimateChopTrendEntrySignal'
] 