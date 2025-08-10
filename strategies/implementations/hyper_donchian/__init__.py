#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperドンチャンブレイクアウトストラテジー

Min/Max範囲内80-20%位置ベースのHyperドンチャンチャネルを使用したブレイクアウト戦略。
従来のドンチャンチャネルよりも外れ値に堅牢で安定性が向上。

主要機能:
- Hyperドンチャンチャネルブレイクアウトエントリーシグナル
- 複数のトレンドフィルター（HyperER, HyperTrendIndex, HyperADX）
- コンセンサスフィルタリング（3つのうち2つが同意）
- HyperER動的適応サポート
- Optuna最適化対応
"""

from .strategy import HyperDonchianStrategy
from .signal_generator import HyperDonchianSignalGenerator, FilterType

__all__ = [
    'HyperDonchianStrategy',
    'HyperDonchianSignalGenerator',
    'FilterType'
]