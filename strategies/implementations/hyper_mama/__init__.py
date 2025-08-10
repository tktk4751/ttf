#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperMAMA Enhanced Strategy Package

このパッケージには、HyperER効率性による動的適応を持つ
HyperMAMA (Hyper Mother of Adaptive Moving Average) ストラテジーが含まれています。

主要コンポーネント:
- signal_generator.py: HyperMAMAシグナルとフィルターの統合生成器
- strategy.py: メインストラテジークラス

特徴:
- HyperER効率性による動的パラメータ適応（fastlimit: 0.1-0.5, slowlimit: 0.01-0.05）
- 4つの高度なフィルターオプション（Phasor Trend, Correlation Cycle, Correlation Trend, Unified Trend Cycle）
- Numba JIT最適化による高速処理
- Optunaによる最適化サポート
"""

from .strategy import HyperMAMAEnhancedStrategy
from .signal_generator import HyperMAMAEnhancedSignalGenerator, FilterType

__all__ = [
    'HyperMAMAEnhancedStrategy',
    'HyperMAMAEnhancedSignalGenerator', 
    'FilterType'
]

__version__ = '1.0.0'
__author__ = 'TTF Development Team'