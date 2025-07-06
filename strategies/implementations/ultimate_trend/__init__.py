#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Trendストラテジー実装

Ultimate MAフィルタリングシステムとスーパートレンドロジックを統合した
高精度なトレンド検出ストラテジー。

主な特徴:
- ATRベースの動的バンド調整
- Ultimate MAによる高度なフィルタリング
- Numbaによる高速化処理
- オプションのトレンドシグナルフィルタリング
"""

from .strategy import UltimateTrendStrategy
from .signal_generator import UltimateTrendSignalGenerator

__all__ = [
    'UltimateTrendStrategy',
    'UltimateTrendSignalGenerator'
] 