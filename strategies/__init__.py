#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ストラテジーモジュール
"""

from .base.strategy import BaseStrategy
from .implementations.supertrend_rsi_chop.strategy import SupertrendRsiChopStrategy
from .implementations.alpha_trend_predictor.strategy import AlphaTrendPredictorStrategy
from .implementations.alpha_mav2_keltner_filter.strategy import AlphaMAV2KeltnerFilterStrategy
from .implementations.zc_breakout.strategy import ZCBreakoutStrategy
from .implementations.z_trend.strategy import ZTrendStrategy
from .implementations.z_divergence.strategy import ZDivergenceStrategy

__all__ = [
    'BaseStrategy',
    'SupertrendRsiChopStrategy',
    'MultiSignalScoreStrategy',
    'AlphaTrendPredictorStrategy',
    'AlphaMAV2KeltnerFilterStrategy',
    'ZCBreakoutStrategy',
    'ZTrendStrategy',
    'ZDivergenceStrategy'
]
