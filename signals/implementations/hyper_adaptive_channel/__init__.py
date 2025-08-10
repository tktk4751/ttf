#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper Adaptive Channel シグナル実装

ハイパーアダプティブチャネルインジケーターに基づくシグナル実装
"""

from .breakout_entry import HyperAdaptiveChannelBreakoutEntrySignal
from .signal_factory import (
    create_hyper_frama_breakout_signal,
    create_ultimate_ma_breakout_signal,
    create_laguerre_breakout_signal,
    create_super_smoother_breakout_signal,
    create_z_adaptive_breakout_signal,
    create_custom_atr_breakout_signal,
    create_high_sensitivity_signal,
    create_low_sensitivity_signal,
    create_signal_by_preset,
    SIGNAL_PRESETS
)

__all__ = [
    'HyperAdaptiveChannelBreakoutEntrySignal',
    # ファクトリー関数
    'create_hyper_frama_breakout_signal',
    'create_ultimate_ma_breakout_signal',
    'create_laguerre_breakout_signal',
    'create_super_smoother_breakout_signal',
    'create_z_adaptive_breakout_signal',
    'create_custom_atr_breakout_signal',
    'create_high_sensitivity_signal',
    'create_low_sensitivity_signal',
    'create_signal_by_preset',
    'SIGNAL_PRESETS'
]