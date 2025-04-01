#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Signals package
"""

from .base_signal import BaseSignal
from .implementations.rsi.entry import RSICounterTrendEntrySignal
from .implementations.rsi.entry import RSIEntrySignal
from .implementations.rsi.exit import RSIExitSignal
from .implementations.rsi.exit import RSIExit2Signal
from .implementations.rsi.filter import RSIFilterSignal
from .implementations.rsi.filter import RSIRangeFilterSignal
from .implementations.supertrend.direction import SupertrendDirectionSignal
from .implementations.kama_keltner.breakout_entry import KAMAKeltnerBreakoutEntrySignal
from .implementations.donchian.entry import DonchianBreakoutEntrySignal
from .implementations.donchian_atr.entry import DonchianATRBreakoutEntrySignal
from .implementations.chop.filter import ChopFilterSignal
from .implementations.adx.filter import ADXFilterSignal
from .implementations.alma.direction import ALMATrendFollowingStrategy
from .implementations.roc.entry import ROCEntrySignal
from .implementations.squeeze.entry import SqueezeMomentumEntrySignal
from .implementations.alma.entry import ALMACrossoverEntrySignal
from .implementations.bollinger.entry import BollingerCounterTrendEntrySignal
from .implementations.bollinger.exit import BollingerBreakoutExitSignal
from .implementations.candlestick.pinbar_entry import PinbarEntrySignal
from .implementations.candlestick.engulfing_entry import EngulfingEntrySignal
from .implementations.divergence.stoch_rsi_divergence import StochRSIDivergenceSignal
from .implementations.divergence.rsi_divergence import RSIDivergenceSignal
from .implementations.divergence.roc_divergence import ROCDivergenceSignal
from .implementations.divergence.mfi_divergence import MFIDivergenceSignal
from .implementations.divergence.macd_divergence import MACDDivergenceSignal
from .implementations.divergence.alpha_macd_divergence import AlphaMACDDivergenceSignal
from .implementations.divergence.alpha_macd_hidden_divergence import AlphaMACDHiddenDivergenceSignal
# 新しいダイバージェンスシグナル
from .implementations.divergence.z_macd_divergence import ZMACDDivergenceSignal
from .implementations.divergence.z_macd_hidden_divergence import ZMACDHiddenDivergenceSignal
from .implementations.divergence.z_roc_divergence import ZROCDivergenceSignal
from .implementations.divergence.z_roc_hidden_divergence import ZROCHiddenDivergenceSignal

# Direction signals
from .implementations.squeeze.direction import SqueezeDirectionSignal
from .implementations.alpha_squeeze.direction import AlphaSqueezeDirectionSignal
from .implementations.alpha_ma.direction import AlphaMADirectionSignal2, AlphaMATrendFollowingSignal
from .implementations.alpha_trend.direction import AlphaTrendDirectionSignal


# Entry signals

from .implementations.alpha_squeeze.entry import AlphaSqueezeEntrySignal

from .implementations.alpha_momentum.entry import AlphaMomentumEntrySignal

# Keltner signals
from .implementations.alpha_mav2_keltner.breakout_entry import AlphaMAV2KeltnerBreakoutEntrySignal

# Z Channel signals
from .implementations.z_channel.breakout_entry import ZChannelBreakoutEntrySignal

# Z Donchian signals
from .implementations.z_donchian.entry import ZDonchianBreakoutEntrySignal

# Filter signals
from .implementations.alpha_trend_filter.filter import AlphaTrendFilterSignal
from .implementations.z_trend_filter.filter import ZTrendFilterSignal
from .implementations.z_reversal_filter.filter import ZReversalFilterSignal

# Divergence signals
from .implementations.divergence.alpha_roc_divergence import AlphaROCDivergenceSignal

__all__ = [
    'BaseSignal',
    'SupertrendDirectionSignal',
    'RSICounterTrendEntrySignal',
    'RSIEntrySignal',
    'RSIExitSignal',
    'RSIExit2Signal',
    'RSIFilterSignal',
    'RSIRangeFilterSignal',
    'KAMAKeltnerBreakoutEntrySignal',
    'DonchianBreakoutEntrySignal',
    'DonchianATRBreakoutEntrySignal',
    'ZDonchianBreakoutEntrySignal',
    'ChopFilterSignal',
    'ADXFilterSignal',
    'ALMATrendFollowingStrategy',
    'ROCEntrySignal',
    'SqueezeMomentumEntrySignal',
    'ALMACrossoverEntrySignal',
    'BollingerCounterTrendEntrySignal',
    'BollingerBreakoutExitSignal',
    'PinbarEntrySignal',
    'EngulfingEntrySignal',
    'StochRSIDivergenceSignal',
    'RSIDivergenceSignal',
    'ROCDivergenceSignal',
    'MFIDivergenceSignal',
    'MACDDivergenceSignal',
    'AlphaMACDDivergenceSignal',
    'AlphaMACDHiddenDivergenceSignal',
    # 新しいダイバージェンスシグナル
    'ZMACDDivergenceSignal',
    'ZMACDHiddenDivergenceSignal',
    'ZROCDivergenceSignal',
    'ZROCHiddenDivergenceSignal',
    # Direction signals
    'SqueezeDirectionSignal',
    'AlphaSqueezeDirectionSignal',
    'AlphaMADirectionSignal2',
    'AlphaMATrendFollowingSignal',
    'AlphaTrendDirectionSignal',
    'AlphaMomentumDirectionSignal',
    'AlphaFilterDirectionSignal',
    'AlphaKeltnerDirectionSignal',
    # Entry signals
    'SqueezeEntrySignal',
    'AlphaSqueezeEntrySignal',
    'AlphaMAEntrySignal',
    'AlphaTrendEntrySignal',
    'AlphaMomentumEntrySignal',
    'AlphaFilterEntrySignal',
    'AlphaKeltnerEntrySignal',
    'AlphaMAV2KeltnerBreakoutEntrySignal',
    'ZChannelBreakoutEntrySignal',
    'ZDonchianBreakoutEntrySignal',
    # Filter signals
    'AlphaTrendFilterSignal',
    'ZTrendFilterSignal',
    'ZReversalFilterSignal',
    # Exit signals
    'AlphaMAExitSignal',
    'AlphaTrendExitSignal',
    'AlphaMomentumExitSignal',
    'AlphaFilterExitSignal',
    'AlphaKeltnerExitSignal',
    # Divergence signals
    'AlphaROCDivergenceSignal'
]
