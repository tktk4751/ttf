#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Indicators package
"""

from .indicator import Indicator
from .supertrend import Supertrend
from .rsi import RSI
from .choppiness import ChoppinessIndex
from .alpha_ma import AlphaMA
from .alpha_ma_v2 import AlphaMAV2
from .alpha_atr import AlphaATR
from .alpha_choppiness import AlphaChoppiness
from .alpha_adx import AlphaADX
from .alpha_keltner_channel import AlphaKeltnerChannel
from .alpha_ma_v2_keltner import AlphaMAV2KeltnerChannel
from .alpha_bollinger_bands import AlphaBollingerBands
from .alpha_trend import AlphaTrend
from .alpha_momentum import AlphaMomentum
from .alpha_filter import AlphaFilter
from .alpha_macd import AlphaMACD
from .alpha_squeeze import AlphaSqueeze
from .alpha_rsx import AlphaRSX
from .rsx import RSX
from .hma import HMA
from .alpha_donchian import AlphaDonchian
from .alpha_volatility import AlphaVolatility
from .alpha_vix import AlphaVIX
from .alpha_er import AlphaER
from .alpha_vol_band import AlphaVolBand
from .alpha_trend_index import AlphaTrendIndex
from .alpha_trend_filter import AlphaTrendFilter
from .efficiency_ratio import EfficiencyRatio
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .alpha_band import AlphaBand
from .price_source import PriceSource
from .z_ma import ZMA
from .z_atr import ZATR
from .z_adx import ZADX
from .z_rsx import ZRSX
from .z_channel import ZChannel
from .z_trend_index import XTrendIndex
from .z_trend_filter import ZTrendFilter
from .z_donchian import ZDonchian
from .z_long_reversal_index import ZLongReversalIndex
from .z_short_reversal_index import ZShortReversalIndex
from .z_bollinger_bands import ZBollingerBands
from .z_v_channel import ZVChannel
from .c_ma import CMA
from .c_atr import CATR
from .kalman_hull_supertrend import KalmanHullSupertrend
from .kalman_filter import KalmanFilter
from .x_trend import XTrend # Added
from .alma import calculate_alma_numba as calculate_alma
from .dubuc_hurst_exponent import DubucHurstExponent, DubucHurstResult

# エーラーズのドミナントサイクル検出クラス
from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from .ehlers_hody_dc import EhlersHoDyDC
from .ehlers_phac_dc import EhlersPhAcDC
from .ehlers_dudi_dc import EhlersDuDiDC
from .ehlers_dudi_dce import EhlersDuDiDCE
from .ehlers_hody_dce import EhlersHoDyDCE
from .ehlers_phac_dce import EhlersPhAcDCE
# 新しく作成したサイクル検出器
from .ehlers_cycle_period import EhlersCyclePeriod
from .ehlers_cycle_period2 import EhlersCyclePeriod2
from .ehlers_bandpass_zero_crossings import EhlersBandpassZeroCrossings
from .ehlers_autocorrelation_periodogram import EhlersAutocorrelationPeriodogram
from .ehlers_dft_dominant_cycle import EhlersDFTDominantCycle
from .ehlers_multiple_bandpass import EhlersMultipleBandpass
from .ehlers_unified_dc import EhlersUnifiedDC
# 革新的な次世代サイクル検出器
from .ehlers_adaptive_ensemble_cycle import EhlersAdaptiveEnsembleCycle
from .ehlers_quantum_adaptive_cycle import EhlersQuantumAdaptiveCycle
from .ehlers_ultimate_cycle import EhlersUltimateCycle
from .ehlers_supreme_ultimate_cycle import EhlersSupremeUltimateCycle
from .ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle

# エーラーズのドミナントサイクル検出アルゴリズム一覧
__all__ = [
    'Indicator',
    'Supertrend',
    'KalmanHullSupertrend',
    'KalmanFilter',
    'XTrend', # Added
    'RSI',
    'ChoppinessIndex',
    'AlphaMA',
    'AlphaMAV2',
    'AlphaATR',
    'HMA',
    'HyperMA',
    'OmegaMA',
    'AlphaChoppiness',
    'AlphaADX',
    'AlphaKeltnerChannel',
    'AlphaMAV2KeltnerChannel',
    'AlphaBollingerBands',
    'AlphaTrend',
    'AlphaMomentum',
    'AlphaFilter',
    'AlphaMACD',
    'AlphaSqueeze',
    'RSX',
    'AlphaRSX',
    'AlphaDonchian',
    'AlphaVolatility',
    'AlphaVIX',
    'AlphaER',
    'AlphaXMA',
    'AlphaVolBand',
    'AlphaTrendIndex',   # アルファトレンドインデックス
    'AlphaTrendFilter',  # アルファトレンドフィルター
    'EfficiencyRatio',   # 効率比
    'CycleEfficiencyRatio',  # サイクル効率比
    'AlphaBand',         # アルファバンド
    'PriceSource',       # 価格ソース計算
    'ZMA',
    'ZATR',
    'ZADX',
    'ZRSX',
    'ZChannel',
    'ZTrendIndex',       # Zトレンドインデックス
    'ZTrendFilter',      # Zトレンドフィルター
    'ZLongReversalIndex',    # Zロングリバーサルインデックス
    'ZShortReversalIndex',   # Zショートリバーサルインデックス
    'ZDonchian',
    'ZBollingerBands',   # Zボリンジャーバンド
    'ZVChannel',         # ZVチャネル（ZBBとZCのハイブリッド）
    'DubucHurstExponent',    # Dubucハースト指数
    'DubucHurstResult',      # Dubucハースト指数の結果
    'EhlersDominantCycle',
    'DominantCycleResult',
    'EhlersHoDyDC',
    'EhlersPhAcDC',
    'EhlersDuDiDC',
    'EhlersDuDiDCE',
    'EhlersHoDyDCE',
    'EhlersPhAcDCE',
    # 新しく作成したサイクル検出器
    'EhlersCyclePeriod',
    'EhlersCyclePeriod2',
    'EhlersBandpassZeroCrossings',
    'EhlersAutocorrelationPeriodogram',
    'EhlersDFTDominantCycle',
    'EhlersMultipleBandpass',
    'EhlersUnifiedDC',
    'CMA',
    'CATR',
    # 革新的な次世代サイクル検出器
    'EhlersAdaptiveEnsembleCycle',
    'EhlersQuantumAdaptiveCycle',
    'EhlersUltimateCycle',
    'EhlersSupremeUltimateCycle',
    'EhlersAbsoluteUltimateCycle',
]

# Version
__version__ = '0.1.0'
