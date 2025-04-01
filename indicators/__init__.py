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
from .omega_ma import OmegaMA
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
from .alpha_donchian import AlphaDonchian
from .alpha_volatility import AlphaVolatility
from .alpha_vix import AlphaVIX
from .alpha_er import AlphaER
from .alpha_xma import AlphaXMA
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
from .z_trend_index import ZTrendIndex
from .z_trend_filter import ZTrendFilter
from .z_donchian import ZDonchian
from .z_long_reversal_index import ZLongReversalIndex
from .z_short_reversal_index import ZShortReversalIndex
from .z_bollinger_bands import ZBollingerBands
from .z_v_channel import ZVChannel
from .c_ma import CMA
from .c_atr import CATR

# エーラーズのドミナントサイクル検出クラス
from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from .ehlers_dft_dc import EhlersDFTDC
from .ehlers_hody_dc import EhlersHoDyDC
from .ehlers_phac_dc import EhlersPhAcDC
from .ehlers_dudi_dc import EhlersDuDiDC
from .ehlers_dudi_dce import EhlersDuDiDCE
from .ehlers_hody_dce import EhlersHoDyDCE
from .ehlers_phac_dce import EhlersPhAcDCE
from .ehlers_unified_dc import EhlersUnifiedDC

# エーラーズのドミナントサイクル検出アルゴリズム一覧
__all__ = [
    'Indicator',
    'Supertrend',
    'RSI',
    'ChoppinessIndex',
    'AlphaMA',
    'AlphaMAV2',
    'AlphaATR',
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
    'EhlersDominantCycle',
    'DominantCycleResult',
    'EhlersDFTDC',
    'EhlersHoDyDC',
    'EhlersPhAcDC',
    'EhlersDuDiDC',
    'EhlersDuDiDCE',
    'EhlersHoDyDCE',
    'EhlersPhAcDCE',
    'EhlersUnifiedDC',
    'CMA',
    'CATR',
]

# Version
__version__ = '0.1.0'
