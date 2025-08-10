#!/usr/bin/env python3
"""
Hyper Adaptive Channel Indicator

@indicators/z_adaptive_channel.py をベースに、複数のスムーザーを選択可能な
ミッドライン計算と動的乗数適応を実装したハイパーアダプティブチャネル

Author: Claude
Date: 2025-08-03
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union, List
from numba import njit

from .indicator import Indicator
from .price_source import PriceSource
from .hyper_frama import HyperFRAMA
from .smoother.ultimate_ma import UltimateMA
from .smoother.laguerre_filter import LaguerreFilter
from .smoother.z_adaptive_ma import ZAdaptiveMA
from .smoother.super_smoother import SuperSmoother
from .volatility.x_atr import XATR
from .trend_filter.hyper_er import HyperER


class MidlineSmootherType(Enum):
    """ミッドライン計算用のスムーザータイプ"""
    HYPER_FRAMA = "hyper_frama"
    ULTIMATE_MA = "ultimate_ma"
    LAGUERRE_FILTER = "laguerre_filter"
    Z_ADAPTIVE_MA = "z_adaptive_ma"
    SUPER_SMOOTHER = "super_smoother"


class MultiplierMode(Enum):
    """乗数モード"""
    FIXED = "fixed"
    DYNAMIC = "dynamic"


@dataclass
class HyperAdaptiveChannelResult:
    """ハイパーアダプティブチャネル結果"""
    # チャネル値
    midline: np.ndarray
    upper_band: np.ndarray
    lower_band: np.ndarray
    bandwidth: np.ndarray
    
    # 動的パラメータ
    atr_values: np.ndarray
    multiplier_values: np.ndarray
    er_values: Optional[np.ndarray] = None
    
    # チャネル信号
    channel_position: np.ndarray = None  # -1: lower, 0: inside, 1: upper
    squeeze_signal: np.ndarray = None
    expansion_signal: np.ndarray = None
    
    # 統計情報
    channel_width_percentile: Optional[np.ndarray] = None
    volatility_regime: Optional[np.ndarray] = None


class HyperAdaptiveChannel(Indicator):
    """
    ハイパーアダプティブチャネルインジケーター
    
    複数のスムーザーを選択可能なミッドライン計算と
    HyperERベースの動的乗数適応を実装
    """
    
    def __init__(
        self,
        period: int = 14,
        midline_smoother: str = "hyper_frama",
        multiplier_mode: str = "dynamic",
        fixed_multiplier: float = 2.5,
        src_type: str = "hlc3",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        hyper_frama_period: int = 16,
        hyper_frama_src_type: str = 'hl2',
        hyper_frama_fc: int = 1,
        hyper_frama_sc: int = 198,
        hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        hyper_frama_period_mode: str = 'fixed',
        hyper_frama_cycle_detector_type: str = 'hody_e',
        hyper_frama_lp_period: int = 13,
        hyper_frama_hp_period: int = 124,
        hyper_frama_cycle_part: float = 0.5,
        hyper_frama_max_cycle: int = 89,
        hyper_frama_min_cycle: int = 8,
        hyper_frama_max_output: int = 124,
        hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        hyper_frama_enable_indicator_adaptation: bool = True,
        hyper_frama_adaptation_indicator: str = 'hyper_er',
        hyper_frama_hyper_er_period: int = 14,
        hyper_frama_hyper_er_midline_period: int = 100,
        hyper_frama_hyper_adx_period: int = 14,
        hyper_frama_hyper_adx_midline_period: int = 100,
        hyper_frama_hyper_trend_index_period: int = 14,
        hyper_frama_hyper_trend_index_midline_period: int = 100,
        hyper_frama_fc_min: float = 1.0,
        hyper_frama_fc_max: float = 8.0,
        hyper_frama_sc_min: float = 50.0,
        hyper_frama_sc_max: float = 250.0,
        hyper_frama_period_min: int = 4,
        hyper_frama_period_max: int = 88,
        
        # === UltimateMA パラメータ ===
        ultimate_ma_ultimate_smoother_period: float = 5.0,
        ultimate_ma_zero_lag_period: int = 21,
        ultimate_ma_realtime_window: int = 89,
        ultimate_ma_src_type: str = 'hlc3',  # ukf_hlc3 は問題があるため hlc3 に変更
        ultimate_ma_slope_index: int = 1,
        ultimate_ma_range_threshold: float = 0.005,
        # 適応的カルマンフィルターパラメータ
        ultimate_ma_use_adaptive_kalman: bool = True,
        ultimate_ma_kalman_process_variance: float = 1e-5,
        ultimate_ma_kalman_measurement_variance: float = 0.01,
        ultimate_ma_kalman_volatility_window: int = 5,
        # 動的適応パラメータ
        ultimate_ma_zero_lag_period_mode: str = 'dynamic',
        ultimate_ma_realtime_window_mode: str = 'dynamic',
        # ゼロラグ用サイクル検出器パラメータ
        ultimate_ma_zl_cycle_detector_type: str = 'absolute_ultimate',
        ultimate_ma_zl_cycle_detector_cycle_part: float = 1.0,
        ultimate_ma_zl_cycle_detector_max_cycle: int = 120,
        ultimate_ma_zl_cycle_detector_min_cycle: int = 5,
        ultimate_ma_zl_cycle_period_multiplier: float = 1.0,
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        ultimate_ma_rt_cycle_detector_type: str = 'absolute_ultimate',
        ultimate_ma_rt_cycle_detector_cycle_part: float = 0.5,
        ultimate_ma_rt_cycle_detector_max_cycle: int = 120,
        ultimate_ma_rt_cycle_detector_min_cycle: int = 5,
        ultimate_ma_rt_cycle_period_multiplier: float = 0.5,
        # period_rangeパラメータ
        ultimate_ma_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        ultimate_ma_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        
        # === LaguerreFilter パラメータ ===
        laguerre_gamma: float = 0.5,
        laguerre_order: int = 4,
        laguerre_coefficients: Optional[List[float]] = None,
        laguerre_src_type: str = 'close',
        laguerre_period: int = 4,
        laguerre_period_mode: str = 'fixed',
        laguerre_cycle_detector_type: str = 'hody_e',
        laguerre_cycle_part: float = 0.5,
        laguerre_max_cycle: int = 124,
        laguerre_min_cycle: int = 13,
        laguerre_max_output: int = 124,
        laguerre_min_output: int = 13,
        laguerre_lp_period: int = 13,
        laguerre_hp_period: int = 124,
        
        # === ZAdaptiveMA パラメータ ===
        z_adaptive_fast_period: int = 2,
        z_adaptive_slow_period: int = 120,
        z_adaptive_src_type: str = 'hlc3',
        z_adaptive_slope_index: int = 1,
        z_adaptive_range_threshold: float = 0.005,
        
        # === SuperSmoother パラメータ ===
        super_smoother_length: int = 15,
        super_smoother_num_poles: int = 2,
        super_smoother_src_type: str = 'cc2',
        # 動的期間パラメータ
        super_smoother_period_mode: str = 'fixed',
        super_smoother_cycle_detector_type: str = 'hody_e',
        super_smoother_lp_period: int = 13,
        super_smoother_hp_period: int = 124,
        super_smoother_cycle_part: float = 0.5,
        super_smoother_max_cycle: int = 124,
        super_smoother_min_cycle: int = 13,
        super_smoother_max_output: int = 124,
        super_smoother_min_output: int = 13,
        
        # === X_ATR パラメータ ===
        x_atr_period: float = 12.0,
        x_atr_tr_method: str = 'str',
        x_atr_smoother_type: str = 'frama',
        x_atr_src_type: str = 'close',
        x_atr_enable_kalman: bool = False,
        x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        x_atr_period_mode: str = 'fixed',
        x_atr_cycle_detector_type: str = 'absolute_ultimate',
        x_atr_cycle_detector_cycle_part: float = 0.5,
        x_atr_cycle_detector_max_cycle: int = 55,
        x_atr_cycle_detector_min_cycle: int = 5,
        x_atr_cycle_period_multiplier: float = 1.0,
        x_atr_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # ミッドラインパラメータ
        x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        x_atr_enable_percentile_analysis: bool = True,
        x_atr_percentile_lookback_period: int = 50,
        x_atr_percentile_low_threshold: float = 0.25,
        x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        x_atr_smoother_params: Optional[Dict[str, Any]] = None,
        # カルマンフィルターパラメータ
        x_atr_kalman_params: Optional[Dict[str, Any]] = None,
        
        # === HyperER パラメータ ===
        hyper_er_period: int = 8,
        hyper_er_midline_period: int = 100,
        # ERパラメータ
        hyper_er_er_period: int = 13,
        hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        hyper_er_use_kalman_filter: bool = True,
        hyper_er_kalman_filter_type: str = 'unscented',
        hyper_er_kalman_process_noise: float = 1e-5,
        hyper_er_kalman_min_observation_noise: float = 1e-6,
        hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,
        hyper_er_roofing_hp_cutoff: float = 55.0,
        hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        hyper_er_use_laguerre_filter: bool = False,
        hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        hyper_er_use_smoothing: bool = True,
        hyper_er_smoother_type: str = 'laguerre',
        hyper_er_smoother_period: int = 12,
        hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = False,
        hyper_er_detector_type: str = 'dft_dominant',
        hyper_er_lp_period: int = 13,
        hyper_er_hp_period: int = 124,
        hyper_er_cycle_part: float = 0.4,
        hyper_er_max_cycle: int = 124,
        hyper_er_min_cycle: int = 13,
        hyper_er_max_output: int = 89,
        hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,
        hyper_er_percentile_lookback_period: int = 50,
        hyper_er_percentile_low_threshold: float = 0.25,
        hyper_er_percentile_high_threshold: float = 0.75,
        
        # === ハイパーアダプティブチャネル独自パラメータ ===
        enable_signals: bool = True,
        enable_percentile: bool = True,
        percentile_period: int = 100,
    ):
        """
        Parameters:
        -----------
        period : int
            基本期間
        midline_smoother : str
            ミッドラインスムーザータイプ
        multiplier_mode : str
            乗数モード ("fixed" or "dynamic")
        fixed_multiplier : float
            固定乗数値
        src_type : str
            価格ソースタイプ
        """
        
        super().__init__("HyperAdaptiveChannel")
        
        # 基本パラメータ
        self.period = period
        self.midline_smoother = MidlineSmootherType(midline_smoother)
        self.multiplier_mode = MultiplierMode(multiplier_mode)
        self.fixed_multiplier = fixed_multiplier
        self.src_type = src_type
        
        # 各パラメータの保存
        # HyperFRAMAパラメータ
        self.hyper_frama_params = {
            "period": hyper_frama_period,
            "src_type": hyper_frama_src_type,
            "fc": hyper_frama_fc,
            "sc": hyper_frama_sc,
            "alpha_multiplier": hyper_frama_alpha_multiplier,
            "period_mode": hyper_frama_period_mode,
            "cycle_detector_type": hyper_frama_cycle_detector_type,
            "lp_period": hyper_frama_lp_period,
            "hp_period": hyper_frama_hp_period,
            "cycle_part": hyper_frama_cycle_part,
            "max_cycle": hyper_frama_max_cycle,
            "min_cycle": hyper_frama_min_cycle,
            "max_output": hyper_frama_max_output,
            "min_output": hyper_frama_min_output,
            "enable_indicator_adaptation": hyper_frama_enable_indicator_adaptation,
            "adaptation_indicator": hyper_frama_adaptation_indicator,
            "hyper_er_period": hyper_frama_hyper_er_period,
            "hyper_er_midline_period": hyper_frama_hyper_er_midline_period,
            "hyper_adx_period": hyper_frama_hyper_adx_period,
            "hyper_adx_midline_period": hyper_frama_hyper_adx_midline_period,
            "hyper_trend_index_period": hyper_frama_hyper_trend_index_period,
            "hyper_trend_index_midline_period": hyper_frama_hyper_trend_index_midline_period,
            "fc_min": hyper_frama_fc_min,
            "fc_max": hyper_frama_fc_max,
            "sc_min": hyper_frama_sc_min,
            "sc_max": hyper_frama_sc_max,
            "period_min": hyper_frama_period_min,
            "period_max": hyper_frama_period_max
        }
        
        # UltimateMAパラメータ
        self.ultimate_ma_params = {
            "ultimate_smoother_period": ultimate_ma_ultimate_smoother_period,
            "zero_lag_period": ultimate_ma_zero_lag_period,
            "realtime_window": ultimate_ma_realtime_window,
            "src_type": ultimate_ma_src_type,
            "slope_index": ultimate_ma_slope_index,
            "range_threshold": ultimate_ma_range_threshold,
            "use_adaptive_kalman": ultimate_ma_use_adaptive_kalman,
            "kalman_process_variance": ultimate_ma_kalman_process_variance,
            "kalman_measurement_variance": ultimate_ma_kalman_measurement_variance,
            "kalman_volatility_window": ultimate_ma_kalman_volatility_window,
            "zero_lag_period_mode": ultimate_ma_zero_lag_period_mode,
            "realtime_window_mode": ultimate_ma_realtime_window_mode,
            "zl_cycle_detector_type": ultimate_ma_zl_cycle_detector_type,
            "zl_cycle_detector_cycle_part": ultimate_ma_zl_cycle_detector_cycle_part,
            "zl_cycle_detector_max_cycle": ultimate_ma_zl_cycle_detector_max_cycle,
            "zl_cycle_detector_min_cycle": ultimate_ma_zl_cycle_detector_min_cycle,
            "zl_cycle_period_multiplier": ultimate_ma_zl_cycle_period_multiplier,
            "rt_cycle_detector_type": ultimate_ma_rt_cycle_detector_type,
            "rt_cycle_detector_cycle_part": ultimate_ma_rt_cycle_detector_cycle_part,
            "rt_cycle_detector_max_cycle": ultimate_ma_rt_cycle_detector_max_cycle,
            "rt_cycle_detector_min_cycle": ultimate_ma_rt_cycle_detector_min_cycle,
            "rt_cycle_period_multiplier": ultimate_ma_rt_cycle_period_multiplier,
            "zl_cycle_detector_period_range": ultimate_ma_zl_cycle_detector_period_range,
            "rt_cycle_detector_period_range": ultimate_ma_rt_cycle_detector_period_range
        }
        
        # LaguerreFilterパラメータ
        self.laguerre_params = {
            "gamma": laguerre_gamma,
            "order": laguerre_order,
            "coefficients": laguerre_coefficients,
            "src_type": laguerre_src_type,
            "period": laguerre_period,
            "period_mode": laguerre_period_mode,
            "cycle_detector_type": laguerre_cycle_detector_type,
            "cycle_part": laguerre_cycle_part,
            "max_cycle": laguerre_max_cycle,
            "min_cycle": laguerre_min_cycle,
            "max_output": laguerre_max_output,
            "min_output": laguerre_min_output,
            "lp_period": laguerre_lp_period,
            "hp_period": laguerre_hp_period
        }
        
        # ZAdaptiveMAパラメータ
        self.z_adaptive_params = {
            "fast_period": z_adaptive_fast_period,
            "slow_period": z_adaptive_slow_period,
            "src_type": z_adaptive_src_type,
            "slope_index": z_adaptive_slope_index,
            "range_threshold": z_adaptive_range_threshold
        }
        
        # SuperSmootherパラメータ
        self.super_smoother_params = {
            "length": super_smoother_length,
            "num_poles": super_smoother_num_poles,
            "src_type": super_smoother_src_type,
            "period_mode": super_smoother_period_mode,
            "cycle_detector_type": super_smoother_cycle_detector_type,
            "lp_period": super_smoother_lp_period,
            "hp_period": super_smoother_hp_period,
            "cycle_part": super_smoother_cycle_part,
            "max_cycle": super_smoother_max_cycle,
            "min_cycle": super_smoother_min_cycle,
            "max_output": super_smoother_max_output,
            "min_output": super_smoother_min_output
        }
        
        # X_ATRパラメータ
        self.x_atr_params = {
            "period": x_atr_period,
            "tr_method": x_atr_tr_method,
            "smoother_type": x_atr_smoother_type,
            "src_type": x_atr_src_type,
            "enable_kalman": x_atr_enable_kalman,
            "kalman_type": x_atr_kalman_type,
            "period_mode": x_atr_period_mode,
            "cycle_detector_type": x_atr_cycle_detector_type,
            "cycle_detector_cycle_part": x_atr_cycle_detector_cycle_part,
            "cycle_detector_max_cycle": x_atr_cycle_detector_max_cycle,
            "cycle_detector_min_cycle": x_atr_cycle_detector_min_cycle,
            "cycle_period_multiplier": x_atr_cycle_period_multiplier,
            "cycle_detector_period_range": x_atr_cycle_detector_period_range,
            "midline_period": x_atr_midline_period,
            "enable_percentile_analysis": x_atr_enable_percentile_analysis,
            "percentile_lookback_period": x_atr_percentile_lookback_period,
            "percentile_low_threshold": x_atr_percentile_low_threshold,
            "percentile_high_threshold": x_atr_percentile_high_threshold,
            "smoother_params": x_atr_smoother_params,
            "kalman_params": x_atr_kalman_params
        }
        
        # HyperERパラメータ
        self.hyper_er_params = {
            "period": hyper_er_period,
            "midline_period": hyper_er_midline_period,
            "er_period": hyper_er_er_period,
            "er_src_type": hyper_er_er_src_type,
            "use_kalman_filter": hyper_er_use_kalman_filter,
            "kalman_filter_type": hyper_er_kalman_filter_type,
            "kalman_process_noise": hyper_er_kalman_process_noise,
            "kalman_min_observation_noise": hyper_er_kalman_min_observation_noise,
            "kalman_adaptation_window": hyper_er_kalman_adaptation_window,
            "use_roofing_filter": hyper_er_use_roofing_filter,
            "roofing_hp_cutoff": hyper_er_roofing_hp_cutoff,
            "roofing_ss_band_edge": hyper_er_roofing_ss_band_edge,
            "use_laguerre_filter": hyper_er_use_laguerre_filter,
            "laguerre_gamma": hyper_er_laguerre_gamma,
            "use_smoothing": hyper_er_use_smoothing,
            "smoother_type": hyper_er_smoother_type,
            "smoother_period": hyper_er_smoother_period,
            "smoother_src_type": hyper_er_smoother_src_type,
            "use_dynamic_period": hyper_er_use_dynamic_period,
            "detector_type": hyper_er_detector_type,
            "lp_period": hyper_er_lp_period,
            "hp_period": hyper_er_hp_period,
            "cycle_part": hyper_er_cycle_part,
            "max_cycle": hyper_er_max_cycle,
            "min_cycle": hyper_er_min_cycle,
            "max_output": hyper_er_max_output,
            "min_output": hyper_er_min_output,
            "enable_percentile_analysis": hyper_er_enable_percentile_analysis,
            "percentile_lookback_period": hyper_er_percentile_lookback_period,
            "percentile_low_threshold": hyper_er_percentile_low_threshold,
            "percentile_high_threshold": hyper_er_percentile_high_threshold
        }
        
        # その他パラメータ
        self.enable_signals = enable_signals
        self.enable_percentile = enable_percentile
        self.percentile_period = percentile_period
        
        # インジケーターインスタンス
        self._setup_indicators()
        
        # キャッシュ
        self._cache = {}
    
    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """データを準備・正規化"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if data.shape[1] >= 5:  # OHLCV
                return pd.DataFrame({
                    'open': data[:, 0],
                    'high': data[:, 1], 
                    'low': data[:, 2],
                    'close': data[:, 3],
                    'volume': data[:, 4]
                })
            else:
                raise ValueError("NumPy配列はOHLCV形式である必要があります")
        else:
            raise ValueError("サポートされていないデータ型です")
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """キャッシュキーを生成"""
        return f"{self.name}_{len(data)}_{hash(str(data.iloc[0]) + str(data.iloc[-1]))}"
        
    def _setup_indicators(self):
        """必要なインジケーターインスタンスを初期化"""
        
        # ミッドラインスムーザー
        if self.midline_smoother == MidlineSmootherType.HYPER_FRAMA:
            self.midline_indicator = HyperFRAMA(**self.hyper_frama_params)
        elif self.midline_smoother == MidlineSmootherType.ULTIMATE_MA:
            self.midline_indicator = UltimateMA(**self.ultimate_ma_params)
        elif self.midline_smoother == MidlineSmootherType.LAGUERRE_FILTER:
            self.midline_indicator = LaguerreFilter(**self.laguerre_params)
        elif self.midline_smoother == MidlineSmootherType.Z_ADAPTIVE_MA:
            self.midline_indicator = ZAdaptiveMA(**self.z_adaptive_params)
        elif self.midline_smoother == MidlineSmootherType.SUPER_SMOOTHER:
            self.midline_indicator = SuperSmoother(**self.super_smoother_params)
        
        # X_ATR
        self.atr_indicator = XATR(**self.x_atr_params)
        
        # HyperER (動的乗数モード時のみ)
        if self.multiplier_mode == MultiplierMode.DYNAMIC:
            self.er_indicator = HyperER(**self.hyper_er_params)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperAdaptiveChannelResult:
        """
        ハイパーアダプティブチャネルを計算
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            OHLCV市場データ
            
        Returns:
        --------
        HyperAdaptiveChannelResult
            計算結果
        """
        
        try:
            # データ前処理
            df = self._prepare_data(data)
            cache_key = self._generate_cache_key(df)
            
            # キャッシュチェック
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # 価格ソース計算
            price_source = PriceSource.calculate_source(df, self.src_type)
            
            # ミッドライン計算
            midline = self._calculate_midline(df, price_source)
            
            # ATR計算
            atr_result = self.atr_indicator.calculate(df)
            atr_values = atr_result.values
            
            # 動的乗数計算
            multiplier_values, er_values = self._calculate_dynamic_multiplier(df, price_source)
            
            # チャネルバンド計算
            upper_band, lower_band, bandwidth = self._calculate_channel_bands(
                midline, atr_values, multiplier_values
            )
            
            # シグナル計算
            channel_position = None
            squeeze_signal = None
            expansion_signal = None
            channel_width_percentile = None
            volatility_regime = None
            
            if self.enable_signals:
                channel_position = self._calculate_channel_position(price_source, upper_band, lower_band)
                squeeze_signal, expansion_signal = self._calculate_squeeze_expansion_signals(bandwidth)
                
            if self.enable_percentile:
                channel_width_percentile = self._calculate_percentile_analysis(bandwidth)
                volatility_regime = self._calculate_volatility_regime(atr_values)
            
            # 結果構築
            result = HyperAdaptiveChannelResult(
                midline=midline,
                upper_band=upper_band,
                lower_band=lower_band,
                bandwidth=bandwidth,
                atr_values=atr_values,
                multiplier_values=multiplier_values,
                er_values=er_values,
                channel_position=channel_position,
                squeeze_signal=squeeze_signal,
                expansion_signal=expansion_signal,
                channel_width_percentile=channel_width_percentile,
                volatility_regime=volatility_regime
            )
            
            # キャッシュ保存
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"HyperAdaptiveChannel計算エラー: {e}")
            raise
    
    def _calculate_midline(self, data: pd.DataFrame, price_source: np.ndarray) -> np.ndarray:
        """ミッドライン計算"""
        
        if self.midline_smoother == MidlineSmootherType.HYPER_FRAMA:
            result = self.midline_indicator.calculate(data)
            return result.frama_values  # 通常のFRAMA値を使用
            
        elif self.midline_smoother == MidlineSmootherType.ULTIMATE_MA:
            result = self.midline_indicator.calculate(data)
            return result.values
            
        elif self.midline_smoother == MidlineSmootherType.LAGUERRE_FILTER:
            result = self.midline_indicator.calculate(data)
            return result.values
            
        elif self.midline_smoother == MidlineSmootherType.Z_ADAPTIVE_MA:
            result = self.midline_indicator.calculate(data)
            return result.values
            
        elif self.midline_smoother == MidlineSmootherType.SUPER_SMOOTHER:
            result = self.midline_indicator.calculate(data)
            return result.values
    
    def _calculate_dynamic_multiplier(
        self, 
        data: pd.DataFrame, 
        price_source: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """動的乗数計算"""
        
        if self.multiplier_mode == MultiplierMode.FIXED:
            # 固定乗数
            multiplier_values = np.full(len(data), self.fixed_multiplier)
            return multiplier_values, None
        
        else:
            # 動的乗数 (HyperER使用)
            er_result = self.er_indicator.calculate(data)
            er_values = er_result.values
            
            # ERを乗数に変換 (0.5-3.0の範囲)
            multiplier_values = self._convert_er_to_multiplier(er_values)
            
            return multiplier_values, er_values
    
    @staticmethod
    @njit
    def _convert_er_to_multiplier(er_values: np.ndarray) -> np.ndarray:
        """効率比を乗数に変換"""
        
        multiplier_values = np.full(len(er_values), 2.5)
        
        for i in range(len(er_values)):
            if not np.isnan(er_values[i]):
                # ERが高い時は乗数を小さく (0.5-1.5)
                # ERが低い時は乗数を大きく (2.0-3.0)
                if er_values[i] > 0.6:
                    multiplier_values[i] = 0.5 + (1.0 - er_values[i]) * 2.0
                else:
                    multiplier_values[i] = 2.0 + (1.0 - er_values[i]) * 1.0
                    
                # 範囲制限
                multiplier_values[i] = max(0.5, min(3.0, multiplier_values[i]))
        
        return multiplier_values
    
    @staticmethod
    @njit
    def _calculate_channel_bands(
        midline: np.ndarray,
        atr_values: np.ndarray, 
        multiplier_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """チャネルバンド計算"""
        
        upper_band = np.full(len(midline), np.nan)
        lower_band = np.full(len(midline), np.nan)
        bandwidth = np.full(len(midline), np.nan)
        
        for i in range(len(midline)):
            if not (np.isnan(midline[i]) or np.isnan(atr_values[i]) or np.isnan(multiplier_values[i])):
                band_width = atr_values[i] * multiplier_values[i]
                upper_band[i] = midline[i] + band_width
                lower_band[i] = midline[i] - band_width
                bandwidth[i] = band_width * 2  # 全幅
        
        return upper_band, lower_band, bandwidth
    
    @staticmethod
    @njit
    def _calculate_channel_position(
        price_source: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """チャネル内ポジション計算"""
        
        position = np.full(len(price_source), 0.0)
        
        for i in range(len(price_source)):
            if not (np.isnan(price_source[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
                if price_source[i] >= upper_band[i]:
                    position[i] = 1.0  # 上バンドブレイク
                elif price_source[i] <= lower_band[i]:
                    position[i] = -1.0  # 下バンドブレイク
                else:
                    position[i] = 0.0  # チャネル内
        
        return position
    
    def _calculate_squeeze_expansion_signals(
        self, 
        bandwidth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """スクイーズ・エクスパンション信号計算"""
        
        if len(bandwidth) < self.period:
            return np.full(len(bandwidth), 0.0), np.full(len(bandwidth), 0.0)
        
        # バンド幅の移動平均とボラティリティ
        bandwidth_sma = pd.Series(bandwidth).rolling(self.period, min_periods=1).mean().values
        bandwidth_std = pd.Series(bandwidth).rolling(self.period, min_periods=1).std().values
        
        squeeze_signal = np.full(len(bandwidth), 0.0)
        expansion_signal = np.full(len(bandwidth), 0.0)
        
        for i in range(self.period, len(bandwidth)):
            if not (np.isnan(bandwidth[i]) or np.isnan(bandwidth_sma[i]) or np.isnan(bandwidth_std[i])):
                # スクイーズ: バンド幅が平均 - 1σ以下
                squeeze_threshold = bandwidth_sma[i] - bandwidth_std[i]
                if bandwidth[i] <= squeeze_threshold:
                    squeeze_signal[i] = 1.0
                
                # エクスパンション: バンド幅が平均 + 1σ以上
                expansion_threshold = bandwidth_sma[i] + bandwidth_std[i]
                if bandwidth[i] >= expansion_threshold:
                    expansion_signal[i] = 1.0
        
        return squeeze_signal, expansion_signal
    
    def _calculate_percentile_analysis(self, bandwidth: np.ndarray) -> np.ndarray:
        """パーセンタイル分析"""
        
        if len(bandwidth) < self.percentile_period:
            return np.full(len(bandwidth), 50.0)
        
        percentile_values = np.full(len(bandwidth), np.nan)
        
        for i in range(self.percentile_period - 1, len(bandwidth)):
            if not np.isnan(bandwidth[i]):
                window = bandwidth[i - self.percentile_period + 1:i + 1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    current_percentile = (np.sum(valid_window <= bandwidth[i]) / len(valid_window)) * 100
                    percentile_values[i] = current_percentile
        
        return percentile_values
    
    def _calculate_volatility_regime(self, atr_values: np.ndarray) -> np.ndarray:
        """ボラティリティレジーム分析"""
        
        if len(atr_values) < self.percentile_period:
            return np.full(len(atr_values), 1.0)  # 通常レジーム
        
        regime = np.full(len(atr_values), 1.0)
        
        for i in range(self.percentile_period - 1, len(atr_values)):
            if not np.isnan(atr_values[i]):
                window = atr_values[i - self.percentile_period + 1:i + 1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    current_percentile = (np.sum(valid_window <= atr_values[i]) / len(valid_window)) * 100
                    
                    if current_percentile >= 80:
                        regime[i] = 2.0  # 高ボラティリティ
                    elif current_percentile <= 20:
                        regime[i] = 0.0  # 低ボラティリティ
                    else:
                        regime[i] = 1.0  # 通常ボラティリティ
        
        return regime


# 便利関数
def create_hyper_adaptive_channel(
    period: int = 14,
    midline_smoother: str = "hyper_frama",
    multiplier_mode: str = "dynamic",
    **kwargs
) -> HyperAdaptiveChannel:
    """ハイパーアダプティブチャネル作成のヘルパー関数"""
    
    return HyperAdaptiveChannel(
        period=period,
        midline_smoother=midline_smoother,
        multiplier_mode=multiplier_mode,
        **kwargs
    )