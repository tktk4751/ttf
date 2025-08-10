#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int8, boolean, int64, optional

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_adaptive_channel import HyperAdaptiveChannel


@njit(int8[:](float64[:], float64[:], float64[:], int64), fastmath=True, parallel=True, cache=True)
def calculate_hyper_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ハイパーアダプティブチャネルのブレイクアウトシグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定（並列処理化）
    for i in prange(lookback + 1, length):
        # 終値とバンドの値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 前回の終値が前回のアッパーバンドを上回っていないかつ現在の終値が現在のアッパーバンドを上回る
        if close[i-1] <= upper[i-1] and close[i] > upper[i]:
            signals[i] = 1
        # ショートエントリー: 前回の終値が前回のロワーバンドを下回っていないかつ現在の終値が現在のロワーバンドを下回る
        elif close[i-1] >= lower[i-1] and close[i] < lower[i]:
            signals[i] = -1
        # 前回のチェックを追加（より多くのシグナルを生成）- 近似クロスオーバー
        elif lookback > 0 and i > lookback:
            # 直近の近似クロスオーバーもチェック
            if close[i] > close[i-1] and close[i-1] <= upper[i-1] and close[i] >= upper[i] * 0.995 and close[i-1] < upper[i-1] * 0.995:
                signals[i] = 1  # ほぼアッパーバンドでクロスオーバー
            elif close[i] < close[i-1] and close[i-1] >= lower[i-1] and close[i] <= lower[i] * 1.005 and close[i-1] > lower[i-1] * 1.005:
                signals[i] = -1  # ほぼロワーバンドでクロスオーバー
    
    return signals


@njit(fastmath=True, cache=True)
def extract_ohlc_from_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy配列からOHLC価格データを抽出する（高速化版）
    
    Args:
        data: 価格データの配列（OHLCフォーマット）
        
    Returns:
        open, high, low, closeの値をそれぞれ含むタプル
    """
    if data.ndim == 1:
        # 1次元配列の場合はすべて同じ値とみなす
        return data, data, data, data
    else:
        # 2次元配列の場合はOHLCとして抽出
        if data.shape[1] >= 4:
            return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        elif data.shape[1] == 1:
            # 1列のみの場合はすべて同じ値とみなす
            return data[:, 0], data[:, 0], data[:, 0], data[:, 0]
        else:
            # 列数が不足している場合は終値のみ使用
            raise ValueError(f"データの列数が不足しています: 必要=4, 実際={data.shape[1]}")


class HyperAdaptiveChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ハイパーアダプティブチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - 複数のスムーザー（HyperFRAMA、UltimateMA、LaguerreFilter、ZAdaptiveMA、SuperSmoother）対応
    - X_ATRベースの動的ボラティリティ測定
    - HyperERによる動的乗数適応
    - 全141パラメーターのカスタマイズ対応
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
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
        ultimate_ma_src_type: str = 'hlc3',
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
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            period: 基本期間
            midline_smoother: ミッドラインスムーザータイプ
            multiplier_mode: 乗数モード ("fixed" or "dynamic")
            fixed_multiplier: 固定乗数値
            src_type: 価格ソースタイプ
        """
        
        # チャネル検出器タイプとパラメータ値を取得（シグナル名用）
        smoother_name = midline_smoother
        mode_name = multiplier_mode
        
        super().__init__(
            f"HyperAdaptiveChannelBreakoutEntrySignal({smoother_name}, {mode_name}, {fixed_multiplier}, {band_lookback})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'band_lookback': band_lookback,
            'period': period,
            'midline_smoother': midline_smoother,
            'multiplier_mode': multiplier_mode,
            'fixed_multiplier': fixed_multiplier,
            'src_type': src_type,
            
            # HyperFRAMAパラメータ
            'hyper_frama_period': hyper_frama_period,
            'hyper_frama_src_type': hyper_frama_src_type,
            'hyper_frama_fc': hyper_frama_fc,
            'hyper_frama_sc': hyper_frama_sc,
            'hyper_frama_alpha_multiplier': hyper_frama_alpha_multiplier,
            'hyper_frama_period_mode': hyper_frama_period_mode,
            'hyper_frama_cycle_detector_type': hyper_frama_cycle_detector_type,
            'hyper_frama_lp_period': hyper_frama_lp_period,
            'hyper_frama_hp_period': hyper_frama_hp_period,
            'hyper_frama_cycle_part': hyper_frama_cycle_part,
            'hyper_frama_max_cycle': hyper_frama_max_cycle,
            'hyper_frama_min_cycle': hyper_frama_min_cycle,
            'hyper_frama_max_output': hyper_frama_max_output,
            'hyper_frama_min_output': hyper_frama_min_output,
            'hyper_frama_enable_indicator_adaptation': hyper_frama_enable_indicator_adaptation,
            'hyper_frama_adaptation_indicator': hyper_frama_adaptation_indicator,
            'hyper_frama_hyper_er_period': hyper_frama_hyper_er_period,
            'hyper_frama_hyper_er_midline_period': hyper_frama_hyper_er_midline_period,
            'hyper_frama_hyper_adx_period': hyper_frama_hyper_adx_period,
            'hyper_frama_hyper_adx_midline_period': hyper_frama_hyper_adx_midline_period,
            'hyper_frama_hyper_trend_index_period': hyper_frama_hyper_trend_index_period,
            'hyper_frama_hyper_trend_index_midline_period': hyper_frama_hyper_trend_index_midline_period,
            'hyper_frama_fc_min': hyper_frama_fc_min,
            'hyper_frama_fc_max': hyper_frama_fc_max,
            'hyper_frama_sc_min': hyper_frama_sc_min,
            'hyper_frama_sc_max': hyper_frama_sc_max,
            'hyper_frama_period_min': hyper_frama_period_min,
            'hyper_frama_period_max': hyper_frama_period_max,
            
            # UltimateMAパラメータ
            'ultimate_ma_ultimate_smoother_period': ultimate_ma_ultimate_smoother_period,
            'ultimate_ma_zero_lag_period': ultimate_ma_zero_lag_period,
            'ultimate_ma_realtime_window': ultimate_ma_realtime_window,
            'ultimate_ma_src_type': ultimate_ma_src_type,
            'ultimate_ma_slope_index': ultimate_ma_slope_index,
            'ultimate_ma_range_threshold': ultimate_ma_range_threshold,
            'ultimate_ma_use_adaptive_kalman': ultimate_ma_use_adaptive_kalman,
            'ultimate_ma_kalman_process_variance': ultimate_ma_kalman_process_variance,
            'ultimate_ma_kalman_measurement_variance': ultimate_ma_kalman_measurement_variance,
            'ultimate_ma_kalman_volatility_window': ultimate_ma_kalman_volatility_window,
            'ultimate_ma_zero_lag_period_mode': ultimate_ma_zero_lag_period_mode,
            'ultimate_ma_realtime_window_mode': ultimate_ma_realtime_window_mode,
            'ultimate_ma_zl_cycle_detector_type': ultimate_ma_zl_cycle_detector_type,
            'ultimate_ma_zl_cycle_detector_cycle_part': ultimate_ma_zl_cycle_detector_cycle_part,
            'ultimate_ma_zl_cycle_detector_max_cycle': ultimate_ma_zl_cycle_detector_max_cycle,
            'ultimate_ma_zl_cycle_detector_min_cycle': ultimate_ma_zl_cycle_detector_min_cycle,
            'ultimate_ma_zl_cycle_period_multiplier': ultimate_ma_zl_cycle_period_multiplier,
            'ultimate_ma_rt_cycle_detector_type': ultimate_ma_rt_cycle_detector_type,
            'ultimate_ma_rt_cycle_detector_cycle_part': ultimate_ma_rt_cycle_detector_cycle_part,
            'ultimate_ma_rt_cycle_detector_max_cycle': ultimate_ma_rt_cycle_detector_max_cycle,
            'ultimate_ma_rt_cycle_detector_min_cycle': ultimate_ma_rt_cycle_detector_min_cycle,
            'ultimate_ma_rt_cycle_period_multiplier': ultimate_ma_rt_cycle_period_multiplier,
            'ultimate_ma_zl_cycle_detector_period_range': ultimate_ma_zl_cycle_detector_period_range,
            'ultimate_ma_rt_cycle_detector_period_range': ultimate_ma_rt_cycle_detector_period_range,
            
            # LaguerreFilterパラメータ
            'laguerre_gamma': laguerre_gamma,
            'laguerre_order': laguerre_order,
            'laguerre_coefficients': laguerre_coefficients,
            'laguerre_src_type': laguerre_src_type,
            'laguerre_period': laguerre_period,
            'laguerre_period_mode': laguerre_period_mode,
            'laguerre_cycle_detector_type': laguerre_cycle_detector_type,
            'laguerre_cycle_part': laguerre_cycle_part,
            'laguerre_max_cycle': laguerre_max_cycle,
            'laguerre_min_cycle': laguerre_min_cycle,
            'laguerre_max_output': laguerre_max_output,
            'laguerre_min_output': laguerre_min_output,
            'laguerre_lp_period': laguerre_lp_period,
            'laguerre_hp_period': laguerre_hp_period,
            
            # ZAdaptiveMAパラメータ
            'z_adaptive_fast_period': z_adaptive_fast_period,
            'z_adaptive_slow_period': z_adaptive_slow_period,
            'z_adaptive_src_type': z_adaptive_src_type,
            'z_adaptive_slope_index': z_adaptive_slope_index,
            'z_adaptive_range_threshold': z_adaptive_range_threshold,
            
            # SuperSmootherパラメータ
            'super_smoother_length': super_smoother_length,
            'super_smoother_num_poles': super_smoother_num_poles,
            'super_smoother_src_type': super_smoother_src_type,
            'super_smoother_period_mode': super_smoother_period_mode,
            'super_smoother_cycle_detector_type': super_smoother_cycle_detector_type,
            'super_smoother_lp_period': super_smoother_lp_period,
            'super_smoother_hp_period': super_smoother_hp_period,
            'super_smoother_cycle_part': super_smoother_cycle_part,
            'super_smoother_max_cycle': super_smoother_max_cycle,
            'super_smoother_min_cycle': super_smoother_min_cycle,
            'super_smoother_max_output': super_smoother_max_output,
            'super_smoother_min_output': super_smoother_min_output,
            
            # X_ATRパラメータ
            'x_atr_period': x_atr_period,
            'x_atr_tr_method': x_atr_tr_method,
            'x_atr_smoother_type': x_atr_smoother_type,
            'x_atr_src_type': x_atr_src_type,
            'x_atr_enable_kalman': x_atr_enable_kalman,
            'x_atr_kalman_type': x_atr_kalman_type,
            'x_atr_period_mode': x_atr_period_mode,
            'x_atr_cycle_detector_type': x_atr_cycle_detector_type,
            'x_atr_cycle_detector_cycle_part': x_atr_cycle_detector_cycle_part,
            'x_atr_cycle_detector_max_cycle': x_atr_cycle_detector_max_cycle,
            'x_atr_cycle_detector_min_cycle': x_atr_cycle_detector_min_cycle,
            'x_atr_cycle_period_multiplier': x_atr_cycle_period_multiplier,
            'x_atr_cycle_detector_period_range': x_atr_cycle_detector_period_range,
            'x_atr_midline_period': x_atr_midline_period,
            'x_atr_enable_percentile_analysis': x_atr_enable_percentile_analysis,
            'x_atr_percentile_lookback_period': x_atr_percentile_lookback_period,
            'x_atr_percentile_low_threshold': x_atr_percentile_low_threshold,
            'x_atr_percentile_high_threshold': x_atr_percentile_high_threshold,
            'x_atr_smoother_params': x_atr_smoother_params,
            'x_atr_kalman_params': x_atr_kalman_params,
            
            # HyperERパラメータ
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'hyper_er_er_period': hyper_er_er_period,
            'hyper_er_er_src_type': hyper_er_er_src_type,
            'hyper_er_use_kalman_filter': hyper_er_use_kalman_filter,
            'hyper_er_kalman_filter_type': hyper_er_kalman_filter_type,
            'hyper_er_kalman_process_noise': hyper_er_kalman_process_noise,
            'hyper_er_kalman_min_observation_noise': hyper_er_kalman_min_observation_noise,
            'hyper_er_kalman_adaptation_window': hyper_er_kalman_adaptation_window,
            'hyper_er_use_roofing_filter': hyper_er_use_roofing_filter,
            'hyper_er_roofing_hp_cutoff': hyper_er_roofing_hp_cutoff,
            'hyper_er_roofing_ss_band_edge': hyper_er_roofing_ss_band_edge,
            'hyper_er_use_laguerre_filter': hyper_er_use_laguerre_filter,
            'hyper_er_laguerre_gamma': hyper_er_laguerre_gamma,
            'hyper_er_use_smoothing': hyper_er_use_smoothing,
            'hyper_er_smoother_type': hyper_er_smoother_type,
            'hyper_er_smoother_period': hyper_er_smoother_period,
            'hyper_er_smoother_src_type': hyper_er_smoother_src_type,
            'hyper_er_use_dynamic_period': hyper_er_use_dynamic_period,
            'hyper_er_detector_type': hyper_er_detector_type,
            'hyper_er_lp_period': hyper_er_lp_period,
            'hyper_er_hp_period': hyper_er_hp_period,
            'hyper_er_cycle_part': hyper_er_cycle_part,
            'hyper_er_max_cycle': hyper_er_max_cycle,
            'hyper_er_min_cycle': hyper_er_min_cycle,
            'hyper_er_max_output': hyper_er_max_output,
            'hyper_er_min_output': hyper_er_min_output,
            'hyper_er_enable_percentile_analysis': hyper_er_enable_percentile_analysis,
            'hyper_er_percentile_lookback_period': hyper_er_percentile_lookback_period,
            'hyper_er_percentile_low_threshold': hyper_er_percentile_low_threshold,
            'hyper_er_percentile_high_threshold': hyper_er_percentile_high_threshold,
            
            # ハイパーアダプティブチャネル独自パラメータ
            'enable_signals': enable_signals,
            'enable_percentile': enable_percentile,
            'percentile_period': percentile_period
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # ハイパーアダプティブチャネルの初期化（すべてのパラメータを渡す）
        channel_params = {k: v for k, v in self._params.items() if k != 'band_lookback'}
        self.hyper_adaptive_channel = HyperAdaptiveChannel(**channel_params)
        
        # 参照期間の設定
        self.band_lookback = band_lookback
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算したバンド値のキャッシュ
        self._last_result = None
        self._last_data_hash = None
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する（超高速化版）
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # 超高速化: 最小限のデータサンプリング
        try:
            if isinstance(ohlcv_data, pd.DataFrame):
                length = len(ohlcv_data)
                if length > 0:
                    first_close = float(ohlcv_data.iloc[0].get('close', ohlcv_data.iloc[0, -1]))
                    last_close = float(ohlcv_data.iloc[-1].get('close', ohlcv_data.iloc[-1, -1]))
                    data_signature = (length, first_close, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                # NumPy配列の場合
                length = len(ohlcv_data)
                if length > 0:
                    if ohlcv_data.ndim > 1:
                        first_val = float(ohlcv_data[0, -1])  # 最後の列（通常close）
                        last_val = float(ohlcv_data[-1, -1])
                    else:
                        first_val = float(ohlcv_data[0])
                        last_val = float(ohlcv_data[-1])
                    data_signature = (length, first_val, last_val)
                else:
                    data_signature = (0, 0.0, 0.0)
            
            # データハッシュの計算（事前計算済みのパラメータハッシュを使用）
            return hash((self._params_hash, hash(data_signature)))
            
        except Exception:
            # フォールバック: 最小限のハッシュ
            return hash((self._params_hash, id(ohlcv_data)))
    
    def _extract_close(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        データから終値を効率的に抽出する（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 終値の配列
        """
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                return data['close'].values
            else:
                raise ValueError("データには'close'カラムが必要です")
        else:
            # NumPy配列
            if data.ndim == 1:
                return data  # 1次元配列はそのまま終値として扱う
            elif data.shape[1] >= 4:
                return data[:, 3]  # 4列以上ある場合は4列目を終値として扱う
            elif data.shape[1] == 1:
                return data[:, 0]  # 1列のみの場合はその列を終値として扱う
            else:
                raise ValueError(f"データの列数が不足しています: 必要=4, 実際={data.shape[1]}")
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する（高速化版）
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # データの長さをチェック
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                data_len = data.shape[0]
            
            if data_len <= self.band_lookback + 1:
                # データが少なすぎる場合はゼロシグナルを返す
                return np.zeros(data_len, dtype=np.int8)
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                # キャッシュヒット - キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._signals_cache[data_hash]
            
            # 終値を取得
            close = self._extract_close(data)
            
            # チャネル結果がキャッシュされている場合はスキップ
            if data_hash == self._last_data_hash and self._last_result is not None:
                result = self._last_result
            else:
                # ハイパーアダプティブチャネルの計算
                result = self.hyper_adaptive_channel.calculate(data)
                
                # 計算が失敗した場合はゼロシグナルを返す
                if result is None:
                    signals = np.zeros(data_len, dtype=np.int8)
                    
                    # キャッシュサイズ管理
                    if len(self._signals_cache) >= self._max_cache_size and self._cache_keys:
                        oldest_key = self._cache_keys.pop(0)
                        if oldest_key in self._signals_cache:
                            del self._signals_cache[oldest_key]
                    
                    # 結果をキャッシュ
                    self._signals_cache[data_hash] = signals
                    self._cache_keys.append(data_hash)
                    
                    return signals
                
                # 結果をキャッシュ
                self._last_result = result
                self._last_data_hash = data_hash
            
            # ブレイクアウトシグナルの計算（高速化版）
            signals = calculate_hyper_breakout_signals(
                close,
                result.upper_band,
                result.lower_band,
                self.band_lookback
            )
            
            # キャッシュサイズ管理
            if len(self._signals_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._signals_cache:
                    del self._signals_cache[oldest_key]
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            self._cache_keys.append(data_hash)
            
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperAdaptiveChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.int8)
            else:
                return np.array([], dtype=np.int8)
    
    def get_channel_result(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        ハイパーアダプティブチャネルの結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            HyperAdaptiveChannelResult: チャネル計算結果
        """
        if data is not None:
            data_hash = self._get_data_hash(data)
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_result is None:
                # 異なる場合は再計算が必要
                self.generate(data)
            # 結果はgenerate内でキャッシュされる
        
        return self._last_result
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ハイパーアダプティブチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        result = self.get_channel_result(data)
        if result is not None:
            return result.midline, result.upper_band, result.lower_band
        else:
            # エラー時は空の配列を返す
            empty_array = np.array([])
            return empty_array, empty_array, empty_array
    
    def get_atr_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_ATRの値
        """
        result = self.get_channel_result(data)
        if result is not None:
            return result.atr_values
        else:
            return np.array([])
    
    def get_multiplier_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        result = self.get_channel_result(data)
        if result is not None:
            return result.multiplier_values
        else:
            return np.array([])
    
    def get_er_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[np.ndarray]:
        """
        効率比の値を取得する（動的乗数モード時のみ）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Optional[np.ndarray]: 効率比の値
        """
        result = self.get_channel_result(data)
        if result is not None:
            return result.er_values
        else:
            return None
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self.hyper_adaptive_channel, 'reset'):
            self.hyper_adaptive_channel.reset()
        
        # キャッシュをクリア
        self._signals_cache = {}
        self._cache_keys = []
        
        # 結果のキャッシュもクリア
        self._last_result = None
        self._last_data_hash = None