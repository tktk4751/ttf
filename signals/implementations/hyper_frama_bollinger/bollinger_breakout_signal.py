#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperFRAMAボリンジャーバンドブレイクアウトシグナル

HyperFRAMAボリンジャーバンドのブレイクアウト・リバーサルシグナルを生成
"""

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, int8, int64, float64, boolean, optional

from signals.interfaces.entry import IEntrySignal
from signals.interfaces.exit import IExitSignal
from indicators.hyper_frama_bollinger import HyperFRAMABollinger
from indicators.price_source import PriceSource


@njit(int8[:](float64[:], float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def calculate_bollinger_breakout_entry_signals(
    close: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray,
    percent_b: np.ndarray,
    lookback: int
) -> np.ndarray:
    """
    HyperFRAMAボリンジャーバンド ブレイクアウトエントリーシグナルを計算
    
    Args:
        close: 終値の配列
        upper: HyperFRAMAボリンジャー上限バンドの配列
        lower: HyperFRAMAボリンジャー下限バンドの配列
        percent_b: パーセントBの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: エントリーなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定
    for i in prange(lookback + 1, length):
        # すべての値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1]) or
            np.isnan(percent_b[i])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 上バンドブレイクアウト
        if close[i-1] <= upper[i-1] and close[i] > upper[i]:
            # パーセントBが1.0を上回った場合の確認エントリー
            if percent_b[i] > 1.0:
                signals[i] = 1
        # ショートエントリー: 下バンドブレイクアウト
        elif close[i-1] >= lower[i-1] and close[i] < lower[i]:
            # パーセントBが0.0を下回った場合の確認エントリー
            if percent_b[i] < 0.0:
                signals[i] = -1
        # 近似ブレイクアウト検出（より敏感なシグナル）
        elif lookback > 0 and i > lookback:
            # ロング近似ブレイクアウト
            if (close[i] > close[i-1] and 
                close[i-1] <= upper[i-1] and 
                close[i] >= upper[i] * 0.998 and 
                close[i-1] < upper[i-1] * 0.998 and
                percent_b[i] >= 0.95):
                signals[i] = 1
            # ショート近似ブレイクアウト
            elif (close[i] < close[i-1] and 
                  close[i-1] >= lower[i-1] and 
                  close[i] <= lower[i] * 1.002 and 
                  close[i-1] > lower[i-1] * 1.002 and
                  percent_b[i] <= 0.05):
                signals[i] = -1
    
    return signals


@njit(int8[:](float64[:], float64[:], float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def calculate_bollinger_reversal_entry_signals(
    close: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray,
    midline: np.ndarray,
    percent_b: np.ndarray,
    lookback: int
) -> np.ndarray:
    """
    HyperFRAMAボリンジャーバンド リバーサルエントリーシグナルを計算
    
    Args:
        close: 終値の配列
        upper: HyperFRAMAボリンジャー上限バンドの配列
        lower: HyperFRAMAボリンジャー下限バンドの配列
        midline: HyperFRAMAミッドラインの配列
        percent_b: パーセントBの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: エントリーなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(lookback + 1, length):
        # すべての値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(midline[i]) or np.isnan(percent_b[i])):
            signals[i] = 0
            continue
        
        # ロングリバーサルエントリー: 下バンド付近からの反発
        # 条件: パーセントBが0.2以下から0.3以上に上昇
        if (percent_b[i-1] <= 0.2 and percent_b[i] >= 0.3 and
            close[i] > close[i-1] and
            close[i-1] <= lower[i-1] * 1.02):
            signals[i] = 1
            
        # ショートリバーサルエントリー: 上バンド付近からの反落
        # 条件: パーセントBが0.8以上から0.7以下に下落
        elif (percent_b[i-1] >= 0.8 and percent_b[i] <= 0.7 and
              close[i] < close[i-1] and
              close[i-1] >= upper[i-1] * 0.98):
            signals[i] = -1
    
    return signals


@njit(int8[:](float64[:], float64[:], float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def calculate_bollinger_exit_signals(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    midline: np.ndarray,
    percent_b: np.ndarray,
    exit_mode: int
) -> np.ndarray:
    """
    HyperFRAMAボリンジャーバンド エグジットシグナルを計算
    
    Args:
        close: 終値の配列
        upper: HyperFRAMAボリンジャー上限バンドの配列
        lower: HyperFRAMAボリンジャー下限バンドの配列
        midline: HyperFRAMAミッドラインの配列
        percent_b: パーセントBの配列
        exit_mode: エグジットモード (1: 逆ブレイクアウト, 2: ミッドラインクロス, 3: パーセントB反転)
    
    Returns:
        エグジットシグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: エグジットなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(1, length):
        # すべての値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(midline[i]) or np.isnan(percent_b[i])):
            signals[i] = 0
            continue
        
        if exit_mode == 1:
            # モード1: 逆ブレイクアウトによるエグジット
            # ロングエグジット: 価格が下限バンドを下抜けた場合
            if close[i-1] >= lower[i-1] and close[i] < lower[i]:
                signals[i] = 1
            # ショートエグジット: 価格が上限バンドを上抜けた場合
            elif close[i-1] <= upper[i-1] and close[i] > upper[i]:
                signals[i] = -1
        
        elif exit_mode == 2:
            # モード2: ミッドラインクロスによるエグジット
            # ロングエグジット: 価格がミッドラインを下抜けた場合
            if close[i-1] >= midline[i-1] and close[i] < midline[i]:
                signals[i] = 1
            # ショートエグジット: 価格がミッドラインを上抜けた場合
            elif close[i-1] <= midline[i-1] and close[i] > midline[i]:
                signals[i] = -1
        
        elif exit_mode == 3:
            # モード3: パーセントB反転によるエグジット
            # ロングエグジット: パーセントBが0.8以上から0.6以下に下落
            if percent_b[i-1] >= 0.8 and percent_b[i] <= 0.6:
                signals[i] = 1
            # ショートエグジット: パーセントBが0.2以下から0.4以上に上昇
            elif percent_b[i-1] <= 0.2 and percent_b[i] >= 0.4:
                signals[i] = -1
    
    return signals


class HyperFRAMABollingerBreakoutSignal(IEntrySignal, IExitSignal):
    """
    HyperFRAMAボリンジャーバンドブレイクアウトシグナル
    
    ブレイクアウト戦略:
    - ロングエントリー: 上バンドブレイクアウト (1)
    - ショートエントリー: 下バンドブレイクアウト (-1)
    
    リバーサル戦略:
    - ロングエントリー: 下バンド付近からの反発 (1)
    - ショートエントリー: 上バンド付近からの反落 (-1)
    
    エグジット戦略:
    モード1 (逆ブレイクアウト):
    - ロングエグジット: 下バンド割れ (1)
    - ショートエグジット: 上バンド抜け (-1)
    
    モード2 (ミッドラインクロス):
    - ロングエグジット: ミッドライン割れ (1)
    - ショートエグジット: ミッドライン抜け (-1)
    
    モード3 (パーセントB反転):
    - ロングエグジット: パーセントB高値からの反転 (1)
    - ショートエグジット: パーセントB安値からの反転 (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        signal_type: str = "breakout",  # "breakout" or "reversal"
        lookback: int = 1,
        exit_mode: int = 2,  # 1: 逆ブレイクアウト, 2: ミッドラインクロス, 3: パーセントB反転
        src_type: str = 'close',
        
        # === HyperFRAMABollinger 基本パラメータ ===
        bollinger_period: int = 20,
        bollinger_sigma_mode: str = "dynamic",
        bollinger_fixed_sigma: float = 2.0,
        bollinger_src_type: str = "close",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        bollinger_hyper_frama_period: int = 16,
        bollinger_hyper_frama_src_type: str = 'hl2',
        bollinger_hyper_frama_fc: int = 1,
        bollinger_hyper_frama_sc: int = 198,
        bollinger_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        bollinger_hyper_frama_period_mode: str = 'fixed',
        bollinger_hyper_frama_cycle_detector_type: str = 'hody_e',
        bollinger_hyper_frama_lp_period: int = 13,
        bollinger_hyper_frama_hp_period: int = 124,
        bollinger_hyper_frama_cycle_part: float = 0.5,
        bollinger_hyper_frama_max_cycle: int = 89,
        bollinger_hyper_frama_min_cycle: int = 8,
        bollinger_hyper_frama_max_output: int = 124,
        bollinger_hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        bollinger_hyper_frama_enable_indicator_adaptation: bool = True,
        bollinger_hyper_frama_adaptation_indicator: str = 'hyper_er',
        bollinger_hyper_frama_hyper_er_period: int = 14,
        bollinger_hyper_frama_hyper_er_midline_period: int = 100,
        bollinger_hyper_frama_hyper_adx_period: int = 14,
        bollinger_hyper_frama_hyper_adx_midline_period: int = 100,
        bollinger_hyper_frama_hyper_trend_index_period: int = 14,
        bollinger_hyper_frama_hyper_trend_index_midline_period: int = 100,
        bollinger_hyper_frama_fc_min: float = 1.0,
        bollinger_hyper_frama_fc_max: float = 8.0,
        bollinger_hyper_frama_sc_min: float = 50.0,
        bollinger_hyper_frama_sc_max: float = 250.0,
        bollinger_hyper_frama_period_min: int = 4,
        bollinger_hyper_frama_period_max: int = 44,
        
        # === HyperER パラメータ ===
        bollinger_hyper_er_period: int = 8,
        bollinger_hyper_er_midline_period: int = 100,
        # ERパラメータ
        bollinger_hyper_er_er_period: int = 13,
        bollinger_hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        bollinger_hyper_er_use_kalman_filter: bool = True,
        bollinger_hyper_er_kalman_filter_type: str = 'simple',
        bollinger_hyper_er_kalman_process_noise: float = 1e-5,
        bollinger_hyper_er_kalman_min_observation_noise: float = 1e-6,
        bollinger_hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        bollinger_hyper_er_use_roofing_filter: bool = True,
        bollinger_hyper_er_roofing_hp_cutoff: float = 55.0,
        bollinger_hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        bollinger_hyper_er_use_laguerre_filter: bool = False,
        bollinger_hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        bollinger_hyper_er_use_smoothing: bool = True,
        bollinger_hyper_er_smoother_type: str = 'frama',
        bollinger_hyper_er_smoother_period: int = 16,
        bollinger_hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        bollinger_hyper_er_use_dynamic_period: bool = True,
        bollinger_hyper_er_detector_type: str = 'dft_dominant',
        bollinger_hyper_er_lp_period: int = 13,
        bollinger_hyper_er_hp_period: int = 124,
        bollinger_hyper_er_cycle_part: float = 0.4,
        bollinger_hyper_er_max_cycle: int = 124,
        bollinger_hyper_er_min_cycle: int = 13,
        bollinger_hyper_er_max_output: int = 89,
        bollinger_hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        bollinger_hyper_er_enable_percentile_analysis: bool = True,
        bollinger_hyper_er_percentile_lookback_period: int = 50,
        bollinger_hyper_er_percentile_low_threshold: float = 0.25,
        bollinger_hyper_er_percentile_high_threshold: float = 0.75,
        
        # === HyperFRAMAボリンジャー独自パラメータ ===
        # シグマ範囲設定
        bollinger_sigma_min: float = 1.0,
        bollinger_sigma_max: float = 2.5,
        bollinger_enable_signals: bool = True,
        bollinger_enable_percentile: bool = True,
        bollinger_percentile_period: int = 100
    ):
        # 基本パラメータ格納
        self.signal_type = signal_type
        self.lookback = lookback
        self.exit_mode = exit_mode
        self.src_type = src_type
        
        # ボリンジャー基本パラメータ
        self.bollinger_period = bollinger_period
        self.bollinger_sigma_mode = bollinger_sigma_mode
        self.bollinger_fixed_sigma = bollinger_fixed_sigma
        self.bollinger_src_type = bollinger_src_type
        self.bollinger_sigma_min = bollinger_sigma_min
        self.bollinger_sigma_max = bollinger_sigma_max
        
        # HyperFRAMAパラメータを辞書で格納
        self.hyper_frama_params = {
            "period": bollinger_hyper_frama_period,
            "src_type": bollinger_hyper_frama_src_type,
            "fc": bollinger_hyper_frama_fc,
            "sc": bollinger_hyper_frama_sc,
            "alpha_multiplier": bollinger_hyper_frama_alpha_multiplier,
            "period_mode": bollinger_hyper_frama_period_mode,
            "cycle_detector_type": bollinger_hyper_frama_cycle_detector_type,
            "lp_period": bollinger_hyper_frama_lp_period,
            "hp_period": bollinger_hyper_frama_hp_period,
            "cycle_part": bollinger_hyper_frama_cycle_part,
            "max_cycle": bollinger_hyper_frama_max_cycle,
            "min_cycle": bollinger_hyper_frama_min_cycle,
            "max_output": bollinger_hyper_frama_max_output,
            "min_output": bollinger_hyper_frama_min_output,
            "enable_indicator_adaptation": bollinger_hyper_frama_enable_indicator_adaptation,
            "adaptation_indicator": bollinger_hyper_frama_adaptation_indicator,
            "hyper_er_period": bollinger_hyper_frama_hyper_er_period,
            "hyper_er_midline_period": bollinger_hyper_frama_hyper_er_midline_period,
            "hyper_adx_period": bollinger_hyper_frama_hyper_adx_period,
            "hyper_adx_midline_period": bollinger_hyper_frama_hyper_adx_midline_period,
            "hyper_trend_index_period": bollinger_hyper_frama_hyper_trend_index_period,
            "hyper_trend_index_midline_period": bollinger_hyper_frama_hyper_trend_index_midline_period,
            "fc_min": bollinger_hyper_frama_fc_min,
            "fc_max": bollinger_hyper_frama_fc_max,
            "sc_min": bollinger_hyper_frama_sc_min,
            "sc_max": bollinger_hyper_frama_sc_max,
            "period_min": bollinger_hyper_frama_period_min,
            "period_max": bollinger_hyper_frama_period_max
        }
        
        # HyperERパラメータを辞書で格納
        self.hyper_er_params = {
            "period": bollinger_hyper_er_period,
            "midline_period": bollinger_hyper_er_midline_period,
            "er_period": bollinger_hyper_er_er_period,
            "er_src_type": bollinger_hyper_er_er_src_type,
            "use_kalman_filter": bollinger_hyper_er_use_kalman_filter,
            "kalman_filter_type": bollinger_hyper_er_kalman_filter_type,
            "kalman_process_noise": bollinger_hyper_er_kalman_process_noise,
            "kalman_min_observation_noise": bollinger_hyper_er_kalman_min_observation_noise,
            "kalman_adaptation_window": bollinger_hyper_er_kalman_adaptation_window,
            "use_roofing_filter": bollinger_hyper_er_use_roofing_filter,
            "roofing_hp_cutoff": bollinger_hyper_er_roofing_hp_cutoff,
            "roofing_ss_band_edge": bollinger_hyper_er_roofing_ss_band_edge,
            "use_laguerre_filter": bollinger_hyper_er_use_laguerre_filter,
            "laguerre_gamma": bollinger_hyper_er_laguerre_gamma,
            "use_smoothing": bollinger_hyper_er_use_smoothing,
            "smoother_type": bollinger_hyper_er_smoother_type,
            "smoother_period": bollinger_hyper_er_smoother_period,
            "smoother_src_type": bollinger_hyper_er_smoother_src_type,
            "use_dynamic_period": bollinger_hyper_er_use_dynamic_period,
            "detector_type": bollinger_hyper_er_detector_type,
            "lp_period": bollinger_hyper_er_lp_period,
            "hp_period": bollinger_hyper_er_hp_period,
            "cycle_part": bollinger_hyper_er_cycle_part,
            "max_cycle": bollinger_hyper_er_max_cycle,
            "min_cycle": bollinger_hyper_er_min_cycle,
            "max_output": bollinger_hyper_er_max_output,
            "min_output": bollinger_hyper_er_min_output,
            "enable_percentile_analysis": bollinger_hyper_er_enable_percentile_analysis,
            "percentile_lookback_period": bollinger_hyper_er_percentile_lookback_period,
            "percentile_low_threshold": bollinger_hyper_er_percentile_low_threshold,
            "percentile_high_threshold": bollinger_hyper_er_percentile_high_threshold
        }
        
        # その他パラメータ
        self.bollinger_enable_signals = bollinger_enable_signals
        self.bollinger_enable_percentile = bollinger_enable_percentile
        self.bollinger_percentile_period = bollinger_percentile_period
        
        # インジケーター初期化
        self._setup_indicators()
        
        # パラメータをまとめて格納
        self._params = {
            'signal_type': signal_type,
            'lookback': lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            'bollinger_period': bollinger_period,
            'bollinger_sigma_mode': bollinger_sigma_mode
        }

    def _setup_indicators(self):
        """インジケーターのセットアップ"""
        try:
            # HyperFRAMABollinger の設定
            self.hyper_frama_bollinger = HyperFRAMABollinger(
                period=self.bollinger_period,
                sigma_mode=self.bollinger_sigma_mode,
                fixed_sigma=self.bollinger_fixed_sigma,
                src_type=self.bollinger_src_type,
                
                # HyperFRAMAパラメータ
                **{f"hyper_frama_{k}": v for k, v in self.hyper_frama_params.items()},
                
                # HyperERパラメータ
                **{f"hyper_er_{k}": v for k, v in self.hyper_er_params.items()},
                
                # シグマ範囲設定
                sigma_min=self.bollinger_sigma_min,
                sigma_max=self.bollinger_sigma_max,
                enable_signals=self.bollinger_enable_signals,
                enable_percentile=self.bollinger_enable_percentile,
                percentile_period=self.bollinger_percentile_period
            )
            
        except Exception as e:
            raise RuntimeError(f"HyperFRAMAボリンジャーシグナルの初期化に失敗しました: {e}")

    def _calculate_bollinger_indicators(self, data: Union[pd.DataFrame, np.ndarray]) -> tuple:
        """ボリンジャーインジケーターの計算"""
        try:
            # ボリンジャー計算
            bollinger_result = self.hyper_frama_bollinger.calculate(data)
            
            return (
                bollinger_result.midline,
                bollinger_result.upper_band,
                bollinger_result.lower_band,
                bollinger_result.percent_b
            )
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return tuple(np.full(n_points, np.nan) for _ in range(4))

    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成"""
        try:
            # インジケーター計算
            midline, upper_band, lower_band, percent_b = self._calculate_bollinger_indicators(data)
            
            # 価格データ取得
            if isinstance(data, pd.DataFrame):
                close = data['close'].values
            else:
                close = data[:, 3] if data.shape[1] > 3 else data[:, -1]
            
            # シグナルタイプに応じてエントリーシグナル計算
            if self.signal_type == "breakout":
                entry_signals = calculate_bollinger_breakout_entry_signals(
                    close, upper_band, lower_band, percent_b, self.lookback
                )
            elif self.signal_type == "reversal":
                entry_signals = calculate_bollinger_reversal_entry_signals(
                    close, upper_band, lower_band, midline, percent_b, self.lookback
                )
            else:
                entry_signals = np.zeros(len(close), dtype=np.int8)
            
            return entry_signals
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.zeros(n_points, dtype=np.int8)

    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エグジットシグナル生成"""
        try:
            # インジケーター計算
            midline, upper_band, lower_band, percent_b = self._calculate_bollinger_indicators(data)
            
            # 価格データ取得
            if isinstance(data, pd.DataFrame):
                close = data['close'].values
            else:
                close = data[:, 3] if data.shape[1] > 3 else data[:, -1]
            
            # エグジットシグナル計算（Numba最適化）
            exit_signals = calculate_bollinger_exit_signals(
                close, upper_band, lower_band, midline, percent_b, self.exit_mode
            )
            
            return exit_signals
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.zeros(n_points, dtype=np.int8)

    # IEntrySignal インターフェース実装
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成（IEntrySignal用）"""
        return self.generate_entry(data)

    # 追加メソッド（付加情報取得用）
    def get_bollinger_values(self, data: Union[pd.DataFrame, np.ndarray]) -> tuple:
        """ボリンジャーバンド値を取得"""
        try:
            bollinger_result = self.hyper_frama_bollinger.calculate(data)
            return (
                bollinger_result.midline, 
                bollinger_result.upper_band, 
                bollinger_result.lower_band,
                bollinger_result.percent_b,
                bollinger_result.sigma_values
            )
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return tuple(np.full(n_points, np.nan) for _ in range(5))

    def get_source_price(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ソース価格を取得"""
        try:
            if isinstance(data, pd.DataFrame):
                data_array = data
            else:
                # NumPy配列をDataFrameに変換
                data_array = pd.DataFrame({
                    'open': data[:, 0],
                    'high': data[:, 1],
                    'low': data[:, 2],
                    'close': data[:, 3],
                    'volume': data[:, 4] if data.shape[1] > 4 else data[:, 3]
                })
            return PriceSource.calculate_source(data_array, self.src_type)
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.full(n_points, np.nan)

    @property
    def name(self) -> str:
        """シグナル名を取得"""
        exit_mode_str = {1: "逆ブレイクアウト", 2: "ミッドラインクロス", 3: "パーセントB反転"}.get(self.exit_mode, "不明")
        return f"HyperFRAMABollingerSignal(type={self.signal_type}, sigma={self.bollinger_sigma_mode}, exit={exit_mode_str}, lookback={self.lookback})"