#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_frama_bollinger.bollinger_breakout_signal import HyperFRAMABollingerBreakoutSignal
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class SignalType(Enum):
    """HyperFRAMAボリンジャー戦略用のシグナルタイプ"""
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class FilterType(Enum):
    """HyperFRAMAボリンジャー戦略用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True, parallel=False)
def combine_bollinger_signals_numba(
    bollinger_entry_signals: np.ndarray, 
    bollinger_exit_signals: np.ndarray,
    filter_signals: np.ndarray, 
    filter_type: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperFRAMAボリンジャーシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        bollinger_entry_signals: ボリンジャーエントリーシグナル配列
        bollinger_exit_signals: ボリンジャーエグジットシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=HyperER, 2=HyperTrendIndex, 3=HyperADX, 4=Consensus）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (統合エントリーシグナル, 統合エグジットシグナル)
    """
    length = len(bollinger_entry_signals)
    final_entry_signals = np.zeros(length, dtype=np.int8)
    final_exit_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        bollinger_entry = bollinger_entry_signals[i]
        bollinger_exit = bollinger_exit_signals[i]
        
        # フィルターなしの場合
        if filter_type == 0:
            final_entry_signals[i] = bollinger_entry
            final_exit_signals[i] = bollinger_exit
        else:
            # フィルターありの場合
            if i < len(filter_signals):
                filter_signal = filter_signals[i]
                
                # エントリーシグナル: ボリンジャーシグナル かつ フィルター許可
                if bollinger_entry != 0 and filter_signal == 1:
                    final_entry_signals[i] = bollinger_entry
                
                # エグジットシグナル: ボリンジャーエグジットはフィルター関係なく実行
                final_exit_signals[i] = bollinger_exit
    
    return final_entry_signals, final_exit_signals


@njit(fastmath=True)
def consensus_filter_numba(
    hyper_er_signals: np.ndarray,
    trend_index_signals: np.ndarray,
    hyper_adx_signals: np.ndarray
) -> np.ndarray:
    """3つの指標のうち2つ以上が1の場合に1を出力"""
    n = len(hyper_er_signals)
    result = np.zeros(n)
    
    for i in range(n):
        count = 0
        if hyper_er_signals[i] == 1.0:
            count += 1
        if trend_index_signals[i] == 1.0:
            count += 1
        if hyper_adx_signals[i] == 1.0:
            count += 1
        
        # 2つ以上が1の場合に1を出力、それ以外は-1
        if count >= 2:
            result[i] = 1.0
        else:
            result[i] = -1.0
    
    return result


@njit(fastmath=True, parallel=False)
def check_bollinger_exit_conditions_numba(
    bollinger_exit_signals: np.ndarray, 
    position: int, 
    index: int
) -> bool:
    """
    ボリンジャーエグジット条件をチェックする（Numba最適化版）
    
    Args:
        bollinger_exit_signals: ボリンジャーエグジットシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(bollinger_exit_signals):
        return False
    
    bollinger_exit = bollinger_exit_signals[index]
    
    # ロングポジション: エグジットシグナル=1でエグジット
    if position == 1 and bollinger_exit == 1:
        return True
    
    # ショートポジション: エグジットシグナル=-1でエグジット
    if position == -1 and bollinger_exit == -1:
        return True
    
    return False


class HyperFRAMABollingerSignalGenerator(BaseSignalGenerator):
    """
    HyperFRAMAボリンジャー シグナル生成クラス
    
    特徴:
    - HyperFRAMAボリンジャーバンドベースのブレイクアウト/リバーサル戦略
    - 動的シグマ適応（HyperERベース）による市場環境適応
    - 複数フィルターによる誤シグナル削減
    - Numba最適化による高速処理
    
    エントリー戦略:
    - ブレイクアウト: バンド突破でエントリー
    - リバーサル: バンド付近からの反転でエントリー
    
    エグジット戦略:
    - 逆ブレイクアウト: 反対側バンド突破
    - ミッドラインクロス: HyperFRAMAミッドライン突破
    - パーセントB反転: パーセントBの反転
    """
    
    def __init__(
        self,
        # ボリンジャー基本パラメータ
        signal_type: SignalType = SignalType.BREAKOUT,
        lookback: int = 1,
        exit_mode: int = 2,  # 1: 逆ブレイクアウト, 2: ミッドラインクロス, 3: パーセントB反転
        src_type: str = 'close',
        
        # === HyperFRAMABollinger パラメータ ===
        bollinger_period: int = 20,
        bollinger_sigma_mode: str = "dynamic",
        bollinger_fixed_sigma: float = 2.0,
        bollinger_src_type: str = "close",
        
        # === HyperFRAMA パラメータ ===
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
        bollinger_hyper_er_er_period: int = 13,
        bollinger_hyper_er_er_src_type: str = 'oc2',
        bollinger_hyper_er_use_kalman_filter: bool = True,
        bollinger_hyper_er_kalman_filter_type: str = 'simple',
        bollinger_hyper_er_kalman_process_noise: float = 1e-5,
        bollinger_hyper_er_kalman_min_observation_noise: float = 1e-6,
        bollinger_hyper_er_kalman_adaptation_window: int = 5,
        bollinger_hyper_er_use_roofing_filter: bool = True,
        bollinger_hyper_er_roofing_hp_cutoff: float = 55.0,
        bollinger_hyper_er_roofing_ss_band_edge: float = 10.0,
        bollinger_hyper_er_use_laguerre_filter: bool = False,
        bollinger_hyper_er_laguerre_gamma: float = 0.5,
        bollinger_hyper_er_use_smoothing: bool = True,
        bollinger_hyper_er_smoother_type: str = 'frama',
        bollinger_hyper_er_smoother_period: int = 16,
        bollinger_hyper_er_smoother_src_type: str = 'close',
        bollinger_hyper_er_use_dynamic_period: bool = True,
        bollinger_hyper_er_detector_type: str = 'dft_dominant',
        bollinger_hyper_er_lp_period: int = 13,
        bollinger_hyper_er_hp_period: int = 124,
        bollinger_hyper_er_cycle_part: float = 0.4,
        bollinger_hyper_er_max_cycle: int = 124,
        bollinger_hyper_er_min_cycle: int = 13,
        bollinger_hyper_er_max_output: int = 89,
        bollinger_hyper_er_min_output: int = 5,
        bollinger_hyper_er_enable_percentile_analysis: bool = True,
        bollinger_hyper_er_percentile_lookback_period: int = 50,
        bollinger_hyper_er_percentile_low_threshold: float = 0.25,
        bollinger_hyper_er_percentile_high_threshold: float = 0.75,
        
        # シグマ範囲設定
        bollinger_sigma_min: float = 1.0,
        bollinger_sigma_max: float = 2.5,
        bollinger_enable_signals: bool = True,
        bollinger_enable_percentile: bool = True,
        bollinger_percentile_period: int = 100,
        
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,
        # HyperER フィルターパラメータ
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """
        初期化
        
        Args:
            signal_type: シグナルタイプ（ブレイクアウト or リバーサル）
            lookback: ルックバック期間
            exit_mode: エグジットモード
            src_type: 価格ソースタイプ
            その他: HyperFRAMAボリンジャーと各フィルターのパラメータ
        """
        signal_name = signal_type.value if isinstance(signal_type, SignalType) else str(signal_type)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        super().__init__(f"HyperFRAMABollinger_{signal_name}_{filter_name}")
        
        # パラメータの設定
        self._params = {
            # ボリンジャー基本パラメータ
            'signal_type': signal_type,
            'lookback': lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            # HyperFRAMAボリンジャーパラメータ
            'bollinger_period': bollinger_period,
            'bollinger_sigma_mode': bollinger_sigma_mode,
            'bollinger_fixed_sigma': bollinger_fixed_sigma,
            'bollinger_src_type': bollinger_src_type,
            'bollinger_sigma_min': bollinger_sigma_min,
            'bollinger_sigma_max': bollinger_sigma_max,
            # フィルター設定
            'filter_type': filter_type,
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        self.signal_type = signal_type if isinstance(signal_type, SignalType) else SignalType(signal_type)
        self.filter_type = filter_type if isinstance(filter_type, FilterType) else FilterType(filter_type)
        
        # HyperFRAMAボリンジャーシグナルの初期化
        self.bollinger_signal = HyperFRAMABollingerBreakoutSignal(
            signal_type=signal_name,
            lookback=lookback,
            exit_mode=exit_mode,
            src_type=src_type,
            # ボリンジャー基本パラメータ
            bollinger_period=bollinger_period,
            bollinger_sigma_mode=bollinger_sigma_mode,
            bollinger_fixed_sigma=bollinger_fixed_sigma,
            bollinger_src_type=bollinger_src_type,
            
            # HyperFRAMAパラメータ
            bollinger_hyper_frama_period=bollinger_hyper_frama_period,
            bollinger_hyper_frama_src_type=bollinger_hyper_frama_src_type,
            bollinger_hyper_frama_fc=bollinger_hyper_frama_fc,
            bollinger_hyper_frama_sc=bollinger_hyper_frama_sc,
            bollinger_hyper_frama_alpha_multiplier=bollinger_hyper_frama_alpha_multiplier,
            bollinger_hyper_frama_period_mode=bollinger_hyper_frama_period_mode,
            bollinger_hyper_frama_cycle_detector_type=bollinger_hyper_frama_cycle_detector_type,
            bollinger_hyper_frama_lp_period=bollinger_hyper_frama_lp_period,
            bollinger_hyper_frama_hp_period=bollinger_hyper_frama_hp_period,
            bollinger_hyper_frama_cycle_part=bollinger_hyper_frama_cycle_part,
            bollinger_hyper_frama_max_cycle=bollinger_hyper_frama_max_cycle,
            bollinger_hyper_frama_min_cycle=bollinger_hyper_frama_min_cycle,
            bollinger_hyper_frama_max_output=bollinger_hyper_frama_max_output,
            bollinger_hyper_frama_min_output=bollinger_hyper_frama_min_output,
            bollinger_hyper_frama_enable_indicator_adaptation=bollinger_hyper_frama_enable_indicator_adaptation,
            bollinger_hyper_frama_adaptation_indicator=bollinger_hyper_frama_adaptation_indicator,
            bollinger_hyper_frama_hyper_er_period=bollinger_hyper_frama_hyper_er_period,
            bollinger_hyper_frama_hyper_er_midline_period=bollinger_hyper_frama_hyper_er_midline_period,
            bollinger_hyper_frama_hyper_adx_period=bollinger_hyper_frama_hyper_adx_period,
            bollinger_hyper_frama_hyper_adx_midline_period=bollinger_hyper_frama_hyper_adx_midline_period,
            bollinger_hyper_frama_hyper_trend_index_period=bollinger_hyper_frama_hyper_trend_index_period,
            bollinger_hyper_frama_hyper_trend_index_midline_period=bollinger_hyper_frama_hyper_trend_index_midline_period,
            bollinger_hyper_frama_fc_min=bollinger_hyper_frama_fc_min,
            bollinger_hyper_frama_fc_max=bollinger_hyper_frama_fc_max,
            bollinger_hyper_frama_sc_min=bollinger_hyper_frama_sc_min,
            bollinger_hyper_frama_sc_max=bollinger_hyper_frama_sc_max,
            bollinger_hyper_frama_period_min=bollinger_hyper_frama_period_min,
            bollinger_hyper_frama_period_max=bollinger_hyper_frama_period_max,
            
            # HyperERパラメータ
            bollinger_hyper_er_period=bollinger_hyper_er_period,
            bollinger_hyper_er_midline_period=bollinger_hyper_er_midline_period,
            bollinger_hyper_er_er_period=bollinger_hyper_er_er_period,
            bollinger_hyper_er_er_src_type=bollinger_hyper_er_er_src_type,
            bollinger_hyper_er_use_kalman_filter=bollinger_hyper_er_use_kalman_filter,
            bollinger_hyper_er_kalman_filter_type=bollinger_hyper_er_kalman_filter_type,
            bollinger_hyper_er_kalman_process_noise=bollinger_hyper_er_kalman_process_noise,
            bollinger_hyper_er_kalman_min_observation_noise=bollinger_hyper_er_kalman_min_observation_noise,
            bollinger_hyper_er_kalman_adaptation_window=bollinger_hyper_er_kalman_adaptation_window,
            bollinger_hyper_er_use_roofing_filter=bollinger_hyper_er_use_roofing_filter,
            bollinger_hyper_er_roofing_hp_cutoff=bollinger_hyper_er_roofing_hp_cutoff,
            bollinger_hyper_er_roofing_ss_band_edge=bollinger_hyper_er_roofing_ss_band_edge,
            bollinger_hyper_er_use_laguerre_filter=bollinger_hyper_er_use_laguerre_filter,
            bollinger_hyper_er_laguerre_gamma=bollinger_hyper_er_laguerre_gamma,
            bollinger_hyper_er_use_smoothing=bollinger_hyper_er_use_smoothing,
            bollinger_hyper_er_smoother_type=bollinger_hyper_er_smoother_type,
            bollinger_hyper_er_smoother_period=bollinger_hyper_er_smoother_period,
            bollinger_hyper_er_smoother_src_type=bollinger_hyper_er_smoother_src_type,
            bollinger_hyper_er_use_dynamic_period=bollinger_hyper_er_use_dynamic_period,
            bollinger_hyper_er_detector_type=bollinger_hyper_er_detector_type,
            bollinger_hyper_er_lp_period=bollinger_hyper_er_lp_period,
            bollinger_hyper_er_hp_period=bollinger_hyper_er_hp_period,
            bollinger_hyper_er_cycle_part=bollinger_hyper_er_cycle_part,
            bollinger_hyper_er_max_cycle=bollinger_hyper_er_max_cycle,
            bollinger_hyper_er_min_cycle=bollinger_hyper_er_min_cycle,
            bollinger_hyper_er_max_output=bollinger_hyper_er_max_output,
            bollinger_hyper_er_min_output=bollinger_hyper_er_min_output,
            bollinger_hyper_er_enable_percentile_analysis=bollinger_hyper_er_enable_percentile_analysis,
            bollinger_hyper_er_percentile_lookback_period=bollinger_hyper_er_percentile_lookback_period,
            bollinger_hyper_er_percentile_low_threshold=bollinger_hyper_er_percentile_low_threshold,
            bollinger_hyper_er_percentile_high_threshold=bollinger_hyper_er_percentile_high_threshold,
            
            # シグマ範囲設定
            bollinger_sigma_min=bollinger_sigma_min,
            bollinger_sigma_max=bollinger_sigma_max,
            bollinger_enable_signals=bollinger_enable_signals,
            bollinger_enable_percentile=bollinger_enable_percentile,
            bollinger_percentile_period=bollinger_percentile_period
        )
        
        # フィルターインジケーター（必要に応じて初期化）
        self.hyper_er_filter = None
        self.hyper_trend_index_filter = None
        self.hyper_adx_filter = None
        
        if self.filter_type != FilterType.NONE:
            # フィルター用インジケーターを初期化
            self.hyper_er_filter = HyperER(
                period=filter_hyper_er_period,
                midline_period=filter_hyper_er_midline_period
            )
            
            self.hyper_trend_index_filter = HyperTrendIndex(
                period=filter_hyper_trend_index_period,
                midline_period=filter_hyper_trend_index_midline_period
            )
            
            self.hyper_adx_filter = HyperADX(
                period=filter_hyper_adx_period,
                midline_period=filter_hyper_adx_midline_period
            )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        self._bollinger_entry_signals = None
        self._bollinger_exit_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._entry_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                    if 'volume' in data.columns:
                        df = data[['open', 'high', 'low', 'close', 'volume']]
                else:
                    if data.shape[1] >= 5:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                    else:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # ボリンジャーシグナルの計算
                    self._bollinger_entry_signals = self.bollinger_signal.generate_entry(df)
                    self._bollinger_exit_signals = self.bollinger_signal.generate_exit(df)
                    
                    # フィルターシグナルの計算
                    self._filter_signals = self._get_filter_signals(df)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    self._entry_signals, self._exit_signals = combine_bollinger_signals_numba(
                        self._bollinger_entry_signals, 
                        self._bollinger_exit_signals,
                        self._filter_signals, 
                        filter_type_int
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._entry_signals = np.zeros(current_len, dtype=np.int8)
                    self._exit_signals = np.zeros(current_len, dtype=np.int8)
                    self._bollinger_entry_signals = np.zeros(current_len, dtype=np.int8)
                    self._bollinger_exit_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                self._exit_signals = np.zeros(len(data), dtype=np.int8)
                self._bollinger_entry_signals = np.zeros(len(data), dtype=np.int8)
                self._bollinger_exit_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def _get_filter_signals(self, data: pd.DataFrame) -> np.ndarray:
        """フィルターシグナルを取得"""
        if self.filter_type == FilterType.HYPER_ER:
            hyper_er_result = self.hyper_er_filter.calculate(data)
            return self.hyper_er_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_TREND_INDEX:
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            return self.hyper_trend_index_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_ADX:
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            return self.hyper_adx_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.CONSENSUS:
            # 統合フィルター（3つのうち2つが1なら1）
            hyper_er_result = self.hyper_er_filter.calculate(data)
            hyper_er_signals = self.hyper_er_filter.get_trend_signal()
            
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            trend_index_signals = self.hyper_trend_index_filter.get_trend_signal()
            
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            hyper_adx_signals = self.hyper_adx_filter.get_trend_signal()
            
            return consensus_filter_numba(
                hyper_er_signals,
                trend_index_signals, 
                hyper_adx_signals
            )
        
        # デフォルト（フィルターなし）
        return np.ones(len(data))

    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得
        
        Returns:
            統合されたエントリーシグナル（ロング=1、ショート=-1、なし=0）
        """
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_bollinger_exit_conditions_numba(self._exit_signals, position, index)
    
    def get_bollinger_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ボリンジャーエントリーシグナル取得"""
        if self._bollinger_entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._bollinger_entry_signals
    
    def get_bollinger_exit_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ボリンジャーエグジットシグナル取得"""
        if self._bollinger_exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._bollinger_exit_signals
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        if self._filter_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._filter_signals
    
    def get_bollinger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.bollinger_signal.get_bollinger_values(data)
        except Exception as e:
            self.logger.error(f"ボリンジャーバンド値取得中にエラー: {str(e)}")
            n = len(data) if data is not None else 100
            return tuple(np.full(n, np.nan) for _ in range(5))
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        フィルター詳細情報を取得
        
        Returns:
            フィルタータイプに応じた詳細データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self.filter_type == FilterType.NONE:
                return {}
            
            details = {}
            
            if self.filter_type == FilterType.HYPER_ER:
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_trend_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_TREND_INDEX:
                details.update({
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_ADX:
                details.update({
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.CONSENSUS:
                # 各インジケーターの個別シグナルも追加
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([]),
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([]),
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            
            return details
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            HyperFRAMAボリンジャーとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            # ボリンジャーバンド値取得
            midline, upper_band, lower_band, percent_b, sigma_values = self.get_bollinger_values(data)
            
            metrics = {
                # ボリンジャーバンドメトリクス
                'midline': midline,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'percent_b': percent_b,
                'sigma_values': sigma_values,
                # シグナルメトリクス
                'entry_signals': self.get_entry_signals(data),
                'bollinger_entry_signals': self.get_bollinger_entry_signals(data),
                'bollinger_exit_signals': self.get_bollinger_exit_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # 設定情報
                'signal_type': self.signal_type.value,
                'filter_type': self.filter_type.value
            }
            
            # フィルター固有のメトリクスを追加
            filter_details = self.get_filter_details()
            metrics.update(filter_details)
            
            return metrics
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        self._bollinger_entry_signals = None
        self._bollinger_exit_signals = None
        self._filter_signals = None
        
        if hasattr(self.bollinger_signal, 'reset'):
            self.bollinger_signal.reset()
        if self.hyper_er_filter is not None and hasattr(self.hyper_er_filter, 'reset'):
            self.hyper_er_filter.reset()
        if self.hyper_trend_index_filter is not None and hasattr(self.hyper_trend_index_filter, 'reset'):
            self.hyper_trend_index_filter.reset()
        if self.hyper_adx_filter is not None and hasattr(self.hyper_adx_filter, 'reset'):
            self.hyper_adx_filter.reset()