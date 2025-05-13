#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_adaptive_ma import ZAdaptiveMACrossoverEntrySignal


@njit(fastmath=True, parallel=True, cache=True)
def calculate_crossover_signals(short_zma: np.ndarray, long_zma: np.ndarray) -> np.ndarray:
    """
    クロスオーバーシグナルを高速計算する（Numba最適化版）
    
    Args:
        short_zma: 短期ZAdaptiveMAの値
        long_zma: 長期ZAdaptiveMAの値
        
    Returns:
        np.ndarray: シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(short_zma)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の要素はクロスの判定ができないので処理しない
    for i in prange(1, length):
        # 前日と当日の状態を比較
        prev_short = short_zma[i-1]
        prev_long = long_zma[i-1]
        curr_short = short_zma[i]
        curr_long = long_zma[i]
        
        # ゴールデンクロス（短期が長期を上抜け）
        if prev_short <= prev_long and curr_short > curr_long:
            signals[i] = 1
        
        # デッドクロス（短期が長期を下抜け）
        elif prev_short >= prev_long and curr_short < curr_long:
            signals[i] = -1
    
    return signals


class ZAdaptiveMACrossoverSignalGenerator(BaseSignalGenerator):
    """
    ZAdaptiveMACrossoverのシグナル生成クラス（両方向・Numba最適化版）
    
    エントリー条件:
    - ロング: 短期ZMA > 長期ZMA (ゴールデンクロス)
    - ショート: 短期ZMA < 長期ZMA (デッドクロス)
    
    エグジット条件:
    - ロング: 短期ZMA < 長期ZMA (デッドクロス)
    - ショート: 短期ZMA > 長期ZMA (ゴールデンクロス)
    """
    
    def __init__(
        self,
        # 短期ZMAパラメータ
        short_fast_period: int = 2,
        short_slow_period: int = 30,
        # 長期ZMAパラメータ
        long_fast_period: int = 2,
        long_slow_period: int = 60,
        # 共通パラメータ
        src_type: str = 'hlc3',
        # 短期サイクルERパラメータ
        short_detector_type: str = 'hody',
        short_lp_period: int = 5,
        short_hp_period: int = 89,
        short_cycle_part: float = 0.5,
        short_cycle_max: int = 89,
        short_cycle_min: int = 5,
        short_max_output: int = 55,
        short_min_output: int = 5,
        short_use_kalman_filter: bool = False,
        short_kalman_measurement_noise: float = 1.0,
        short_kalman_process_noise: float = 0.01,
        short_kalman_n_states: int = 5,
        short_smooth_er: bool = True,
        short_er_alma_period: int = 5,
        short_er_alma_offset: float = 0.85,
        short_er_alma_sigma: float = 6,
        short_self_adaptive: bool = False,
        # 長期サイクルERパラメータ
        long_detector_type: str = 'hody',
        long_lp_period: int = 5,
        long_hp_period: int = 144,
        long_cycle_part: float = 0.5,
        long_cycle_max: int = 144,
        long_cycle_min: int = 8,
        long_max_output: int = 89,
        long_min_output: int = 8,
        long_use_kalman_filter: bool = False,
        long_kalman_measurement_noise: float = 1.0,
        long_kalman_process_noise: float = 0.01,
        long_kalman_n_states: int = 5,
        long_smooth_er: bool = True,
        long_er_alma_period: int = 5,
        long_er_alma_offset: float = 0.85,
        long_er_alma_sigma: float = 6,
        long_self_adaptive: bool = False
    ):
        """
        初期化
        
        Args:
            # 短期ZMAパラメータ
            short_fast_period: 短期ZMAの速い移動平均期間
            short_slow_period: 短期ZMAの遅い移動平均期間
            # 長期ZMAパラメータ
            long_fast_period: 長期ZMAの速い移動平均期間
            long_slow_period: 長期ZMAの遅い移動平均期間
            # 共通パラメータ
            src_type: 価格ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            
            # 短期サイクルERパラメータ
            short_detector_type: 短期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            short_lp_period: 短期サイクル用ローパスフィルター期間
            short_hp_period: 短期サイクル用ハイパスフィルター期間
            short_cycle_part: 短期サイクル部分比率
            short_cycle_max: 短期サイクルの最大期間
            short_cycle_min: 短期サイクルの最小期間
            short_max_output: 短期サイクルの最大出力値
            short_min_output: 短期サイクルの最小出力値
            short_use_kalman_filter: 短期サイクルでカルマンフィルターを使用するか
            short_kalman_measurement_noise: 短期カルマンフィルターの測定ノイズ
            short_kalman_process_noise: 短期カルマンフィルターのプロセスノイズ
            short_kalman_n_states: 短期カルマンフィルターの状態数
            short_smooth_er: 短期効率比をスムージングするかどうか
            short_er_alma_period: 短期効率比スムージング用ALMAの期間
            short_er_alma_offset: 短期効率比スムージング用ALMAのオフセット
            short_er_alma_sigma: 短期効率比スムージング用ALMAのシグマ
            short_self_adaptive: 短期効率比をセルフアダプティブにするかどうか
            
            # 長期サイクルERパラメータ
            long_detector_type: 長期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            long_lp_period: 長期サイクル用ローパスフィルター期間
            long_hp_period: 長期サイクル用ハイパスフィルター期間
            long_cycle_part: 長期サイクル部分比率
            long_cycle_max: 長期サイクルの最大期間
            long_cycle_min: 長期サイクルの最小期間
            long_max_output: 長期サイクルの最大出力値
            long_min_output: 長期サイクルの最小出力値
            long_use_kalman_filter: 長期サイクルでカルマンフィルターを使用するか
            long_kalman_measurement_noise: 長期カルマンフィルターの測定ノイズ
            long_kalman_process_noise: 長期カルマンフィルターのプロセスノイズ
            long_kalman_n_states: 長期カルマンフィルターの状態数
            long_smooth_er: 長期効率比をスムージングするかどうか
            long_er_alma_period: 長期効率比スムージング用ALMAの期間
            long_er_alma_offset: 長期効率比スムージング用ALMAのオフセット
            long_er_alma_sigma: 長期効率比スムージング用ALMAのシグマ
            long_self_adaptive: 長期効率比をセルフアダプティブにするかどうか
        """
        super().__init__("ZAdaptiveMACrossoverSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # 短期ZMAパラメータ
            'short_fast_period': short_fast_period,
            'short_slow_period': short_slow_period,
            # 長期ZMAパラメータ
            'long_fast_period': long_fast_period,
            'long_slow_period': long_slow_period,
            # 共通パラメータ
            'src_type': src_type,
            # 短期サイクルERパラメータ
            'short_detector_type': short_detector_type,
            'short_lp_period': short_lp_period,
            'short_hp_period': short_hp_period,
            'short_cycle_part': short_cycle_part,
            'short_cycle_max': short_cycle_max,
            'short_cycle_min': short_cycle_min,
            'short_max_output': short_max_output,
            'short_min_output': short_min_output,
            'short_use_kalman_filter': short_use_kalman_filter,
            'short_kalman_measurement_noise': short_kalman_measurement_noise,
            'short_kalman_process_noise': short_kalman_process_noise,
            'short_kalman_n_states': short_kalman_n_states,
            'short_smooth_er': short_smooth_er,
            'short_er_alma_period': short_er_alma_period,
            'short_er_alma_offset': short_er_alma_offset,
            'short_er_alma_sigma': short_er_alma_sigma,
            'short_self_adaptive': short_self_adaptive,
            # 長期サイクルERパラメータ
            'long_detector_type': long_detector_type,
            'long_lp_period': long_lp_period,
            'long_hp_period': long_hp_period,
            'long_cycle_part': long_cycle_part,
            'long_cycle_max': long_cycle_max,
            'long_cycle_min': long_cycle_min,
            'long_max_output': long_max_output,
            'long_min_output': long_min_output,
            'long_use_kalman_filter': long_use_kalman_filter,
            'long_kalman_measurement_noise': long_kalman_measurement_noise,
            'long_kalman_process_noise': long_kalman_process_noise,
            'long_kalman_n_states': long_kalman_n_states,
            'long_smooth_er': long_smooth_er,
            'long_er_alma_period': long_er_alma_period,
            'long_er_alma_offset': long_er_alma_offset,
            'long_er_alma_sigma': long_er_alma_sigma,
            'long_self_adaptive': long_self_adaptive
        }
        
        # ZAdaptiveMACrossoverシグナルの初期化
        self.z_adaptive_ma_signal = ZAdaptiveMACrossoverEntrySignal(
            # 短期ZMAパラメータ
            short_fast_period=short_fast_period,
            short_slow_period=short_slow_period,
            # 長期ZMAパラメータ
            long_fast_period=long_fast_period,
            long_slow_period=long_slow_period,
            # 共通パラメータ
            src_type=src_type,
            # 短期サイクルERパラメータ
            short_detector_type=short_detector_type,
            short_lp_period=short_lp_period,
            short_hp_period=short_hp_period,
            short_cycle_part=short_cycle_part,
            short_cycle_max=short_cycle_max,
            short_cycle_min=short_cycle_min,
            short_max_output=short_max_output,
            short_min_output=short_min_output,
            short_use_kalman_filter=short_use_kalman_filter,
            short_kalman_measurement_noise=short_kalman_measurement_noise,
            short_kalman_process_noise=short_kalman_process_noise,
            short_kalman_n_states=short_kalman_n_states,
            short_smooth_er=short_smooth_er,
            short_er_alma_period=short_er_alma_period,
            short_er_alma_offset=short_er_alma_offset,
            short_er_alma_sigma=short_er_alma_sigma,
            short_self_adaptive=short_self_adaptive,
            # 長期サイクルERパラメータ
            long_detector_type=long_detector_type,
            long_lp_period=long_lp_period,
            long_hp_period=long_hp_period,
            long_cycle_part=long_cycle_part,
            long_cycle_max=long_cycle_max,
            long_cycle_min=long_cycle_min,
            long_max_output=long_max_output,
            long_min_output=long_min_output,
            long_use_kalman_filter=long_use_kalman_filter,
            long_kalman_measurement_noise=long_kalman_measurement_noise,
            long_kalman_process_noise=long_kalman_process_noise,
            long_kalman_n_states=long_kalman_n_states,
            long_smooth_er=long_smooth_er,
            long_er_alma_period=long_er_alma_period,
            long_er_alma_offset=long_er_alma_offset,
            long_er_alma_sigma=long_er_alma_sigma,
            long_self_adaptive=long_self_adaptive
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._short_zma = None
        self._long_zma = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（Numba最適化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # ZAdaptiveMACrossoverシグナルから内部ZMA値を取得
                # インジケーターそのものを直接呼び出す場合
                try:
                    # シグナルを取得（ZAdaptiveMACrossoverEntrySignalのgenerateメソッドを実行）
                    # このメソッド内で短期・長期ZMAが計算される
                    cross_signals = self.z_adaptive_ma_signal.generate(df)
                    
                    # 内部の計算済みZMA値へのアクセス
                    # Numba最適化用にget_short_ma()とget_long_ma()メソッドを使用
                    short_zma_values = self.z_adaptive_ma_signal.get_short_ma()
                    long_zma_values = self.z_adaptive_ma_signal.get_long_ma()
                    
                    # ZMA値がない場合は再計算
                    if short_zma_values is None or long_zma_values is None or len(short_zma_values) != current_len:
                        # 既存のシグナルをそのまま使用
                        self._signals = cross_signals
                    else:
                        # ZMA値をキャッシュ
                        self._short_zma = short_zma_values
                        self._long_zma = long_zma_values
                        
                        # Numba高速化関数でシグナルを計算
                        self._signals = calculate_crossover_signals(short_zma_values, long_zma_values)
                except Exception as e:
                    self.logger.error(f"ZMA値取得中にエラー: {str(e)}")
                    # エラー時はシグナル単純計算に切り替え
                    cross_signals = self.z_adaptive_ma_signal.generate(df)
                    self._signals = cross_signals
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（Numba最適化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（Numba最適化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._signals[index] == -1)  # デッドクロスでエグジット
        elif position == -1:  # ショートポジション
            return bool(self._signals[index] == 1)   # ゴールデンクロスでエグジット
        return False
    
    def get_short_ma(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期ZMAの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行。
            
        Returns:
            np.ndarray: 短期ZMAの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._short_zma is not None:
                return self._short_zma
            
            # 値が取得できない場合はエラー
            return np.array([])
        except Exception as e:
            self.logger.error(f"短期ZMA取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_ma(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期ZMAの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行。
            
        Returns:
            np.ndarray: 長期ZMAの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._long_zma is not None:
                return self._long_zma
            
            # 値が取得できない場合はエラー
            return np.array([])
        except Exception as e:
            self.logger.error(f"長期ZMA取得中にエラー: {str(e)}")
            return np.array([]) 