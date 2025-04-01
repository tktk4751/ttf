#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_trend.direction import ZTrendDirectionSignal
from signals.implementations.z_trend_filter.filter import ZTrendFilterSignal


@njit(fastmath=True, parallel=True)
def calculate_entry_signals(direction_signal: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）
    
    Parameters
    ----------
    direction_signal : np.ndarray
        ZTrendDirectionSignalの配列（1=上昇トレンド、-1=下降トレンド）
    filter_signal : np.ndarray
        ZTrendFilterSignalの配列（1=トレンド相場、-1=レンジ相場）
    
    Returns
    -------
    np.ndarray
        エントリーシグナルの配列 (1=ロング、-1=ショート、0=ニュートラル)
    """
    signals = np.zeros_like(direction_signal, dtype=np.int8)
    
    # ロングエントリー: ZTrendDirectionSignalが1(上昇トレンド) + ZTrendFilterSignalが1(トレンド相場)
    long_condition = (direction_signal == 1) & (filter_signal == 1)
    
    # ショートエントリー: ZTrendDirectionSignalが-1(下降トレンド) + ZTrendFilterSignalが1(トレンド相場)
    short_condition = (direction_signal == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class ZTrendSignalGenerator(BaseSignalGenerator):
    """
    ZTrendDirectionSignal + ZTrendFilterSignalのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: ZTrendDirectionSignalが1(上昇トレンド) + ZTrendFilterSignalが1(トレンド相場)
    - ショート: ZTrendDirectionSignalが-1(下降トレンド) + ZTrendFilterSignalが1(トレンド相場)
    
    エグジット条件:
    - ロング: ZTrendDirectionSignalが-1(下降トレンド)に変化
    - ショート: ZTrendDirectionSignalが1(上昇トレンド)に変化
    """
    
    def __init__(
        self,
        # ZTrendDirectionSignalのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 233,
        cer_min_cycle: int = 13,
        cer_max_output: int = 144,
        cer_min_output: int = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        # ATR乗数
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3',
        
        # Zトレンドフィルターのパラメータ
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        combination_weight: float = 0.6,
        zadx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        
        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5
    ):
        """初期化"""
        super().__init__("ZTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'max_percentile_dc_cycle_part': max_percentile_dc_cycle_part,
            'max_percentile_dc_max_cycle': max_percentile_dc_max_cycle,
            'max_percentile_dc_min_cycle': max_percentile_dc_min_cycle,
            'max_percentile_dc_max_output': max_percentile_dc_max_output,
            'max_percentile_dc_min_output': max_percentile_dc_min_output,
            'min_percentile_dc_cycle_part': min_percentile_dc_cycle_part,
            'min_percentile_dc_max_cycle': min_percentile_dc_max_cycle,
            'min_percentile_dc_min_cycle': min_percentile_dc_min_cycle,
            'min_percentile_dc_max_output': min_percentile_dc_max_output,
            'min_percentile_dc_min_output': min_percentile_dc_min_output,
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            'max_percentile_cycle_mult': max_percentile_cycle_mult,
            'min_percentile_cycle_mult': min_percentile_cycle_mult,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_weight': combination_weight,
            'zadx_weight': zadx_weight,
            'combination_method': combination_method,
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output
        }
        
        # ZTrendDirectionSignalの初期化
        self.direction_signal = ZTrendDirectionSignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type
        )
        
        # ZTrendFilterSignalの初期化
        self.filter_signal = ZTrendFilterSignal(
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            combination_weight=combination_weight,
            zadx_weight=zadx_weight,
            combination_method=combination_method,
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output,
            smoother_type=smoother_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._direction_values = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    direction_values = self.direction_signal.generate(df)
                    filter_values = self.filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(direction_values, filter_values)
                    
                    # エグジット用のシグナルを事前計算
                    self._direction_values = direction_values
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._direction_values = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._direction_values = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            # ZTrendDirectionSignalが-1になったらロング決済
            return bool(self._direction_values[index] == -1)
        elif position == -1:  # ショートポジション
            # ZTrendDirectionSignalが1になったらショート決済
            return bool(self._direction_values[index] == 1)
        return False
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZTrendの上バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 上バンド値の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_upper_band(data)
        except Exception as e:
            self.logger.error(f"上バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZTrendの下バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 下バンド値の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_lower_band(data)
        except Exception as e:
            self.logger.error(f"下バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_cycle_er(data)
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドフィルターの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.filter_signal.get_filter_values()
        except Exception as e:
            self.logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
        
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドフィルターの動的しきい値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的しきい値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.filter_signal.get_threshold_values()
        except Exception as e:
            self.logger.error(f"しきい値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドインデックスの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドインデックス値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.filter_signal.get_trend_index()
        except Exception as e:
            self.logger.error(f"トレンドインデックス取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_dynamic_multiplier(data)
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_percentile_length(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的パーセンタイル期間の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的パーセンタイル期間の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_dynamic_percentile_length(data)
        except Exception as e:
            self.logger.error(f"動的パーセンタイル期間取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZATRの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.direction_signal.get_z_atr(data)
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([]) 