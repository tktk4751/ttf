#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.mesa_frama.entry import MESAFRAMACrossoverEntrySignal


class MESAFRAMASignalGenerator(BaseSignalGenerator):
    """
    MESA_FRAMAデュアルクロスオーバーシグナル生成クラス
    
    エントリー条件:
    1. クロスオーバーモード:
       - ロング: 短期MESA_FRAMAが長期MESA_FRAMAを上抜け
       - ショート: 短期MESA_FRAMAが長期MESA_FRAMAを下抜け
    2. 位置関係モード:
       - ロング: 短期MESA_FRAMA > 長期MESA_FRAMA
       - ショート: 短期MESA_FRAMA < 長期MESA_FRAMA
    
    エグジット条件:
    - ロング: 短期MESA_FRAMAが長期MESA_FRAMAを下抜け
    - ショート: 短期MESA_FRAMAが長期MESA_FRAMAを上抜け
    """
    
    def __init__(
        self,
        # 短期MESA_FRAMAパラメータ
        fast_base_period: int = 8,               # 短期基本期間
        fast_src_type: str = 'hl2',              # 短期ソースタイプ
        fast_fc: int = 1,                        # 短期Fast Constant
        fast_sc: int = 198,                      # 短期Slow Constant
        fast_mesa_fast_limit: float = 0.7,      # 短期MESA高速制限値
        fast_mesa_slow_limit: float = 0.1,      # 短期MESA低速制限値
        # 長期MESA_FRAMAパラメータ
        slow_base_period: int = 32,              # 長期基本期間
        slow_src_type: str = 'hl2',              # 長期ソースタイプ
        slow_fc: int = 1,                        # 長期Fast Constant
        slow_sc: int = 198,                      # 長期Slow Constant
        slow_mesa_fast_limit: float = 0.3,      # 長期MESA高速制限値
        slow_mesa_slow_limit: float = 0.02,     # 長期MESA低速制限値
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,               # ゼロラグ処理を使用するか
        # シグナルパラメータ
        signal_mode: str = 'crossover',          # シグナルモード ('crossover' または 'position')
        crossover_threshold: float = 0.0        # クロスオーバー閾値
    ):
        """
        初期化
        
        Args:
            fast_base_period: 短期基本期間（偶数である必要がある、デフォルト: 8）
            fast_src_type: 短期ソースタイプ（デフォルト: 'hl2'）
            fast_fc: 短期Fast Constant（デフォルト: 1）
            fast_sc: 短期Slow Constant（デフォルト: 198）
            fast_mesa_fast_limit: 短期MESA高速制限値（デフォルト: 0.7）
            fast_mesa_slow_limit: 短期MESA低速制限値（デフォルト: 0.1）
            slow_base_period: 長期基本期間（偶数である必要がある、デフォルト: 32）
            slow_src_type: 長期ソースタイプ（デフォルト: 'hl2'）
            slow_fc: 長期Fast Constant（デフォルト: 1）
            slow_sc: 長期Slow Constant（デフォルト: 198）
            slow_mesa_fast_limit: 長期MESA高速制限値（デフォルト: 0.3）
            slow_mesa_slow_limit: 長期MESA低速制限値（デフォルト: 0.02）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            signal_mode: シグナルモード（デフォルト: 'crossover'）
            crossover_threshold: クロスオーバー閾値（デフォルト: 0.0）
        """
        super().__init__("MESAFRAMADualSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'fast_base_period': fast_base_period,
            'fast_src_type': fast_src_type,
            'fast_fc': fast_fc,
            'fast_sc': fast_sc,
            'fast_mesa_fast_limit': fast_mesa_fast_limit,
            'fast_mesa_slow_limit': fast_mesa_slow_limit,
            'slow_base_period': slow_base_period,
            'slow_src_type': slow_src_type,
            'slow_fc': slow_fc,
            'slow_sc': slow_sc,
            'slow_mesa_fast_limit': slow_mesa_fast_limit,
            'slow_mesa_slow_limit': slow_mesa_slow_limit,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'signal_mode': signal_mode,
            'crossover_threshold': crossover_threshold
        }
        
        # デュアルMESA_FRAMAエントリーシグナルの初期化
        self.mesa_frama_entry_signal = MESAFRAMACrossoverEntrySignal(
            fast_base_period=fast_base_period,
            fast_src_type=fast_src_type,
            fast_fc=fast_fc,
            fast_sc=fast_sc,
            fast_mesa_fast_limit=fast_mesa_fast_limit,
            fast_mesa_slow_limit=fast_mesa_slow_limit,
            slow_base_period=slow_base_period,
            slow_src_type=slow_src_type,
            slow_fc=slow_fc,
            slow_sc=slow_sc,
            slow_mesa_fast_limit=slow_mesa_fast_limit,
            slow_mesa_slow_limit=slow_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            signal_mode=signal_mode,
            crossover_threshold=crossover_threshold
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._mesa_frama_signals = None
        self._last_fast_mesa_frama_values = None
        self._last_slow_mesa_frama_values = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data.copy()
                else:
                    # NumPy配列の場合、OHLCカラムでDataFrameを作成
                    if data.shape[1] >= 4:
                        df = pd.DataFrame(data[:, :4], columns=['open', 'high', 'low', 'close'])
                    else:
                        df = pd.DataFrame(data, columns=['close'])
                
                # デュアルMESA_FRAMAシグナルの計算
                try:
                    mesa_frama_signals = self.mesa_frama_entry_signal.generate(df)
                    
                    # エントリーシグナル
                    self._signals = mesa_frama_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._mesa_frama_signals = mesa_frama_signals
                    
                    # MESA_FRAMA値をキャッシュ（エグジット判定用）
                    fast_mesa_frama_values, slow_mesa_frama_values = self.mesa_frama_entry_signal.get_mesa_frama_values()
                    self._last_fast_mesa_frama_values = fast_mesa_frama_values
                    self._last_slow_mesa_frama_values = slow_mesa_frama_values
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._mesa_frama_signals = np.zeros(current_len, dtype=np.int8)
                    self._last_fast_mesa_frama_values = np.zeros(current_len, dtype=np.float64)
                    self._last_slow_mesa_frama_values = np.zeros(current_len, dtype=np.float64)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._mesa_frama_signals = np.zeros(len(data), dtype=np.int8)
                self._last_fast_mesa_frama_values = np.zeros(len(data), dtype=np.float64)
                self._last_slow_mesa_frama_values = np.zeros(len(data), dtype=np.float64)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # 境界チェック
        if index < 0 or index >= len(self._mesa_frama_signals):
            return False
        
        # シグナルモードに応じたエグジット判定
        if self._params['signal_mode'] == 'crossover':
            # クロスオーバーモード: 逆方向のクロスオーバーでエグジット
            if position == 1:  # ロングポジション
                return bool(self._mesa_frama_signals[index] == -1)
            elif position == -1:  # ショートポジション
                return bool(self._mesa_frama_signals[index] == 1)
        else:
            # 位置関係モード: 位置関係の逆転でエグジット
            try:
                if (self._last_fast_mesa_frama_values is not None and 
                    self._last_slow_mesa_frama_values is not None and
                    index < len(self._last_fast_mesa_frama_values) and
                    index < len(self._last_slow_mesa_frama_values)):
                    
                    fast_mesa_frama_val = self._last_fast_mesa_frama_values[index]
                    slow_mesa_frama_val = self._last_slow_mesa_frama_values[index]
                    
                    if not (np.isnan(fast_mesa_frama_val) or np.isnan(slow_mesa_frama_val)):
                        if position == 1:  # ロングポジション
                            return fast_mesa_frama_val < slow_mesa_frama_val
                        elif position == -1:  # ショートポジション
                            return fast_mesa_frama_val > slow_mesa_frama_val
            except Exception as e:
                self.logger.error(f"位置関係エグジット判定でエラー: {str(e)}")
        
        return False
    
    def get_mesa_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        両方のMESA_FRAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA_FRAMA値, 長期MESA_FRAMA値)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_mesa_frama_values()
        except Exception as e:
            self.logger.error(f"MESA_FRAMA値取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタル次元を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フラクタル次元, 長期フラクタル次元)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_fractal_dimension()
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA動的期間を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA動的期間, 長期MESA動的期間)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_dynamic_periods()
        except Exception as e:
            self.logger.error(f"MESA動的期間取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_mesa_phase(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA位相を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA位相, 長期MESA位相)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_mesa_phase()
        except Exception as e:
            self.logger.error(f"MESA位相取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタルアルファ値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フラクタルアルファ値, 長期フラクタルアルファ値)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_alpha_values()
        except Exception as e:
            self.logger.error(f"フラクタルアルファ値取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_filtered_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        カルマンフィルター後の価格を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フィルタリングされた価格, 長期フィルタリングされた価格)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mesa_frama_entry_signal.get_filtered_price()
        except Exception as e:
            self.logger.error(f"フィルタリング済み価格取得中にエラー: {str(e)}")
            return np.array([]), np.array([])