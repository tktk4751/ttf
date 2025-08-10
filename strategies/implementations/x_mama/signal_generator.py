#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.x_mama.entry import XMAMACrossoverEntrySignal


class XMAMASignalGenerator(BaseSignalGenerator):
    """
    X_MAMAシグナル生成クラス
    
    エントリー条件:
    - クロスオーバーモード: ゴールデンクロス（X_MAMA > X_FAMAに転換）でロング、デッドクロス（X_MAMA < X_FAMAに転換）でショート
    - 位置関係モード: X_MAMA > X_FAMAでロング、X_MAMA < X_FAMAでショート
    
    エグジット条件:
    - ロング: X_MAMA < X_FAMAに転換
    - ショート: X_MAMA > X_FAMAに転換
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = False            # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        """
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_MAMA{signal_type}SignalGenerator{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode
        }
        
        self.position_mode = position_mode
        
        # X_MAMAエントリーシグナルの初期化
        self.x_mama_entry_signal = XMAMACrossoverEntrySignal(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._x_mama_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # X_MAMAシグナルの計算
                try:
                    x_mama_signals = self.x_mama_entry_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = x_mama_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._x_mama_signals = x_mama_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._x_mama_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._x_mama_signals = np.zeros(len(data), dtype=np.int8)
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
        
        # インデックスの範囲チェック
        if index < 0 or index >= len(self._x_mama_signals):
            return False
        
        # キャッシュされたシグナルを使用
        if self.position_mode:
            # 位置関係モードでは、現在のポジションと逆のシグナルが出たらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_mama_signals[index] == -1)
            elif position == -1:  # ショートポジション
                return bool(self._x_mama_signals[index] == 1)
        else:
            # クロスオーバーモードでは、逆方向のクロスオーバーが発生したらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_mama_signals[index] == -1)  # デッドクロス
            elif position == -1:  # ショートポジション
                return bool(self._x_mama_signals[index] == 1)   # ゴールデンクロス
        
        return False
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_MAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_MAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_mama_values()
        except Exception as e:
            self.logger.error(f"X_MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_FAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_fama_values()
        except Exception as e:
            self.logger.error(f"X_FAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_period_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Period値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Period値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_period_values()
        except Exception as e:
            self.logger.error(f"Period値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_alpha_values()
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_phase_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Phase値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Phase値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_phase_values()
        except Exception as e:
            self.logger.error(f"Phase値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_i1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        I1値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: I1値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_i1_values()
        except Exception as e:
            self.logger.error(f"I1値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_q1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Q1値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Q1値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mama_entry_signal.get_q1_values()
        except Exception as e:
            self.logger.error(f"Q1値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: 全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return {
                'mama_values': self.get_mama_values(),
                'fama_values': self.get_fama_values(),
                'period_values': self.get_period_values(),
                'alpha_values': self.get_alpha_values(),
                'phase_values': self.get_phase_values(),
                'i1_values': self.get_i1_values(),
                'q1_values': self.get_q1_values()
            }
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """
        シグナルジェネレーターの状態をリセット
        """
        super().reset()
        self._data_len = 0
        self._signals = None
        self._x_mama_signals = None
        if hasattr(self.x_mama_entry_signal, 'reset'):
            self.x_mama_entry_signal.reset()