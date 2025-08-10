#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.x_fama.entry import XFAMACrossoverEntrySignal


class XFAMASignalGenerator(BaseSignalGenerator):
    """
    X_FAMAシグナル生成クラス
    
    エントリー条件:
    - クロスオーバーモード: ゴールデンクロス（Fast X_FAMA > X_FAMAに転換）でロング、デッドクロス（Fast X_FAMA < X_FAMAに転換）でショート
    - 位置関係モード: Fast X_FAMA > X_FAMAでロング、Fast X_FAMA < X_FAMAでショート
    - フラクタルモード: フラクタル次元とアルファ値によるトレンド強度でフィルタリング
    
    エグジット条件:
    - ロング: Fast X_FAMA < X_FAMAに転換
    - ショート: Fast X_FAMA > X_FAMAに転換
    """
    
    def __init__(
        self,
        # X_FAMAパラメータ
        period: int = 16,                      # 期間（偶数である必要がある）
        src_type: str = 'hl2',                 # ソースタイプ
        fc: int = 1,                           # Fast Constant
        sc: int = 198,                         # Slow Constant
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = False,           # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        fractal_mode: bool = False,            # フラクタル次元ベースのシグナル追加
        trend_threshold: float = 1.5,          # フラクタル次元のトレンド閾値
        alpha_threshold: float = 0.5           # アルファ値の閾値
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            fractal_mode: フラクタル次元ベースのシグナル追加
            trend_threshold: フラクタル次元のトレンド閾値
            alpha_threshold: アルファ値の閾値
        """
        signal_type = "Position" if position_mode else "Crossover"
        fractal_str = "_fractal" if fractal_mode else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_FAMA{signal_type}SignalGenerator{fractal_str}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._params = {
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
            'fractal_mode': fractal_mode,
            'trend_threshold': trend_threshold,
            'alpha_threshold': alpha_threshold
        }
        
        self.position_mode = position_mode
        self.fractal_mode = fractal_mode
        
        # X_FAMAエントリーシグナルの初期化
        self.x_fama_entry_signal = XFAMACrossoverEntrySignal(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode,
            fractal_mode=fractal_mode,
            trend_threshold=trend_threshold,
            alpha_threshold=alpha_threshold
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._x_fama_signals = None
    
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
                
                # X_FAMAシグナルの計算
                try:
                    x_fama_signals = self.x_fama_entry_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = x_fama_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._x_fama_signals = x_fama_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._x_fama_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._x_fama_signals = np.zeros(len(data), dtype=np.int8)
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
        if index < 0 or index >= len(self._x_fama_signals):
            return False
        
        # キャッシュされたシグナルを使用
        if self.position_mode:
            # 位置関係モードでは、現在のポジションと逆のシグナルが出たらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_fama_signals[index] == -1)
            elif position == -1:  # ショートポジション
                return bool(self._x_fama_signals[index] == 1)
        else:
            # クロスオーバーモードでは、逆方向のクロスオーバーが発生したらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_fama_signals[index] == -1)  # デッドクロス
            elif position == -1:  # ショートポジション
                return bool(self._x_fama_signals[index] == 1)   # ゴールデンクロス
        
        return False
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FRAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_FRAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_fama_entry_signal.get_frama_values()
        except Exception as e:
            self.logger.error(f"X_FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fast_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Fast X_FRAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Fast X_FRAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_fama_entry_signal.get_fast_fama_values()
        except Exception as e:
            self.logger.error(f"Fast X_FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_fama_entry_signal.get_fractal_dimension()
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファ値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルファ値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_fama_entry_signal.get_alpha_values()
        except Exception as e:
            self.logger.error(f"アルファ値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filtered_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カルマンフィルター後の価格を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルタリングされた価格
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_fama_entry_signal.get_filtered_price()
        except Exception as e:
            self.logger.error(f"フィルタリング価格取得中にエラー: {str(e)}")
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
                'frama_values': self.get_frama_values(),
                'fast_fama_values': self.get_fast_fama_values(),
                'fractal_dimension': self.get_fractal_dimension(),
                'alpha_values': self.get_alpha_values(),
                'filtered_price': self.get_filtered_price()
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
        self._x_fama_signals = None
        if hasattr(self.x_fama_entry_signal, 'reset'):
            self.x_fama_entry_signal.reset()