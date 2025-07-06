#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_channel.breakout_entry import UltimateChannelBreakoutEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_exit_signals_numba(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, 
                                current_position: int, current_index: int) -> bool:
    """
    決済シグナル計算（Numba高速化版）
    
    Args:
        close: 終値の配列
        upper: 上部チャネルの配列
        lower: 下部チャネルの配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(close) or current_index >= len(upper) or current_index >= len(lower):
        return False
    
    # NaN値チェック
    if (np.isnan(close[current_index]) or np.isnan(upper[current_index]) or 
        np.isnan(lower[current_index])):
        return False
    
    # ロングポジション決済: 終値が下部チャネル以下
    if current_position == 1:
        return close[current_index] <= lower[current_index]
    # ショートポジション決済: 終値が上部チャネル以上
    elif current_position == -1:
        return close[current_index] >= upper[current_index]
    
    return False


class UltimateChannelBreakoutSignalGenerator(BaseSignalGenerator):
    """
    Ultimate Channel Breakoutのシグナル生成クラス（高速化版）
    
    エントリー条件:
    - ロング: 前回終値が前回の上部チャネル以下かつ現在終値が現在の上部チャネル以上
    - ショート: 前回終値が前回の下部チャネル以上かつ現在終値が現在の下部チャネル以下
    
    エグジット条件:
    - ロング決済: 終値が下部チャネル以下
    - ショート決済: 終値が上部チャネル以上
    """
    
    def __init__(
        self,
        # 基本パラメータ
        channel_lookback: int = 1,
        
        # アルティメットチャネルのパラメータ
        length: float = 20.0,
        num_strs: float = 2.0,
        multiplier_mode: str = 'fixed',
        src_type: str = 'hlc3',
        # 追加のアルティメットチャネルパラメータ
        ultimate_channel_params: Dict[str, Any] = None
    ):
        """初期化"""
        super().__init__("UltimateChannelBreakoutSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'channel_lookback': channel_lookback,
            'length': length,
            'num_strs': num_strs,
            'multiplier_mode': multiplier_mode,
            'src_type': src_type,
            **(ultimate_channel_params or {})
        }
        
        # Ultimate Channel Breakoutエントリーシグナルの初期化
        # channel_lookbackを除外したパラメータを作成
        ultimate_channel_params = {k: v for k, v in self._params.items() if k != 'channel_lookback'}
        self.ultimate_channel_signal = UltimateChannelBreakoutEntrySignal(
            channel_lookback=channel_lookback,
            ultimate_channel_params=ultimate_channel_params
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._close = None
        self._upper = None
        self._lower = None
        self._centerline = None
        self._ultimate_channel_result = None
    
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
                
                # Ultimate Channel Breakoutシグナルの計算
                try:
                    # エントリーシグナルの計算
                    ultimate_channel_signals = self.ultimate_channel_signal.generate(df)
                    
                    # チャネル値を取得
                    self._centerline, self._upper, self._lower = self.ultimate_channel_signal.get_channel_values(df)
                    
                    # 終値を取得
                    if isinstance(data, pd.DataFrame):
                        self._close = data['close'].values
                    else:
                        self._close = data[:, 3]  # close価格のインデックスは3
                    
                    # エントリーシグナルを設定
                    self._signals = ultimate_channel_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate Channel Breakoutシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._close = np.zeros(current_len, dtype=np.float64)
                    self._upper = np.zeros(current_len, dtype=np.float64)
                    self._lower = np.zeros(current_len, dtype=np.float64)
                    self._centerline = np.zeros(current_len, dtype=np.float64)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._close = np.zeros(len(data), dtype=np.float64)
                self._upper = np.zeros(len(data), dtype=np.float64)
                self._lower = np.zeros(len(data), dtype=np.float64)
                self._centerline = np.zeros(len(data), dtype=np.float64)
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
        
        # Numba高速化された決済シグナル計算
        if (self._close is not None and self._upper is not None and 
            self._lower is not None):
            return calculate_exit_signals_numba(
                self._close, 
                self._upper, 
                self._lower, 
                position, 
                index
            )
        
        return False
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルティメットチャネルのチャネル値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上部チャネル, 下部チャネル)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if (self._centerline is not None and self._upper is not None and 
                self._lower is not None):
                return self._centerline.copy(), self._upper.copy(), self._lower.copy()
            else:
                return np.array([]), np.array([]), np.array([])
        except Exception as e:
            self.logger.error(f"チャネル値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_dynamic_multipliers(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_channel_signal.get_dynamic_multipliers(data)
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_uqatrd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRD値の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRD値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_channel_signal.get_uqatrd_values(data)
        except Exception as e:
            self.logger.error(f"UQATRD値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_multiplier_mode(self, data: Union[pd.DataFrame, np.ndarray] = None) -> str:
        """
        乗数モードを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            str: 乗数モード ('fixed' or 'dynamic')
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_channel_signal.get_multiplier_mode(data)
        except Exception as e:
            self.logger.error(f"乗数モード取得中にエラー: {str(e)}")
            return 'fixed'
    
    def get_channel_statistics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict:
        """
        チャネル統計情報を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict: チャネル統計情報
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_channel_signal.get_channel_statistics(data)
        except Exception as e:
            self.logger.error(f"チャネル統計取得中にエラー: {str(e)}")
            return {"status": "no_data"} 