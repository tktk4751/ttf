#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.mama.entry import MAMACrossoverEntrySignal


class MAMASignalGenerator(BaseSignalGenerator):
    """
    MAMAシグナル生成クラス
    
    エントリー条件:
    - ロング: MAMA > FAMA
    - ショート: MAMA < FAMA
    
    エグジット条件:
    - ロング: MAMA < FAMA
    - ショート: MAMA > FAMA
    """
    
    def __init__(
        self,
        # MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        ukf_params: Optional[Dict] = None      # UKFパラメータ
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
        """
        super().__init__("MAMASignalGenerator")
        
        # パラメータの設定
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'ukf_params': ukf_params
        }
        
        # MAMAエントリーシグナルの初期化
        self.mama_entry_signal = MAMACrossoverEntrySignal(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            ukf_params=ukf_params
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._mama_signals = None
    
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
                
                # MAMAシグナルの計算
                try:
                    mama_signals = self.mama_entry_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = mama_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._mama_signals = mama_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._mama_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._mama_signals = np.zeros(len(data), dtype=np.int8)
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
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._mama_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._mama_signals[index] == 1)
        return False
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: MAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mama_entry_signal.get_mama_values()
        except Exception as e:
            self.logger.error(f"MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.mama_entry_signal.get_fama_values()
        except Exception as e:
            self.logger.error(f"FAMA値取得中にエラー: {str(e)}")
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
                
            return self.mama_entry_signal.get_period_values()
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
                
            return self.mama_entry_signal.get_alpha_values()
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])