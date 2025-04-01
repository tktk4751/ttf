#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit
import logging

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_ma.direction import AlphaMADirectionSignal2
from signals.implementations.divergence.alpha_roc_divergence import AlphaROCDivergenceSignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal
from signals.implementations.bollinger.exit import BollingerBreakoutExitSignal

# ロガーの設定
logger = logging.getLogger(__name__)

@jit(nopython=True)
def calculate_entry_signals(
    alpha_ma_signals: np.ndarray, 
    alpha_roc_div_signals: np.ndarray, 
    filter_signals: np.ndarray
) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_ma_signals, dtype=np.int8)
    
    # ロングエントリー: AlphaMADirectionSignal2が-1 + AlphaROCDivergenceSignalが1 + AlphaFilterSignalが1
    long_condition = (alpha_ma_signals == -1) & (alpha_roc_div_signals == 1) & (filter_signals == 1)
    
    # ショートエントリー: AlphaMADirectionSignal2が1 + AlphaROCDivergenceSignalが-1 + AlphaFilterSignalが1
    short_condition = (alpha_ma_signals == 1) & (alpha_roc_div_signals == -1) & (filter_signals == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaROCDivergenceSignalGenerator(BaseSignalGenerator):
    """
    AlphaROCダイバージェンス戦略のシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: AlphaMADirectionSignal2が-1 + AlphaROCDivergenceSignalが1 + AlphaFilterSignalが1
    - ショート: AlphaMADirectionSignal2が1 + AlphaROCDivergenceSignalが-1 + AlphaFilterSignalが1
    
    エグジット条件:
    - ロング: BollingerBreakoutExitSignalが1 または AlphaROCDivergenceSignalが-1
    - ショート: BollingerBreakoutExitSignalが-1 または AlphaROCDivergenceSignalが1
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaMA用パラメータ
        max_ma_period: int = 200,
        min_ma_period: int = 20,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        # AlphaROC用パラメータ
        max_roc_period: int = 50,
        min_roc_period: int = 5,
        lookback: int = 30,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5,
        # ボリンジャーバンド用パラメータ
        bb_period: int = 20,
        bb_std_dev: float = 2.0
    ):
        """初期化"""
        super().__init__("AlphaROCDivergenceSignalGenerator")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'max_ma_period': max_ma_period,
            'min_ma_period': min_ma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'max_roc_period': max_roc_period,
            'min_roc_period': min_roc_period,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold,
            'bb_period': bb_period,
            'bb_std_dev': bb_std_dev
        }
        \
        
        
        # AlphaMADirectionSignal2の初期化
        self.alpha_ma_signal = AlphaMADirectionSignal2(
            er_period=er_period,
            max_kama_period=max_ma_period,
            min_kama_period=min_ma_period,
        )
        
        # AlphaROCDivergenceSignalの初期化
        self.alpha_roc_div_signal = AlphaROCDivergenceSignal(
            er_period=er_period,
            max_roc_period=max_roc_period,
            min_roc_period=min_roc_period,
            lookback=lookback
        )
        
        # AlphaFilterSignalの初期化
        self.alpha_filter_signal = AlphaFilterSignal(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_period=10,  # 固定値
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            solid={
                'threshold': filter_threshold
            }
        )
        
        # BollingerBreakoutExitSignalの初期化
        self.bollinger_exit_signal = BollingerBreakoutExitSignal(
            period=bb_period,
            num_std=bb_std_dev
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_ma_signals = None
        self._alpha_roc_div_signals = None
        self._bollinger_exit_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    alpha_ma_signals = self.alpha_ma_signal.generate(df)
                    alpha_roc_div_signals = self.alpha_roc_div_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    bollinger_exit_signals = self.bollinger_exit_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(
                        alpha_ma_signals, 
                        alpha_roc_div_signals, 
                        filter_signals
                    )
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_ma_signals = alpha_ma_signals
                    self._alpha_roc_div_signals = alpha_roc_div_signals
                    self._bollinger_exit_signals = bollinger_exit_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_ma_signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_roc_div_signals = np.zeros(current_len, dtype=np.int8)
                    self._bollinger_exit_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_ma_signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_roc_div_signals = np.zeros(len(data), dtype=np.int8)
                self._bollinger_exit_signals = np.zeros(len(data), dtype=np.int8)
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
            # BollingerBreakoutExitSignalが1 または AlphaROCDivergenceSignalが-1
            return bool(self._bollinger_exit_signals[index] == 1 or self._alpha_roc_div_signals[index] == -1)
        elif position == -1:  # ショートポジション
            # BollingerBreakoutExitSignalが-1 または AlphaROCDivergenceSignalが1
            return bool(self._bollinger_exit_signals[index] == -1 or self._alpha_roc_div_signals[index] == 1)
        return False
    
    def get_alpha_ma_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaMAシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaMAシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_ma_signals
        except Exception as e:
            self.logger.error(f"AlphaMAシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_roc_div_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaROCダイバージェンスシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaROCダイバージェンスシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_roc_div_signals
        except Exception as e:
            self.logger.error(f"AlphaROCダイバージェンスシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファフィルターの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_filter_signal.get_filter_values()
        except Exception as e:
            self.logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_bollinger_exit_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ボリンジャーバンドエグジットシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ボリンジャーバンドエグジットシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._bollinger_exit_signals
        except Exception as e:
            self.logger.error(f"ボリンジャーバンドエグジットシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_roc_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaROCの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaROCの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_roc_div_signal.get_alpha_roc_values(data)
        except Exception as e:
            self.logger.error(f"AlphaROC値取得中にエラー: {str(e)}")
            return np.array([]) 