#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_ma.direction import AlphaMACirculationSignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal
from signals.implementations.divergence.alpha_macd_divergence import AlphaMACDDivergenceSignal


@jit(nopython=True)
def calculate_entry_signals(
    stages: np.ndarray,
    filter_signals: np.ndarray,
    divergence_signals: np.ndarray
) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）
    
    Args:
        stages: AlphaMACirculationSignalのステージ配列
        filter_signals: AlphaFilterSignalのシグナル配列
        divergence_signals: AlphaMACDDivergenceSignalのシグナル配列
        
    Returns:
        np.ndarray: エントリーシグナル配列
    """
    length = len(stages)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # ステージ1: 短期 > 中期 > 長期（安定上昇相場）+ AlphaFilterSignal が1でロング
        if stages[i] == 1 and filter_signals[i] == 1:
            signals[i] = 1  # ロングエントリー
        
        # ステージ2: 中期 > 短期 > 長期（上昇相場の終焉）+ AlphaMACDDivergenceSignal が-1でショート
        elif stages[i] == 2 and divergence_signals[i] == -1:
            signals[i] = -1  # ショートエントリー
        
        # ステージ4: 長期 > 中期 > 短期（安定下降相場）+ AlphaFilterSignal が-1でショート
        elif stages[i] == 4 and filter_signals[i] == -1:
            signals[i] = -1  # ショートエントリー
        
        # ステージ6: 短期 > 長期 > 中期（上昇相場の入口）+ AlphaMACDDivergenceSignal が1でロング
        elif stages[i] == 6 and divergence_signals[i] == 1:
            signals[i] = 1  # ロングエントリー
    
    return signals


@jit(nopython=True)
def calculate_exit_signal(
    stage: int,
    position: int
) -> bool:
    """エグジットシグナルを計算（高速化版）
    
    Args:
        stage: 現在のステージ
        position: 現在のポジション（1: ロング、-1: ショート）
        
    Returns:
        bool: エグジットすべきかどうか
    """
    # ロングポジションの場合
    if position == 1:
        # ステージ3: 中期 > 長期 > 短期（下降相場の入口）でロングエグジット
        return stage == 3
    
    # ショートポジションの場合
    elif position == -1:
        # ステージ5: 長期 > 短期 > 中期（下降相場の終焉）でショートエグジット
        return stage == 5
    
    # ポジションがない場合
    return False


class AlphaCirculationSignalGenerator(BaseSignalGenerator):
    """
    アルファ循環戦略のシグナル生成クラス
    
    エントリー条件:
    - ステージ1: 短期 > 中期 > 長期（安定上昇相場）+ AlphaFilterSignal が1でロング
    - ステージ2: 中期 > 短期 > 長期（上昇相場の終焉）+ AlphaMACDDivergenceSignal が-1でショート
    - ステージ4: 長期 > 中期 > 短期（安定下降相場）+ AlphaFilterSignal が-1でショート
    - ステージ6: 短期 > 長期 > 中期（上昇相場の入口）+ AlphaMACDDivergenceSignal が1でロング
    
    エグジット条件:
    - ステージ3: 中期 > 長期 > 短期（下降相場の入口）でロングエグジット
    - ステージ5: 長期 > 短期 > 中期（下降相場の終焉）でショートエグジット
    """
    
    def __init__(
        self,
        # AlphaMACirculationSignalのパラメータ
        er_period: int = 21,
        short_max_kama_period: int = 55,
        short_min_kama_period: int = 3,
        middle_max_kama_period: int = 144,
        middle_min_kama_period: int = 21,
        long_max_kama_period: int = 377,
        long_min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        
        # AlphaFilterSignalのパラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        alma_period: int = 10,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        filter_threshold: float = 0.5,
        
        # AlphaMACDDivergenceSignalのパラメータ
        fast_max_kama_period: int = 89,
        fast_min_kama_period: int = 8,
        slow_max_kama_period: int = 144,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 55,
        signal_min_kama_period: int = 5,
        lookback: int = 30
    ):
        """初期化"""
        super().__init__("AlphaCirculationSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # AlphaMACirculationSignalのパラメータ
            'er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'middle_max_kama_period': middle_max_kama_period,
            'middle_min_kama_period': middle_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            
            # AlphaFilterSignalのパラメータ
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_period': alma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold,
            
            # AlphaMACDDivergenceSignalのパラメータ
            'fast_max_kama_period': fast_max_kama_period,
            'fast_min_kama_period': fast_min_kama_period,
            'slow_max_kama_period': slow_max_kama_period,
            'slow_min_kama_period': slow_min_kama_period,
            'signal_max_kama_period': signal_max_kama_period,
            'signal_min_kama_period': signal_min_kama_period,
            'lookback': lookback
        }
        
        # AlphaMACirculationSignalの初期化
        self.alpha_ma_circulation = AlphaMACirculationSignal(
            er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            middle_max_kama_period=middle_max_kama_period,
            middle_min_kama_period=middle_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        # AlphaFilterSignalの初期化
        self.alpha_filter = AlphaFilterSignal(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_period=alma_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            solid={
                'threshold': filter_threshold
            }
        )
        
        # AlphaMACDDivergenceSignalの初期化
        self.alpha_macd_divergence = AlphaMACDDivergenceSignal(
            er_period=er_period,
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=lookback
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._stages = None
    
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
                    # ステージの計算
                    self._stages = self.alpha_ma_circulation.get_stage(df)
                    
                    # フィルターシグナルの計算
                    filter_signals = self.alpha_filter.generate(df)
                    
                    # ダイバージェンスシグナルの計算
                    divergence_signals = self.alpha_macd_divergence.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(
                        self._stages,
                        filter_signals,
                        divergence_signals
                    )
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._stages = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._stages = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._stages is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # 現在のステージを取得
        current_stage = self._stages[index]
        
        # エグジットシグナルの計算（高速化版）
        return calculate_exit_signal(current_stage, position)
    
    def get_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ステージの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ステージの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._stages
        except Exception as e:
            self.logger.error(f"ステージ取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
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
                
            return self.alpha_filter.get_filter_values()
        except Exception as e:
            self.logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_divergence_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        アルファMACDダイバージェンスの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: アルファMACDの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_macd_divergence.get_alpha_macd_values(data)
        except Exception as e:
            self.logger.error(f"ダイバージェンス値取得中にエラー: {str(e)}")
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])} 