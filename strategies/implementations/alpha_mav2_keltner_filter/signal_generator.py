#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_mav2_keltner.breakout_entry import AlphaMAV2KeltnerBreakoutEntrySignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal


@jit(nopython=True)
def calculate_entry_signals(alpha_mav2_keltner: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_mav2_keltner, dtype=np.int8)
    
    # ロングエントリー: アルファMAV2ケルトナーの買いシグナル + アルファフィルターがトレンド相場
    long_condition = (alpha_mav2_keltner == 1) & (filter_signal == 1)
    
    # ショートエントリー: アルファMAV2ケルトナーの売りシグナル + アルファフィルターがトレンド相場
    short_condition = (alpha_mav2_keltner == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaMAV2KeltnerFilterSignalGenerator(BaseSignalGenerator):
    """
    アルファMAV2ケルトナーチャネル+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファMAV2ケルトナーのブレイクアウトで買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファMAV2ケルトナーのブレイクアウトで売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファMAV2ケルトナーの売りシグナル
    - ショート: アルファMAV2ケルトナーの買いシグナル
    
    特徴:
    - RSXの3段階平滑化を採用したAlphaMAV2を中心線として使用
    - AlphaATRによるボラティリティに適応したバンド幅
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファフィルターによる市場状態フィルタリング
    """
    
    def __init__(
        self,
        ma_er_period: int = 10,
        ma_max_period: int = 34,
        ma_min_period: int = 5,
        atr_er_period: int = 21,
        atr_max_period: int = 89,
        atr_min_period: int = 13,
        max_keltner_multiplier: float = 2.0,
        min_keltner_multiplier: float = 2.0,
        lookback: int = 1,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5
    ):
        """初期化"""
        super().__init__("AlphaMAV2KeltnerFilterSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'ma_er_period': ma_er_period,
            'ma_max_period': ma_max_period,
            'ma_min_period': ma_min_period,
            'atr_er_period': atr_er_period,
            'atr_max_period': atr_max_period,
            'atr_min_period': atr_min_period,
            'max_keltner_multiplier': max_keltner_multiplier,
            'min_keltner_multiplier': min_keltner_multiplier,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold
        }
        
        # アルファMAV2ケルトナーブレイクアウトシグナルの初期化
        self.alpha_mav2_keltner_signal = AlphaMAV2KeltnerBreakoutEntrySignal(
            ma_er_period=ma_er_period,
            ma_max_period=ma_max_period,
            ma_min_period=ma_min_period,
            atr_er_period=atr_er_period,
            atr_max_period=atr_max_period,
            atr_min_period=atr_min_period,
            upper_multiplier=max_keltner_multiplier,
            lower_multiplier=min_keltner_multiplier,
            lookback=lookback
        )
        
        # アルファフィルターシグナルの初期化
        self.alpha_filter_signal = AlphaFilterSignal(
            er_period=atr_er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_mav2_keltner_signals = None
    
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
                    alpha_mav2_keltner_signals = self.alpha_mav2_keltner_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_mav2_keltner_signals, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_mav2_keltner_signals = alpha_mav2_keltner_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_mav2_keltner_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_mav2_keltner_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._alpha_mav2_keltner_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_mav2_keltner_signals[index] == 1)
        return False
    
    def get_keltner_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ケルトナーチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上部バンド, 下部バンド, 中間上部バンド, 中間下部バンド)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_mav2_keltner_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"ケルトナーバンド取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty, empty, empty
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的期間の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (AlphaMAV2の動的期間, AlphaATRの動的期間)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_mav2_keltner_signal.get_dynamic_periods()
        except Exception as e:
            self.logger.error(f"動的期間取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaATRの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_mav2_keltner_signal.get_atr()
        except Exception as e:
            self.logger.error(f"ATR取得中にエラー: {str(e)}")
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
        
    def get_filter_components(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        アルファフィルターのコンポーネント値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: コンポーネント値のディクショナリ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_filter_signal.get_component_values()
        except Exception as e:
            self.logger.error(f"フィルターコンポーネント取得中にエラー: {str(e)}")
            # エラー時は空のディクショナリを返す
            return {'er': np.array([]), 'alpha_chop': np.array([]), 'alpha_adx': np.array([]), 'dynamic_period': np.array([])} 