#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_channel.breakout_entry import ZChannelBreakoutEntrySignal
from signals.implementations.z_rsx.trigger import ZRSXTriggerSignal


class ZCRSXExitSignalGenerator(BaseSignalGenerator):
    """
    Zチャネル&RSXエグジットのシグナル生成クラス
    
    特徴:
    - Zチャネルによる両方向のエントリーシグナル生成
    - ロングポジション：Zチャネルの売りシグナルによるエグジット
    - ショートポジション：
      1. Zチャネルの買いシグナルによるエグジット（従来ロジック）
      2. RSXトリガーシグナルが買いシグナル（1）の場合にもエグジット（新ロジック）
    
    エントリー条件:
    - ロング: Zチャネルのブレイクアウトで買いシグナル
    - ショート: Zチャネルのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル
    - ショート: Zチャネルの買いシグナル または RSXトリガーの買いシグナル
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # RSXトリガーのパラメータ
        rsx_cycle_detector_type: str = 'hody_dc',
        rsx_lp_period: int = 13,
        rsx_hp_period: int = 144,
        rsx_cycle_part: float = 0.5,
        rsx_er_period: int = 10,
        
        # 最大ドミナントサイクル計算パラメータ
        max_dc_cycle_part: float = 0.5,
        max_dc_max_cycle: int = 55,
        max_dc_min_cycle: int = 5,
        max_dc_max_output: int = 34,
        max_dc_min_output: int = 14,
        
        # 最小ドミナントサイクル計算パラメータ
        min_dc_cycle_part: float = 0.25,
        min_dc_max_cycle: int = 34,
        min_dc_min_cycle: int = 3,
        min_dc_max_output: int = 13,
        min_dc_min_output: int = 3,
        
        # 買われすぎ/売られすぎレベルパラメータ
        min_high_level: float = 75.0,
        max_high_level: float = 85.0,
        min_low_level: float = 25.0,
        max_low_level: float = 15.0
    ):
        """初期化"""
        super().__init__("ZCRSXExitSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # Zチャネルのパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # RSXトリガーのパラメータ
            'rsx_cycle_detector_type': rsx_cycle_detector_type,
            'rsx_lp_period': rsx_lp_period,
            'rsx_hp_period': rsx_hp_period,
            'rsx_cycle_part': rsx_cycle_part,
            'rsx_er_period': rsx_er_period,
            
            'max_dc_cycle_part': max_dc_cycle_part,
            'max_dc_max_cycle': max_dc_max_cycle,
            'max_dc_min_cycle': max_dc_min_cycle,
            'max_dc_max_output': max_dc_max_output,
            'max_dc_min_output': max_dc_min_output,
            
            'min_dc_cycle_part': min_dc_cycle_part,
            'min_dc_max_cycle': min_dc_max_cycle,
            'min_dc_min_cycle': min_dc_min_cycle,
            'min_dc_max_output': min_dc_max_output,
            'min_dc_min_output': min_dc_min_output,
            
            'min_high_level': min_high_level,
            'max_high_level': max_high_level,
            'min_low_level': min_low_level,
            'max_low_level': max_low_level
        }
        
        # Zチャネルブレイクアウトシグナルの初期化
        self.z_channel_signal = ZChannelBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            lookback=band_lookback
        )
        
        # RSXトリガーシグナルの初期化
        self.rsx_trigger_signal = ZRSXTriggerSignal(
            cycle_detector_type=rsx_cycle_detector_type,
            lp_period=rsx_lp_period,
            hp_period=rsx_hp_period,
            cycle_part=rsx_cycle_part,
            er_period=rsx_er_period,
            
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=max_dc_max_output,
            max_dc_min_output=max_dc_min_output,
            
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=min_dc_max_output,
            min_dc_min_output=min_dc_min_output,
            
            min_high_level=min_high_level,
            max_high_level=max_high_level,
            min_low_level=min_low_level,
            max_low_level=max_low_level
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_channel_signals = None
        self._rsx_trigger_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if (self._signals is None or self._z_channel_signals is None or 
                self._rsx_trigger_signals is None or current_len != self._data_len):
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # Zチャネルシグナルの計算
                    z_channel_signals = self.z_channel_signal.generate(df)
                    
                    # RSXトリガーシグナルの計算
                    rsx_trigger_signals = self.rsx_trigger_signal.generate(df)
                    
                    # エントリーシグナルはZチャネルシグナルと同じ
                    self._signals = z_channel_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._z_channel_signals = z_channel_signals
                    self._rsx_trigger_signals = rsx_trigger_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._rsx_trigger_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._rsx_trigger_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナル生成（高速化版）
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション - 通常のZチャネルエグジット
            return bool(self._z_channel_signals[index] == -1) or (self._rsx_trigger_signals[index] == -1)
        elif position == -1:  # ショートポジション - Zチャネル買いシグナル または RSXトリガー買いシグナル
            return bool(self._z_channel_signals[index] == 1 or self._rsx_trigger_signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
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
                
            return self.z_channel_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
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
                
            return self.z_channel_signal.get_dynamic_multiplier()
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
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
                
            return self.z_channel_signal.get_z_atr()
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_rsx_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZRSX値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZRSX値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.rsx_trigger_signal.get_rsx_values()
        except Exception as e:
            self.logger.error(f"ZRSX値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_rsx_levels(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZRSXの高値/安値レベルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (高値レベル, 安値レベル)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.rsx_trigger_signal.get_levels()
        except Exception as e:
            self.logger.error(f"RSXレベル取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty
    
    def get_rsx_trigger_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        RSXトリガーシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: RSXトリガーシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._rsx_trigger_signals
        except Exception as e:
            self.logger.error(f"RSXトリガーシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_z_channel_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zチャネルシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Zチャネルシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._z_channel_signals
        except Exception as e:
            self.logger.error(f"Zチャネルシグナル取得中にエラー: {str(e)}")
            return np.array([]) 