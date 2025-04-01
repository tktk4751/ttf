#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_trend import ZTrendBreakoutEntrySignal


class ZTSimpleSignalGenerator(BaseSignalGenerator):
    """
    Zトレンドのシグナル生成クラス（両方向・高速化版）- トレンドフィルターなし
    
    エントリー条件:
    - ロング: Zトレンドのブレイクアウトで買いシグナル
    - ショート: Zトレンドのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: Zトレンドの売りシグナル
    - ショート: Zトレンドの買いシグナル
    """
    
    def __init__(
        self,
        # Zトレンドのパラメータ
        cycle_detector_type: str = 'dudi_dc',
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.5,
        band_lookback: int = 1,
        
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
        
        # 動的乗数の範囲
        max_max_multiplier: float = 5.0,    # 最大乗数の最大値
        min_max_multiplier: float = 2.5,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """初期化"""
        super().__init__("ZTSimpleSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'band_lookback': band_lookback,
            
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
            
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        
        # Zトレンドブレイクアウトシグナルの初期化
        self.z_trend_signal = ZTrendBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            lookback=band_lookback,
            
            # CERのドミナントサイクル検出器用パラメータ
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            
            # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            
            # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            
            # ZATR用ドミナントサイクル検出器のパラメータ
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
            
            # パーセンタイル乗数
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            
            # 動的乗数の範囲
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # その他の設定
            smoother_type=smoother_type,
            src_type=src_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_trend_signals = None
    
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
                
                # Zトレンドシグナルの計算
                try:
                    z_trend_signals = self.z_trend_signal.generate(df)
                    
                    # Zトレンドフィルターなしの単純なシグナル
                    self._signals = z_trend_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._z_trend_signals = z_trend_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_trend_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._z_trend_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._z_trend_signals[index] == 1)
        return False
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zトレンドのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_signal.get_bands()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_signal.get_trend()
        except Exception as e:
            self.logger.error(f"トレンド方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_percentiles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        パーセンタイル値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上側パーセンタイル, 下側パーセンタイル)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_signal.get_percentiles()
        except Exception as e:
            self.logger.error(f"パーセンタイル値取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty
    
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
                
            return self.z_trend_signal.get_cycle_er()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_parameters(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (動的乗数, 動的パーセンタイル期間)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_signal.get_dynamic_parameters()
        except Exception as e:
            self.logger.error(f"動的パラメータ取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty
    
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
                
            return self.z_trend_signal.get_z_atr()
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dominant_cycles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ドミナントサイクルの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最大パーセンタイル期間用ドミナントサイクル, 最小パーセンタイル期間用ドミナントサイクル)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_signal.get_dominant_cycles()
        except Exception as e:
            self.logger.error(f"ドミナントサイクル取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty 