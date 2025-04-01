#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.c_channel.breakout_entry import CCBreakoutEntrySignal


class DualCCBreakoutSignalGenerator(BaseSignalGenerator):
    """
    デュアルCCブレイクアウトのシグナル生成クラス（両方向・高速化版）
    
    特徴:
    - 2つのCチャネルブレイクアウトシグナルを使用
    - 広めのバンド幅を持つシグナルをエントリーに使用
    - 狭めのバンド幅を持つシグナルをエグジットに使用
    
    エントリー条件:
    - ロング: 広めのCCBreakoutEntrySignalの買いシグナル（1）
    - ショート: 広めのCCBreakoutEntrySignalの売りシグナル（-1）
    
    エグジット条件:
    - ロング: 狭めのCCBreakoutEntrySignalの売りシグナル（-1）
    - ショート: 狭めのCCBreakoutEntrySignalの買いシグナル（1）
    """
    
    def __init__(
        self,
        # Cチャネルの基本パラメータ（共通）
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ（デフォルトではdetector_typeと同じ）
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.618,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # エントリー用の動的乗数の範囲パラメータ（広め）
        entry_max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        entry_min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        entry_max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        entry_min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # エグジット用の動的乗数の範囲パラメータ（狭め）
        exit_max_max_multiplier: float = 4.0,     # 最大乗数の最大値
        exit_min_max_multiplier: float = 3.0,     # 最大乗数の最小値
        exit_max_min_multiplier: float = 1.0,     # 最小乗数の最大値
        exit_min_min_multiplier: float = 0.3,     # 最小乗数の最小値
        
        # CMA用パラメータ
        cma_detector_type: str = 'hody_e',
        cma_cycle_part: float = 0.5,
        cma_lp_period: int = 5,
        cma_hp_period: int = 55,
        cma_max_cycle: int = 144,
        cma_min_cycle: int = 5,
        cma_max_output: int = 62,
        cma_min_output: int = 13,
        cma_fast_period: int = 2,
        cma_slow_period: int = 30,
        cma_src_type: str = 'hlc3',
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """初期化"""
        super().__init__("DualCCBreakoutSignalGenerator")
        
        # CER検出器タイプの初期化（None の場合は detector_type を使用）
        if cer_detector_type is None:
            cer_detector_type = detector_type
        
        # パラメータの設定
        self._params = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # エントリー用動的乗数
            'entry_max_max_multiplier': entry_max_max_multiplier,
            'entry_min_max_multiplier': entry_min_max_multiplier,
            'entry_max_min_multiplier': entry_max_min_multiplier,
            'entry_min_min_multiplier': entry_min_min_multiplier,
            
            # エグジット用動的乗数
            'exit_max_max_multiplier': exit_max_max_multiplier,
            'exit_min_max_multiplier': exit_min_max_multiplier,
            'exit_max_min_multiplier': exit_max_min_multiplier,
            'exit_min_min_multiplier': exit_min_min_multiplier,
            
            # CMA用パラメータ
            'cma_detector_type': cma_detector_type,
            'cma_cycle_part': cma_cycle_part,
            'cma_lp_period': cma_lp_period,
            'cma_hp_period': cma_hp_period,
            'cma_max_cycle': cma_max_cycle,
            'cma_min_cycle': cma_min_cycle,
            'cma_max_output': cma_max_output,
            'cma_min_output': cma_min_output,
            'cma_fast_period': cma_fast_period,
            'cma_slow_period': cma_slow_period,
            'cma_src_type': cma_src_type,
            
            # CATR用パラメータ
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type
        }
        
        # エントリー用CCブレイクアウトシグナル（広めのバンド）の初期化
        self.entry_signal = CCBreakoutEntrySignal(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            # 動的乗数の範囲パラメータ（広め）
            max_max_multiplier=entry_max_max_multiplier,
            min_max_multiplier=entry_min_max_multiplier,
            max_min_multiplier=entry_max_min_multiplier,
            min_min_multiplier=entry_min_min_multiplier,
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # エグジット用CCブレイクアウトシグナル（狭めのバンド）の初期化
        self.exit_signal = CCBreakoutEntrySignal(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            # 動的乗数の範囲パラメータ（狭め）
            max_max_multiplier=exit_max_max_multiplier,
            min_max_multiplier=exit_min_max_multiplier,
            max_min_multiplier=exit_max_min_multiplier,
            min_min_multiplier=exit_min_min_multiplier,
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._entry_signals is None or self._exit_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # エントリー用シグナルの計算
                    entry_signals = self.entry_signal.generate(df)
                    
                    # エグジット用シグナルの計算
                    exit_signals = self.exit_signal.generate(df)
                    
                    # シグナルの設定
                    self._entry_signals = entry_signals
                    self._exit_signals = exit_signals
                except Exception as e:
                    import traceback
                    print(f"シグナル計算中にエラー: {str(e)}\n{traceback.format_exc()}")
                    # エラー時はゼロシグナルを設定
                    self._entry_signals = np.zeros(current_len, dtype=np.int8)
                    self._exit_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            import traceback
            print(f"calculate_signals全体でエラー: {str(e)}\n{traceback.format_exc()}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                self._exit_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._exit_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._exit_signals[index] == 1)
        return False
    
    def get_entry_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        エントリー用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.entry_signal.get_band_values()
        except Exception as e:
            import traceback
            print(f"エントリーバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_exit_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        エグジット用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.exit_signal.get_band_values()
        except Exception as e:
            import traceback
            print(f"エグジットバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_entry_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー用サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.entry_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            import traceback
            print(f"エントリー効率比取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_exit_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エグジット用サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.exit_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            import traceback
            print(f"エグジット効率比取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_entry_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.entry_signal.get_dynamic_multiplier()
        except Exception as e:
            import traceback
            print(f"エントリー動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_exit_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エグジット用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.exit_signal.get_dynamic_multiplier()
        except Exception as e:
            import traceback
            print(f"エグジット動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得（エントリー用シグナルのCATRを使用）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.entry_signal.get_c_atr()
        except Exception as e:
            import traceback
            print(f"CATR取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([]) 