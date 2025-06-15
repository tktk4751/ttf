#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_adaptive_flow.trend_change_signal import ZAdaptiveFlowTrendChangeSignal


class ZAdaptiveFlowTrendSignalGenerator(BaseSignalGenerator):
    """
    Z Adaptive Flowトレンドチェンジシグナルベースのシグナルジェネレーターを作成する

    エントリー条件:
    - ロング: ZAdaptiveFlowのトレンドチェンジシグナルが1
    - ショート: ZAdaptiveFlowのトレンドチェンジシグナルが-1
    
    エグジット条件:
    - ロング: ZAdaptiveFlowのトレンドチェンジシグナルが-1
    - ショート: ZAdaptiveFlowのトレンドチェンジシグナルが1
    """
    
    def __init__(
        self,
        # 基本パラメータ
        length: int = 10,
        smooth_length: int = 14,
        src_type: str = 'hlc3',
        
        # MAタイプ選択
        ma_type: str = 'zlema',
        
        # MA共通パラメータ
        ma_period: int = None,
        ma_use_dynamic_period: bool = True,
        ma_slope_index: int = 1,
        ma_range_threshold: float = 0.005,
        
        # ボラティリティタイプ選択
        volatility_type: str = 'volatility',
        
        # 固定乗数パラメータ
        slow_period_multiplier: float = 4.0,
        volatility_multiplier: float = 2.0,
        
        # 共通サイクル検出器パラメータ
        detector_type: str = 'cycle_period2',
        cycle_part: float = 1.0,
        lp_period: int = 10,
        hp_period: int = 48,
        max_cycle: int = 233,
        min_cycle: int = 13,
        max_output: int = 144,
        min_output: int = 13,
        
        # ALMA固有パラメータ
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        
        # ZAdaptiveMA固有パラメータ
        z_adaptive_ma_fast_period: int = 2,
        z_adaptive_ma_slow_period: int = 144,
        
        # EfficiencyRatio固有パラメータ
        efficiency_ratio_period: int = 5,
        efficiency_ratio_smoothing_method: str = 'hma',
        efficiency_ratio_use_dynamic_period: bool = True,
        efficiency_ratio_slope_index: int = 3,
        efficiency_ratio_range_threshold: float = 0.005,
        efficiency_ratio_smoother_period: int = 13,
        
        # ATR固有パラメータ
        atr_period: int = None,
        atr_smoothing_method: str = 'alma',
        atr_use_dynamic_period: bool = False,
        atr_slope_index: int = 1,
        atr_range_threshold: float = 0.005,
        
        # volatility固有パラメータ
        volatility_period_mode: str = 'fixed',
        volatility_fixed_period: int = None,
        volatility_calculation_mode: str = 'return',
        volatility_return_type: str = 'log',
        volatility_ddof: int = 1,
        volatility_smoother_type: str = 'hma',
        volatility_smoother_period: int = None,
        
        # 以下のパラメータは互換性のため残すが使用しない
        adaptive_trigger: str = 'chop_trend',
        adaptive_power: float = 1.0,
        adaptive_invert: bool = False,
        adaptive_reverse_mapping: bool = False,
        **trigger_params
    ):
        """初期化"""
        super().__init__("ZAdaptiveFlowTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # 基本パラメータ
            'length': length,
            'smooth_length': smooth_length,
            'src_type': src_type,
            
            # MAタイプ選択
            'ma_type': ma_type,
            
            # MA共通パラメータ
            'ma_period': ma_period,
            'ma_use_dynamic_period': ma_use_dynamic_period,
            'ma_slope_index': ma_slope_index,
            'ma_range_threshold': ma_range_threshold,
            
            # ボラティリティタイプ選択
            'volatility_type': volatility_type,
            
            # 固定乗数パラメータ
            'slow_period_multiplier': slow_period_multiplier,
            'volatility_multiplier': volatility_multiplier,
            
            # 共通サイクル検出器パラメータ
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            
            # ALMA固有パラメータ
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            
            # ZAdaptiveMA固有パラメータ
            'z_adaptive_ma_fast_period': z_adaptive_ma_fast_period,
            'z_adaptive_ma_slow_period': z_adaptive_ma_slow_period,
            
            # EfficiencyRatio固有パラメータ
            'efficiency_ratio_period': efficiency_ratio_period,
            'efficiency_ratio_smoothing_method': efficiency_ratio_smoothing_method,
            'efficiency_ratio_use_dynamic_period': efficiency_ratio_use_dynamic_period,
            'efficiency_ratio_slope_index': efficiency_ratio_slope_index,
            'efficiency_ratio_range_threshold': efficiency_ratio_range_threshold,
            'efficiency_ratio_smoother_period': efficiency_ratio_smoother_period,
            
            # ATR固有パラメータ
            'atr_period': atr_period,
            'atr_smoothing_method': atr_smoothing_method,
            'atr_use_dynamic_period': atr_use_dynamic_period,
            'atr_slope_index': atr_slope_index,
            'atr_range_threshold': atr_range_threshold,
            
            # volatility固有パラメータ
            'volatility_period_mode': volatility_period_mode,
            'volatility_fixed_period': volatility_fixed_period,
            'volatility_calculation_mode': volatility_calculation_mode,
            'volatility_return_type': volatility_return_type,
            'volatility_ddof': volatility_ddof,
            'volatility_smoother_type': volatility_smoother_type,
            'volatility_smoother_period': volatility_smoother_period,
            
            # 互換性のためのパラメータ
            'adaptive_trigger': adaptive_trigger,
            'adaptive_power': adaptive_power,
            'adaptive_invert': adaptive_invert,
            'adaptive_reverse_mapping': adaptive_reverse_mapping,
            **trigger_params
        }
        
        # Z Adaptive Flowトレンドチェンジシグナルの初期化
        self.z_adaptive_flow_signal = ZAdaptiveFlowTrendChangeSignal(
            z_adaptive_flow_params=self._params
        )
        
        # シグナルキャッシュ
        self._entry_signals = None
        self._last_data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_close = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_close = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                    data_signature = (length, first_close, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                    data_signature = (length, first_val, last_val)
                else:
                    data_signature = (0, 0.0, 0.0)
            
            params_signature = (
                self._params.get('length', 10),
                self._params.get('ma_type', 'zlema'),
                self._params.get('volatility_type', 'volatility')
            )
            
            return f"{hash(data_signature)}_{hash(params_signature)}"
            
        except Exception:
            return f"{id(data)}_{self._params.get('ma_type', 'zlema')}"
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        全てのシグナルを計算してキャッシュする
        
        Args:
            data: 価格データ
        """
        try:
            # データハッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._last_data_hash and self._entry_signals is not None:
                return  # キャッシュヒット
            
            # Z Adaptive Flowトレンドチェンジシグナルの計算
            self._entry_signals = self.z_adaptive_flow_signal.generate(data)
            
            # データハッシュを更新
            self._last_data_hash = data_hash
            
            # ベースクラスのキャッシュにも保存
            self._set_cached_signal('entry', self._entry_signals)
            
        except Exception as e:
            self.logger.error(f"シグナル計算中にエラーが発生しました: {str(e)}")
            # エラー時は空のシグナルを設定
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._entry_signals = np.zeros(data_len, dtype=np.int8)
            self._set_cached_signal('entry', self._entry_signals)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # キャッシュされたシグナルをチェック
        cached_signal = self._get_cached_signal('entry')
        if cached_signal is not None:
            data_hash = self._get_data_hash(data)
            if data_hash == self._last_data_hash:
                return cached_signal
        
        # シグナルを計算
        self.calculate_signals(data)
        return self._entry_signals if self._entry_signals is not None else np.array([], dtype=np.int8)
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            bool: エグジットシグナル（True: 決済, False: 保持）
        """
        try:
            # エントリーシグナルを取得
            entry_signals = self.get_entry_signals(data)
            
            if len(entry_signals) == 0 or abs(index) > len(entry_signals):
                return False
            
            # 指定されたインデックスのシグナルを取得
            current_signal = entry_signals[index]
            
            # エグジット条件:
            # ロングポジション(position=1)の場合、ショートシグナル(-1)で決済
            # ショートポジション(position=-1)の場合、ロングシグナル(1)で決済
            if position == 1 and current_signal == -1:
                return True  # ロング決済
            elif position == -1 and current_signal == 1:
                return True  # ショート決済
            
            return False
            
        except Exception as e:
            self.logger.error(f"エグジットシグナル計算中にエラーが発生しました: {str(e)}")
            return False
    
    def get_trend_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Z Adaptive Flowのトレンド状態を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: トレンド状態の値 (1: bullish, -1: bearish)
        """
        return self.z_adaptive_flow_signal.get_trend_state(data)
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        Z Adaptive Flowの詳細な計算結果を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            ZAdaptiveFlowResult: 詳細な計算結果
        """
        return self.z_adaptive_flow_signal.get_detailed_result(data)
    
    def get_trend_lines(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        メインのトレンドライン（basis, level）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            basis, level のタプル
        """
        return self.z_adaptive_flow_signal.get_trend_lines(data)
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        バンド（upper, lower）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            upper, lower のタプル
        """
        return self.z_adaptive_flow_signal.get_bands(data)
    
    def reset(self) -> None:
        """シグナル生成器の状態をリセット"""
        self.clear_cache()
        self._entry_signals = None
        self._last_data_hash = None
        
        if hasattr(self.z_adaptive_flow_signal, 'reset'):
            self.z_adaptive_flow_signal.reset() 