#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.z_rsx import ZRSX
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    rsx_values: np.ndarray,
    high_levels: np.ndarray,
    low_levels: np.ndarray
) -> np.ndarray:
    """
    ZRSXシグナルを生成する（高速化版）
    
    Args:
        rsx_values: ZRSX値の配列（0-100の範囲）
        high_levels: 高値（買われすぎ）レベルの配列
        low_levels: 安値（売られすぎ）レベルの配列
    
    Returns:
        シグナルの配列 (1: 買いフィルターがオン、-1: 売りフィルターがオン、0: フィルターがオフ)
    """
    length = len(rsx_values)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        if np.isnan(rsx_values[i]):
            continue
        
        # 買いフィルター: RSXが高値レベルを上回っている場合
        if rsx_values[i] > high_levels[i]:
            signals[i] = 1
        # 売りフィルター: RSXが安値レベルを下回っている場合
        elif rsx_values[i] < low_levels[i]:
            signals[i] = -1
    
    return signals


class ZRSXFilterSignal(BaseSignal, IFilterSignal):
    """
    ZRSXを使用したフィルターシグナル
    
    特徴:
    - サイクル効率比(ER)に基づく動的なしきい値レベル
    - ドミナントサイクルに基づく適応的な期間調整
    
    シグナル値:
    - 1: 買いフィルターがオン（RSX > 動的な高値レベル）
    - -1: 売りフィルターがオン（RSX < 動的な安値レベル）
    - 0: フィルターがオフ（その他の場合）
    
    使用例:
    - 相場環境のフィルタリング
    - トレンドの強さの判断
    - 買われすぎ/売られすぎの検出
    """
    
    def __init__(
        self,
        # サイクル効率比(ER)のパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        er_period: int = 10,
        
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
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器のタイプ（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 13）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            er_period: 効率比の計算期間（デフォルト: 10）
            
            max_dc_cycle_part: 最大ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大ドミナントサイクル検出の最大期間（デフォルト: 55）
            max_dc_min_cycle: 最大ドミナントサイクル検出の最小期間（デフォルト: 5）
            max_dc_max_output: 最大ドミナントサイクル出力の最大値（デフォルト: 34）
            max_dc_min_output: 最大ドミナントサイクル出力の最小値（デフォルト: 14）
            
            min_dc_cycle_part: 最小ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小ドミナントサイクル検出の最大期間（デフォルト: 34）
            min_dc_min_cycle: 最小ドミナントサイクル検出の最小期間（デフォルト: 3）
            min_dc_max_output: 最小ドミナントサイクル出力の最大値（デフォルト: 13）
            min_dc_min_output: 最小ドミナントサイクル出力の最小値（デフォルト: 3）
            
            min_high_level: 最小買われすぎレベル（デフォルト: 75.0）
            max_high_level: 最大買われすぎレベル（デフォルト: 85.0）
            min_low_level: 最小売られすぎレベル（デフォルト: 25.0）
            max_low_level: 最大売られすぎレベル（デフォルト: 15.0）
        """
        # パラメータの設定
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'er_period': er_period,
            
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
        
        super().__init__(
            f"ZRSXFilter({max_dc_max_output}-{min_dc_min_output}, {min_high_level}-{max_low_level})",
            params
        )
        
        # サイクル効率比の計算機を初期化
        self._cycle_efficiency_ratio = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            period=er_period
        )
        
        # ZRSXインディケーターの初期化
        self._rsx = ZRSX(
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
            
            er_period=er_period,
            min_high_level=min_high_level,
            max_high_level=max_high_level,
            min_low_level=min_low_level,
            max_low_level=max_low_level
        )
        
        # 結果キャッシュ
        self._signals = None
        self._rsx_values = None
        self._high_levels = None
        self._low_levels = None
        self._er_values = None
        self._data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{hash(frozenset(self._params.items()))}"
        
        return f"{data_hash}_{param_str}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'high', 'low', 'close'カラムが必要（HLC3の計算用）
                または'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            シグナルの配列 (1: 買いフィルターがオン、-1: 売りフィルターがオン、0: フィルターがオフ)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # サイクル効率比の計算
            self._er_values = self._cycle_efficiency_ratio.calculate(data)
            
            # ZRSX値の計算（外部から提供されたサイクル効率比を使用）
            self._rsx_values = self._rsx.calculate(data, self._er_values)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if self._rsx_values is None or len(self._rsx_values) == 0:
                self._signals = np.full(len(data), np.nan)
                return self._signals
            
            # 適応的なレベルの取得
            self._high_levels, self._low_levels = self._rsx.get_adaptive_levels()
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(self._rsx_values, self._high_levels, self._low_levels)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZRSXFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        ZRSX値を取得する
        
        Returns:
            ZRSX値の配列
        """
        return self._rsx_values if self._rsx_values is not None else np.array([])
    
    def get_high_levels(self) -> np.ndarray:
        """
        高値（買われすぎ）レベルを取得する
        
        Returns:
            高値レベルの配列
        """
        return self._high_levels if self._high_levels is not None else np.array([])
    
    def get_low_levels(self) -> np.ndarray:
        """
        安値（売られすぎ）レベルを取得する
        
        Returns:
            安値レベルの配列
        """
        return self._low_levels if self._low_levels is not None else np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（ER）を取得する
        
        Returns:
            効率比の配列
        """
        return self._er_values if self._er_values is not None else np.array([])
    
    def get_adaptive_periods(self) -> np.ndarray:
        """
        適応的な期間を取得する
        
        Returns:
            適応的な期間の配列
        """
        return self._rsx.get_adaptive_periods()
    
    def get_overbought_oversold(self) -> np.ndarray:
        """
        買われすぎ/売られすぎの状態を取得する
        
        Returns:
            状態の配列 (1: 買われすぎ、-1: 売られすぎ、0: ニュートラル)
        """
        return self._signals
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._rsx, 'reset'):
            self._rsx.reset()
        if hasattr(self._cycle_efficiency_ratio, 'reset'):
            self._cycle_efficiency_ratio.reset()
        self._signals = None
        self._rsx_values = None
        self._high_levels = None
        self._low_levels = None
        self._er_values = None
        self._data_hash = None 