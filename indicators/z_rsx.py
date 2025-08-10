#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange, vectorize

from .indicator import Indicator
from .rsx import calculate_rsx_numba
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .cycle.ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZRSXResult:
    """ZRSXの計算結果"""
    values: np.ndarray            # ZRSX値（0-100の範囲）
    er: np.ndarray                # サイクル効率比
    adaptive_periods: np.ndarray  # 適応的なRSX期間
    adaptive_high_levels: np.ndarray  # 適応的な買われすぎレベル
    adaptive_low_levels: np.ndarray   # 適応的な売られすぎレベル
    dc_values: np.ndarray         # ドミナントサイクル値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なRSXピリオドを計算する（ベクトル化版）
    
    Args:
        er: 効率比の値
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の値
    """
    if np.isnan(er):
        return np.nan
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    return np.round(min_period + (1.0 - abs(er)) * (max_period - min_period))


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: np.ndarray, min_period: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的なRSXピリオドを計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間の配列（ドミナントサイクルから計算）
        min_period: 最小期間の配列（ドミナントサイクルから計算）
    
    Returns:
        動的な期間の配列
    """
    size = len(er)
    dynamic_period = np.zeros(size)
    
    for i in range(size):
        if np.isnan(er[i]):
            dynamic_period[i] = max_period[i]
        else:
            # ERが高い（トレンドが強い）ほど期間は短く、
            # ERが低い（トレンドが弱い）ほど期間は長くなる
            dynamic_period[i] = min_period[i] + (1.0 - abs(er[i])) * (max_period[i] - min_period[i])
            # 整数に丸める
            dynamic_period[i] = round(dynamic_period[i])
    
    return dynamic_period


@njit(fastmath=True, parallel=True)
def calculate_adaptive_levels(
    er_values: np.ndarray,
    min_high_level: float = 85.0,
    max_high_level: float = 90.0,
    min_low_level: float = 10.0,
    max_low_level: float = 15.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    効率比（Efficiency Ratio）に基づいて適応的に高値/安値レベルを計算する
    
    Args:
        er_values: 効率比の配列
        min_high_level: 最小高値レベル（デフォルト: 85.0）
        max_high_level: 最大高値レベル（デフォルト: 90.0）
        min_low_level: 最小安値レベル（デフォルト: 10.0）
        max_low_level: 最大安値レベル（デフォルト: 15.0）
    
    Returns:
        (適応的な高値レベルの配列, 適応的な安値レベルの配列)のタプル
    """
    size = len(er_values)
    adaptive_high_levels = np.zeros(size)
    adaptive_low_levels = np.zeros(size)
    
    # ERが高い（トレンドが強い）→ より厳しいレベル（買われすぎ/売られすぎの判断を厳格に）
    # ERが低い（レンジ相場）→ より緩いレベル（買われすぎ/売られすぎの判断を緩やかに）
    for i in prange(size):
        er_abs = abs(er_values[i]) if not np.isnan(er_values[i]) else 0
        adaptive_high_levels[i] = min_high_level + er_abs * (max_high_level - min_high_level)
        adaptive_low_levels[i] = min_low_level + er_abs * (max_low_level - min_low_level)
    
    return adaptive_high_levels, adaptive_low_levels


class ZRSX(Indicator):
    """
    ZRSX インディケーター
    
    Alpha RSXの拡張版。サイクル効率比とドミナントサイクルを用いて動的に期間を調整します。
    
    特徴:
    - ドミナントサイクルを用いた動的な期間計算
    - サイクル効率比による細かな調整
    - 市場の状態に応じた買われすぎ/売られすぎレベルの動的調整
    - RSXの優れたノイズ除去能力と反応速度を維持
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,          # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 55,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 5,              # 最大期間用ドミナントサイクル計算用
        max_dc_max_output: int = 34,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_output: int = 14,             # 最大期間用ドミナントサイクル計算用
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 34,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 3,               # 最小期間用ドミナントサイクル計算用
        min_dc_max_output: int = 13,             # 最小期間用ドミナントサイクル計算用
        min_dc_min_output: int = 3,              # 最小期間用ドミナントサイクル計算用
        
        er_period: int = 10,                     # 効率比の計算期間
        min_high_level: float = 85.0,            # 最小買われすぎレベル
        max_high_level: float = 90.0,            # 最大買われすぎレベル
        min_low_level: float = 10.0,             # 最小売られすぎレベル
        max_low_level: float = 15.0              # 最大売られすぎレベル
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル検出の最大期間（デフォルト: 55）
            max_dc_min_cycle: 最大期間用ドミナントサイクル検出の最小期間（デフォルト: 5）
            max_dc_max_output: 最大期間用ドミナントサイクル出力の最大値（デフォルト: 34）
            max_dc_min_output: 最大期間用ドミナントサイクル出力の最小値（デフォルト: 14）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル検出の最大期間（デフォルト: 34）
            min_dc_min_cycle: 最小期間用ドミナントサイクル検出の最小期間（デフォルト: 3）
            min_dc_max_output: 最小期間用ドミナントサイクル出力の最大値（デフォルト: 13）
            min_dc_min_output: 最小期間用ドミナントサイクル出力の最小値（デフォルト: 3）
            
            er_period: 効率比の計算期間（デフォルト: 10）
            min_high_level: 最小買われすぎレベル（デフォルト: 85.0）
            max_high_level: 最大買われすぎレベル（デフォルト: 90.0）
            min_low_level: 最小売られすぎレベル（デフォルト: 10.0）
            max_low_level: 最大売られすぎレベル（デフォルト: 15.0）
        """
        super().__init__(
            f"ZRSX({max_dc_max_output}-{min_dc_min_output})"
        )
        
        # ドミナントサイクルの最大期間検出器
        self.max_dc = EhlersHoDyDC(
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=max_dc_max_output,
            min_output=max_dc_min_output,
            src_type='hlc3'
        )
        
        # ドミナントサイクルの最小期間検出器
        self.min_dc = EhlersHoDyDC(
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=min_dc_max_output,
            min_output=min_dc_min_output,
            src_type='hlc3'
        )
        
        self.er_period = er_period
        self.min_high_level = min_high_level
        self.max_high_level = max_high_level
        self.min_low_level = min_low_level
        self.max_low_level = max_low_level
        
        # 適応的なパラメーター用の配列
        self._adaptive_periods = None
        self._adaptive_high_levels = None
        self._adaptive_low_levels = None
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            columns = ['high', 'low', 'close']
            if all(col in data.columns for col in columns):
                data_hash = hash(tuple(map(tuple, data[columns].values)))
            elif 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # 必要なカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合は必要な列だけハッシュ
                data_hash = hash(tuple(map(tuple, data[:, 1:4])))  # high, low, close
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = (
            f"{self.max_dc.cycle_part}_{self.max_dc.max_cycle}_{self.max_dc.min_cycle}_"
            f"{self.max_dc.max_output}_{self.max_dc.min_output}_"
            f"{self.min_dc.cycle_part}_{self.min_dc.max_cycle}_{self.min_dc.min_cycle}_"
            f"{self.min_dc.max_output}_{self.min_dc.min_output}_"
            f"{self.er_period}_{self.min_high_level}_{self.max_high_level}_"
            f"{self.min_low_level}_{self.max_low_level}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ZRSXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要（HLC3の計算用）
                または'close'カラムが必要
            external_er: 外部から提供される効率比（オプション）
        
        Returns:
            ZRSX値の配列（0-100の範囲）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if set(['high', 'low', 'close']).issubset(data.columns):
                    # HLC3の計算
                    hlc3 = (data['high'] + data['low'] + data['close']) / 3
                    prices = hlc3.values
                elif 'close' in data.columns:
                    prices = data['close'].values
                else:
                    raise ValueError("DataFrameには'close'または'high','low','close'カラムが必要です")
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    # OHLC配列からHLC3を計算
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                    prices = (high + low + close) / 3
                elif data.ndim == 1:
                    prices = data
                else:
                    raise ValueError("サポートされていないデータ形式です")
            
            # データ長の検証
            data_length = len(prices)
            
            # ドミナントサイクルを使用して動的な最大/最小期間を計算
            max_periods = self.max_dc.calculate(data)
            min_periods = self.min_dc.calculate(data)
            
            # 効率比の計算（外部から提供されない場合は計算）
            if external_er is None:
                er = calculate_efficiency_ratio_for_period(prices, self.er_period)
            else:
                # 外部から提供されるERをそのまま使用
                if len(external_er) != data_length:
                    raise ValueError(f"外部ERの長さ({len(external_er)})がデータ長({data_length})と一致しません")
                er = external_er
            
            # 適応的なパラメーターの計算
            self._adaptive_periods = calculate_dynamic_period(
                er, 
                max_periods, 
                min_periods
            )
            
            self._adaptive_high_levels, self._adaptive_low_levels = calculate_adaptive_levels(
                er,
                self.min_high_level,
                self.max_high_level,
                self.min_low_level,
                self.max_low_level
            )
            
            # 通常のRSXよりも厳密な計算が必要
            # 各ポイントで異なる期間を使用するので、各時点ごとに計算する
            size = len(prices)
            rsx_values = np.zeros(size)
            
            # 最初の部分は常に0（データが不足）
            # 最大期間から計算開始
            max_period = int(np.max(max_periods))
            for i in range(max_period, size):
                # その時点での適応的なperiodを取得
                adaptive_period = int(self._adaptive_periods[i])
                if adaptive_period < 3:  # RSXの最小期間は3
                    adaptive_period = 3
                
                # その時点でのRSXを計算（適応的な期間を使用）
                # 効率のため、必要な部分の配列だけを使用
                window = prices[max(0, i-adaptive_period*2):i+1]  # RSXの計算には少なくともperiod*2が必要
                if len(window) > adaptive_period:
                    try:
                        rsx_value = calculate_rsx_numba(window, adaptive_period)[-1]
                        rsx_values[i] = rsx_value
                    except:
                        # エラー時は前の値を使用
                        rsx_values[i] = rsx_values[i-1] if i > 0 else 50
                else:
                    # データ不足時は前の値を使用
                    rsx_values[i] = rsx_values[i-1] if i > 0 else 50
            
            # 結果を保存
            self._result = ZRSXResult(
                values=rsx_values,
                er=er,
                adaptive_periods=self._adaptive_periods,
                adaptive_high_levels=self._adaptive_high_levels,
                adaptive_low_levels=self._adaptive_low_levels,
                dc_values=max_periods  # 最大ドミナントサイクル値を保存
            )
            
            self._values = rsx_values  # 基底クラスの要件を満たすため
            
            return rsx_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZRSX計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_adaptive_periods(self) -> np.ndarray:
        """
        適応的な期間を取得
        
        Returns:
            適応的な期間の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.adaptive_periods
    
    def get_adaptive_levels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        適応的な高値/安値レベルを取得
        
        Returns:
            (適応的な高値レベルの配列, 適応的な安値レベルの配列)のタプル
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        return self._result.adaptive_high_levels, self._result.adaptive_low_levels
    
    def get_max_dc_values(self) -> np.ndarray:
        """
        最大ドミナントサイクル値を取得する
        
        Returns:
            np.ndarray: 最大ドミナントサイクル値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_min_dc_values(self) -> np.ndarray:
        """
        最小ドミナントサイクル値を取得する
        
        Returns:
            np.ndarray: 最小ドミナントサイクル値
        """
        if self._result is None:
            return np.array([])
        return self.min_dc._values if hasattr(self.min_dc, '_values') else np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_overbought_oversold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        買われすぎ・売られすぎのシグナルを取得（適応的なレベルを使用）
        
        Returns:
            (買われすぎシグナル, 売られすぎシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        
        # 適応的なレベルを使用して買われすぎ/売られすぎを判定
        overbought_signal = np.where(self._values >= self._result.adaptive_high_levels, 1, 0)
        oversold_signal = np.where(self._values <= self._result.adaptive_low_levels, 1, 0)
        
        return overbought_signal, oversold_signal
    
    def get_crossover_signals(self, level: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        クロスオーバー・クロスアンダーのシグナルを取得
        
        Args:
            level: クロスするレベル（デフォルト：50）
        
        Returns:
            (クロスオーバーシグナル, クロスアンダーシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            empty = np.array([])
            return empty, empty
        
        # 1つ前の値を取得（最初の要素には前の値がないので同じ値を使用）
        prev_values = np.roll(self._values, 1)
        prev_values[0] = self._values[0]
        
        # クロスオーバー: 前の値がレベル未満で、現在の値がレベル以上
        crossover = np.where(
            (prev_values < level) & (self._values >= level),
            1, 0
        )
        
        # クロスアンダー: 前の値がレベル以上で、現在の値がレベル未満
        crossunder = np.where(
            (prev_values >= level) & (self._values < level),
            1, 0
        )
        
        return crossover, crossunder
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self._adaptive_periods = None
        self._adaptive_high_levels = None
        self._adaptive_low_levels = None
        self.max_dc.reset()
        self.min_dc.reset() 