#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, vectorize, prange

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .cycle.ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZROCResult:
    """ZROCの計算結果"""
    roc: np.ndarray              # ROC値
    er: np.ndarray               # 効率比
    dynamic_period: np.ndarray   # 動的ROC期間
    dc_values: np.ndarray        # ドミナントサイクル値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なROCピリオドを計算する（ベクトル化版）
    
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
    効率比に基づいて動的なROCピリオドを計算する（高速化版）
    
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


@njit(fastmath=True)
def calculate_z_roc(
    close: np.ndarray,
    dynamic_period: np.ndarray
) -> np.ndarray:
    """
    動的期間を使用してROC（Rate of Change）を計算する
    
    Args:
        close: 終値の配列
        dynamic_period: 動的なROC期間の配列
    
    Returns:
        ROC値の配列
    """
    # 結果を格納する配列
    size = len(close)
    roc = np.zeros(size)
    
    # 各時点でのROCを計算
    for i in range(size):
        period = int(dynamic_period[i])
        if i < period or period < 1:
            roc[i] = np.nan
        else:
            # ROC = ((現在値 - n期前の値) / n期前の値) * 100
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
    
    return roc


class ZROC(Indicator):
    """
    ZROC インディケーター
    
    Alpha ROCの拡張版。サイクル効率比とドミナントサイクルを用いて動的に期間を調整します。
    
    特徴:
    - ドミナントサイクルを用いた動的な期間計算
    - サイクル効率比による細かな調整
    - トレンドが強い時：短い期間で素早く反応
    - レンジ相場時：長い期間でノイズを除去
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,          # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 144,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 5,              # 最大期間用ドミナントサイクル計算用
        max_dc_max_output: int = 50,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_output: int = 5,             # 最大期間用ドミナントサイクル計算用
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 55,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 3,               # 最小期間用ドミナントサイクル計算用
        min_dc_max_output: int = 13,             # 最小期間用ドミナントサイクル計算用
        min_dc_min_output: int = 3,              # 最小期間用ドミナントサイクル計算用
        
        er_period: int = 21                     # 効率比の計算期間
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル検出の最大期間（デフォルト: 144）
            max_dc_min_cycle: 最大期間用ドミナントサイクル検出の最小期間（デフォルト: 5）
            max_dc_max_output: 最大期間用ドミナントサイクル出力の最大値（デフォルト: 50）
            max_dc_min_output: 最大期間用ドミナントサイクル出力の最小値（デフォルト: 5）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル検出の最大期間（デフォルト: 55）
            min_dc_min_cycle: 最小期間用ドミナントサイクル検出の最小期間（デフォルト: 3）
            min_dc_max_output: 最小期間用ドミナントサイクル出力の最大値（デフォルト: 13）
            min_dc_min_output: 最小期間用ドミナントサイクル出力の最小値（デフォルト: 3）
            
            er_period: 効率比の計算期間（デフォルト: 21）
        """
        super().__init__(
            f"ZROC({max_dc_max_output}-{min_dc_min_output})"
        )
        
        # ドミナントサイクルの最大期間検出器
        self.max_dc = EhlersHoDyDC(
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=max_dc_max_output,
            min_output=max_dc_min_output,
            src_type='close'
        )
        
        # ドミナントサイクルの最小期間検出器
        self.min_dc = EhlersHoDyDC(
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=min_dc_max_output,
            min_output=min_dc_min_output,
            src_type='close'
        )
        
        self.er_period = er_period
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # closeカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合はcloseだけハッシュ
                data_hash = hash(tuple(data[:, 3]))  # close
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
            f"{self.er_period}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ZROCを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
            external_er: 外部から提供される効率比（オプション）
        
        Returns:
            ZROC値の配列
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.roc
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # データ長の検証
            data_length = len(close)
            
            # ドミナントサイクルを使用して動的な最大/最小期間を計算
            max_periods = self.max_dc.calculate(data)
            min_periods = self.min_dc.calculate(data)
            
            # 効率比の計算（外部から提供されない場合は計算）
            if external_er is None:
                er = calculate_efficiency_ratio_for_period(close, self.er_period)
            else:
                # 外部から提供されるERをそのまま使用
                if len(external_er) != data_length:
                    raise ValueError(f"外部ERの長さ({len(external_er)})がデータ長({data_length})と一致しません")
                er = external_er
            
            # 動的なROC期間の計算
            dynamic_period = calculate_dynamic_period(er, max_periods, min_periods)
            
            # ZROCの計算
            roc = calculate_z_roc(close, dynamic_period)
            
            # 結果を保存
            self._result = ZROCResult(
                roc=roc,
                er=er,
                dynamic_period=dynamic_period,
                dc_values=max_periods  # 最大ドミナントサイクル値を保存
            )
            
            self._values = roc  # 基底クラスの要件を満たすため
            
            return roc
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZROC計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に空の配列を返す
            return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
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
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的なROC期間の値を取得する
        
        Returns:
            np.ndarray: 動的なROC期間の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.dynamic_period
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.max_dc.reset()
        self.min_dc.reset() 