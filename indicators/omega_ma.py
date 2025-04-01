#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .alma import calculate_alma


@dataclass
class OmegaMAResult:
    """OmegaMAの計算結果"""
    values: np.ndarray           # OmegaMAの値
    er: np.ndarray               # Efficiency Ratio
    dynamic_period: np.ndarray   # 動的ALMAピリオド
    dynamic_offset: np.ndarray   # 動的オフセット
    dynamic_sigma: np.ndarray    # 動的シグマ


@jit(nopython=True)
def calculate_dynamic_alma_parameters(
    er: np.ndarray, 
    max_period: int, min_period: int,
    max_offset: float, min_offset: float,
    max_sigma: float, min_sigma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    効率比に基づいてALMAのパラメータを動的に調整する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: ALMAの最大期間
        min_period: ALMAの最小期間
        max_offset: ALMAの最大オフセット
        min_offset: ALMAの最小オフセット
        max_sigma: ALMAの最大シグマ
        min_sigma: ALMAの最小シグマ
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            動的なALMA期間、オフセット、シグマの配列
    """
    # ALMAピリオド: ERが高い（トレンドが強い）ほど期間は短く
    periods = min_period + (1.0 - er) * (max_period - min_period)
    periods_rounded = np.round(periods).astype(np.int32)
    
    # オフセット: ERが高い（トレンドが強い）ほどオフセットは大きく（最新データ重視）
    offsets = min_offset + er * (max_offset - min_offset)
    
    # シグマ: ERが高い（トレンドが強い）ほどシグマは小さく（シャープな反応）
    sigmas = max_sigma - er * (max_sigma - min_sigma)
    
    return periods_rounded, offsets, sigmas


@jit(nopython=True)
def calculate_dynamic_alma(
    close: np.ndarray, 
    dynamic_period: np.ndarray,
    dynamic_offset: np.ndarray,
    dynamic_sigma: np.ndarray,
    max_period: int,
    er_period: int
) -> np.ndarray:
    """
    各時点で動的パラメータを用いたALMA値を計算する（高速化版）
    
    Args:
        close: 終値の配列
        dynamic_period: 動的なALMA期間の配列
        dynamic_offset: 動的なオフセットの配列
        dynamic_sigma: 動的なシグマの配列
        max_period: 最大ALMA期間
        er_period: ER計算期間（初期値用）
    
    Returns:
        動的ALMA値の配列
    """
    length = len(close)
    dynamic_alma = np.full(length, np.nan)
    
    # 初期値の設定
    dynamic_alma[0] = close[0]
    
    # 各時点での動的ALMA値を計算
    for i in range(max(er_period, max_period), length):
        period = int(dynamic_period[i])
        if period < 2:  # 最小期間の保証
            period = 2
            
        offset = dynamic_offset[i]
        sigma = dynamic_sigma[i]
        
        # 各時点のALMA計算用にデータを抽出
        data_subset = close[max(0, i-period+1):i+1]
        
        if len(data_subset) >= period:
            # 単一時点のALMA値を計算
            alma_result = calculate_alma(data_subset, period, offset, sigma)
            dynamic_alma[i] = alma_result[-1]
        else:
            # データ不足時は単純平均
            dynamic_alma[i] = np.mean(data_subset)
    
    return dynamic_alma


class OmegaMA(Indicator):
    """
    オメガMA (Omega Moving Average) インジケーター
    
    効率比（ER）に基づいてALMAのすべてのパラメータを動的に調整する適応型移動平均線。
    
    特徴:
    - ALMAの期間が動的に調整
    - ALMAのオフセットが動的に調整
    - ALMAのシグマが動的に調整
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - 短いALMA期間（素早い反応）
      - 高いオフセット（最新値重視）
      - 小さいシグマ（シャープな反応）
    
    - トレンド弱い（ER低い）:
      - 長いALMA期間（ノイズ除去）
      - 低いオフセット（平均化重視）
      - 大きいシグマ（滑らかな曲線）
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_period: int = 55,
        min_period: int = 13,
        max_offset: float = 0.99,
        min_offset: float = 0.6,
        max_sigma: float = 9.0,
        min_sigma: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_period: ALMAの最大期間（デフォルト: 55）
            min_period: ALMAの最小期間（デフォルト: 5）
            max_offset: ALMAの最大オフセット（デフォルト: 0.99）
            min_offset: ALMAの最小オフセット（デフォルト: 0.7）
            max_sigma: ALMAの最大シグマ（デフォルト: 9.0）
            min_sigma: ALMAの最小シグマ（デフォルト: 3.0）
        """
        super().__init__(
            f"OmegaMA({er_period}, {max_period}, {min_period}, "
            f"{max_offset}, {min_offset}, {max_sigma}, {min_sigma})"
        )
        self.er_period = er_period
        self.max_period = max_period
        self.min_period = min_period
        self.max_offset = max_offset
        self.min_offset = min_offset
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        OmegaMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            OmegaMAの値
        """
        try:
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
            self._validate_period(self.er_period, data_length)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的なALMAパラメータの計算
            dynamic_period, dynamic_offset, dynamic_sigma = calculate_dynamic_alma_parameters(
                er,
                self.max_period,
                self.min_period,
                self.max_offset,
                self.min_offset,
                self.max_sigma,
                self.min_sigma
            )
            
            # 動的パラメータを使用したALMA値の計算
            omega_ma_values = calculate_dynamic_alma(
                close,
                dynamic_period,
                dynamic_offset,
                dynamic_sigma,
                self.max_period,
                self.er_period
            )
            
            # 結果の保存
            self._result = OmegaMAResult(
                values=omega_ma_values,
                er=er,
                dynamic_period=dynamic_period,
                dynamic_offset=dynamic_offset,
                dynamic_sigma=dynamic_sigma
            )
            
            self._values = omega_ma_values
            return omega_ma_values
            
        except Exception as e:
            self.logger.error(f"OmegaMA計算中にエラー: {str(e)}")
            return None
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_dynamic_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (動的ALMA期間, 動的オフセット, 動的シグマ)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.dynamic_period,
            self._result.dynamic_offset,
            self._result.dynamic_sigma
        ) 