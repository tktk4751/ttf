#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import hyper_smoother


@dataclass
class AlphaERResult:
    """AlphaERの計算結果"""
    values: np.ndarray         # AlphaERの値
    raw_er: np.ndarray         # スムージング前の効率比


class AlphaER(Indicator):
    """
    アルファER（Alpha Efficiency Ratio）インジケーター
    
    特徴:
    - 効率比（Efficiency Ratio）をハイパースムーサーで平滑化
    - より安定したトレンド/レンジの識別が可能
    - ノイズを削減しつつ、市場状態の変化を的確に捉える
    
    使用方法:
    - 0.618以上: 強いトレンド（効率的な価格変動）
    - 0.382-0.618: 中間的な状態
    - 0.382以下: レンジ・ノイズ（非効率な価格変動）
    
    正規化されたバージョン（0-100）も利用可能:
    - 61.8以上: 強いトレンド
    - 38.2-61.8: 中間的な状態
    - 38.2以下: レンジ・ノイズ
    """
    
    def __init__(
        self,
        period: int = 14,
        normalize: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            period: ER計算期間兼スムージング期間（デフォルト: 14）
            normalize: 0-100にスケールするかどうか（デフォルト: False）
        """
        feature = "Normalized(0-100)" if normalize else "Raw(0-1)"
        super().__init__(
            f"AlphaER({period}, {feature})"
        )
        self.period = period
        self.normalize = normalize
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファERを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            アルファERの値
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.period, data_length)
            
            # 効率比（ER）の計算
            raw_er = calculate_efficiency_ratio_for_period(close, self.period)
            
            # ハイパースムーサーで平滑化
            alpha_er_values = hyper_smoother(raw_er, self.period)
            
            # 正規化（0-100スケール）
            if self.normalize:
                alpha_er_values = alpha_er_values * 100.0
            
            # 結果の保存
            self._result = AlphaERResult(
                values=alpha_er_values,
                raw_er=raw_er
            )
            
            self._values = alpha_er_values
            return alpha_er_values
            
        except Exception as e:
            self.logger.error(f"AlphaER計算中にエラー: {str(e)}")
            return np.array([])
    
    def get_raw_efficiency_ratio(self) -> np.ndarray:
        """
        スムージング前の効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.raw_er 