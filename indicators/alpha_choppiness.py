#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import hyper_smoother


@dataclass
class AlphaChoppinessResult:
    """AlphaChoppinessの計算結果"""
    values: np.ndarray          # AlphaChoppinessの値（0-1の範囲で正規化）
    er: np.ndarray              # Efficiency Ratio
    dynamic_period: np.ndarray  # 動的チョピネス期間
    tr: np.ndarray              # True Range


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なチョピネス期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_period + (1.0 - er) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_alpha_choppiness(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    rsx_period: int,
    max_period: int
) -> np.ndarray:
    """
    アルファチョピネスを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列
        dynamic_period: 動的期間の配列
        rsx_period: RSX平滑化期間
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        アルファチョピネスの配列（0-1の範囲で正規化）
        - 1に近いほどトレンド相場
        - 0に近いほどレンジ相場
    """
    length = len(high)
    result = np.full(length, np.nan)
    
    # True Rangeの計算
    tr = calculate_tr(high, low, close)
    
    # 各時点でのチョピネス計算
    for i in range(max_period, length):
        period = int(dynamic_period[i])
        if period < 2:
            period = 2
        
        # TRのサブセット抽出
        tr_subset = tr[max(0, i - period * 3):i+1]
        
        # ハイパースムーサーによるTR平滑化
        if len(tr_subset) > 0:
            tr_smoothed = hyper_smoother(tr_subset, period)[-1]
            
            # 現在位置から動的期間分のデータを抽出
            start_idx = max(0, i - period + 1)
            period_high = np.max(high[start_idx:i+1])
            period_low = np.min(low[start_idx:i+1])
            range_sum = period_high - period_low
            
            if range_sum > 0:  # ゼロ除算を防ぐ
                # ATRの合計値（平滑化TRを使用）
                atr_sum = tr_smoothed * period
                
                # チョピネスの計算と正規化（0-1の範囲に）
                chop = np.log10(atr_sum / range_sum) / np.log10(period)
                # 100を超えることはないが、念のため制限
                chop = min(max(chop, 0.0), 1.0)
                # トレンド指数として反転（1 - chop）
                result[i] = 1.0 - chop
    
    return result


class AlphaChoppiness(Indicator):
    """
    アルファチョピネス（AlphaChoppiness）インジケーター
    
    特徴:
    - 効率比（ER）に基づいて期間を動的に調整
    - ATR計算にALMA（Arnaud Legoux Moving Average）を使用
    - 0-1の範囲で正規化（1がトレンド、0がレンジ）
    
    解釈:
    - 0.7以上: 強いトレンド相場
    - 0.3以下: 強いレンジ相場
    - 0.3-0.7: 混合状態
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        rsx_period: int = 10
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_chop_period: チョピネス期間の最大値（デフォルト: 55）
            min_chop_period: チョピネス期間の最小値（デフォルト: 8）
            rsx_period: RSX平滑化期間（デフォルト: 10）
        """
        super().__init__(
            f"AlphaChop({er_period}, {max_chop_period}, {min_chop_period}, "
            f"{rsx_period})"
        )
        self.er_period = er_period
        self.max_chop_period = max_chop_period
        self.min_chop_period = min_chop_period
        self.rsx_period = rsx_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファチョピネスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            アルファチョピネスの値（0-1の範囲）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(high)
            
            # データ長が最小期間より小さい場合は、NaNで埋めた配列を返す
            if data_length < self.min_chop_period:
                self.logger.warning(f"データ長（{data_length}）が最小期間（{self.min_chop_period}）より小さいため、NaN配列を返します")
                result = np.full(data_length, np.nan)
                self._values = result
                
                # 結果の保存（すべてNaN）
                self._result = AlphaChoppinessResult(
                    values=result,
                    er=np.full(data_length, np.nan),
                    dynamic_period=np.full(data_length, self.min_chop_period),
                    tr=np.full(data_length, np.nan)
                )
                
                return result
            
            # 期間の検証
            self._validate_period(self.er_period, data_length)
            
            # True Range (TR)の計算
            tr = calculate_tr(high, low, close)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, min(self.er_period, data_length - 1))
            
            # 動的なチョピネス期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                min(self.max_chop_period, data_length - 1),
                min(self.min_chop_period, data_length - 1)
            )
            
            # 結果配列の初期化
            alpha_chop_values = np.full(data_length, np.nan)
            
            # 各時点でのTR平滑化とチョピネス計算
            for i in range(min(self.max_chop_period, data_length), data_length):
                # その時点での動的な期間を取得
                curr_period = int(dynamic_period[i])
                curr_period = max(2, min(curr_period, i))  # 期間が2未満またはiより大きくならないように制限
                
                # TRのサブセット抽出
                tr_subset = tr[max(0, i - curr_period * 3):i+1]
                
                # ハイパースムーサーによるTR平滑化
                if len(tr_subset) > 0:
                    tr_smoothed = hyper_smoother(tr_subset, min(curr_period, len(tr_subset)))[-1]
                    
                    # 現在位置から動的期間分のデータを抽出
                    start_idx = max(0, i - curr_period + 1)
                    period_high = np.max(high[start_idx:i+1])
                    period_low = np.min(low[start_idx:i+1])
                    range_sum = period_high - period_low
                    
                    if range_sum > 0:  # ゼロ除算を防ぐ
                        # ATRの合計値（平滑化TRを使用）
                        atr_sum = tr_smoothed * curr_period
                        
                        # チョピネスの計算と正規化（0-1の範囲に）
                        chop = np.log10(atr_sum / range_sum) / np.log10(curr_period)
                        # 100を超えることはないが、念のため制限
                        chop = min(max(chop, 0.0), 1.0)
                        # トレンド指数として反転（1 - chop）
                        alpha_chop_values[i] = 1.0 - chop
            
            # 結果の保存
            self._result = AlphaChoppinessResult(
                values=alpha_chop_values,
                er=er,
                dynamic_period=dynamic_period,
                tr=tr
            )
            
            self._values = alpha_chop_values
            return alpha_chop_values
            
        except Exception as e:
            self.logger.error(f"AlphaChoppiness計算中にエラー: {str(e)}")
            return np.full(len(data) if hasattr(data, '__len__') else 0, np.nan)
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的チョピネス期間の値を取得する
        
        Returns:
            np.ndarray: 動的チョピネス期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range (TR)の値を取得する
        
        Returns:
            np.ndarray: TRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.tr 