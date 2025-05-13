#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .alma import calculate_alma_numba as calculate_alma
from .hyper_smoother import hyper_smoother, calculate_hyper_smoother_numba


@dataclass
class AlphaATRResult:
    """AlphaATRの計算結果"""
    values: np.ndarray        # AlphaATRの値（%ベース）
    absolute_values: np.ndarray  # AlphaATRの値（金額ベース）
    tr: np.ndarray           # True Range
    er: np.ndarray           # サイクル効率比（CER）
    dynamic_period: np.ndarray  # 動的ATR期間


@vectorize(['float64(float64, float64)'], nopython=True, fastmath=True)
def max_vec(a: float, b: float) -> float:
    """aとbの最大値を返す（ベクトル化版）"""
    return max(a, b)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def max3_vec(a: float, b: float, c: float) -> float:
    """a, b, cの最大値を返す（ベクトル化版）"""
    return max(a, max(b, c))


@njit(fastmath=True, parallel=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Rangeを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    if length > 0:
        tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算（並列化）
    for i in prange(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, max(tr2, tr3))
    
    return tr


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なATR期間を計算する（ベクトル化版）
    
    Args:
        er: 効率比の値（ERまたはCER）
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の値
    """
    if np.isnan(er):
        return np.nan
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = abs(er)
    return np.round(min_period + (1.0 - er_abs) * (max_period - min_period))


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なATR期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = np.abs(er)
    periods = min_period + (1.0 - er_abs) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True)
def calculate_alpha_atr(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    max_period: int,
    smoother_type: str = 'alma'  # 'alma'または'hyper'
) -> np.ndarray:
    """
    アルファATRを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列（ERまたはCER）
        dynamic_period: 動的期間の配列
        max_period: 最大期間（計算開始位置用）
        smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
    
    Returns:
        AlphaATRの値を返す
    """
    length = len(high)
    alpha_atr = np.zeros(length, dtype=np.float64)
    
    # True Rangeの計算
    tr = calculate_true_range(high, low, close)
    
    # 各時点での平滑化を計算
    for i in prange(max_period, length):
        # その時点での動的な期間を取得
        curr_period = int(dynamic_period[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置までのTRデータを取得
        window = tr[max(0, i-curr_period*2):i+1]  # 効率化のためウィンドウサイズを制限
        
        # 選択された平滑化アルゴリズムを適用
        if smoother_type == 'alma':
            # ALMAを使用して平滑化（固定パラメータ：offset=0.85, sigma=6）
            smoothed_values = calculate_alma(window, curr_period, 0.85, 6.0)
        else:  # 'hyper'
            # ハイパースムーサーを使用して平滑化
            smoothed_values = calculate_hyper_smoother_numba(window, curr_period)
        
        # 最後の値をATRとして使用
        if len(smoothed_values) > 0 and not np.isnan(smoothed_values[-1]):
            alpha_atr[i] = smoothed_values[-1]
    
    return alpha_atr


@njit(fastmath=True, parallel=True)
def calculate_percent_atr(absolute_atr: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    金額ベースのATRから%ベースのATRを計算する（並列高速化版）
    
    Args:
        absolute_atr: 金額ベースのATR配列
        close: 終値の配列
    
    Returns:
        %ベースのATR配列
    """
    length = len(absolute_atr)
    percent_atr = np.zeros_like(absolute_atr, dtype=np.float64)
    
    for i in prange(length):
        if not np.isnan(absolute_atr[i]) and close[i] > 0:
            percent_atr[i] = absolute_atr[i] / close[i]
    
    return percent_atr


class AlphaATR(Indicator):
    """
    アルファATR（Alpha Average True Range）インジケーター
    
    特徴:
    - サイクル効率比（CER）に基づいて期間を動的に調整
    - RSXの3段階平滑化アルゴリズムで平滑化
    - トレンドの強さに応じた適応性
    - 金額ベースと%ベースの両方の値を提供
    
    使用方法:
    - ボラティリティに基づいた利益確定・損切りレベルの設定
    - ATRチャネルやボラティリティストップの構築
    - トレンドの強さに適応したポジションサイジング
    - 異なる価格帯の銘柄間でのボラティリティ比較（%ベース）
    """
    
    def __init__(
        self,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        smoother_type: str = 'alma'  # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            max_atr_period: ATR期間の最大値（デフォルト: 89）
            min_atr_period: ATR期間の最小値（デフォルト: 13）
            smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
        """
        super().__init__(
            f"AlphaATR({max_atr_period}, {min_atr_period})"
        )
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.smoother_type = smoother_type
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = f"{self.max_atr_period}_{self.min_atr_period}_{self.smoother_type}_{external_er_hash}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        アルファATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            AlphaATRの値（%ベース）
        """
        try:
            # サイクル効率比（CER）の検証
            if external_er is None:
                raise ValueError("サイクル効率比（CER）は必須です。external_erパラメータを指定してください")
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換（最小限の処理）
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
                close = np.asarray(data['close'].values, dtype=np.float64)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = np.asarray(data[:, 1], dtype=np.float64)  # high
                    low = np.asarray(data[:, 2], dtype=np.float64)   # low
                    close = np.asarray(data[:, 3], dtype=np.float64) # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証（簡略化）
            data_length = len(high)
            if data_length < self.max_atr_period:
                raise ValueError(f"データ長({data_length})が必要な期間よりも短いです")
            
            # True Range (TR)の計算
            tr = calculate_true_range(high, low, close)
            
            # サイクル効率比（CER）を使用
            er = np.asarray(external_er, dtype=np.float64)
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # 動的なATR期間の計算（ベクトル化関数を使用）
            dynamic_period = calculate_dynamic_period_vec(
                er,
                self.max_atr_period,
                self.min_atr_period
            ).astype(np.int32)
            
            # アルファATRの計算（並列版）
            alpha_atr_values = calculate_alpha_atr(
                high,
                low,
                close,
                er,
                dynamic_period,
                self.max_atr_period,
                self.smoother_type
            )
            
            # 金額ベースのATR値を保存
            absolute_atr_values = alpha_atr_values
            
            # %ベースのATR値に変換（終値に対する比率）（並列版）
            percent_atr_values = calculate_percent_atr(absolute_atr_values, close)
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = AlphaATRResult(
                values=np.copy(percent_atr_values),  # %ベースのATR
                absolute_values=np.copy(absolute_atr_values),  # 金額ベースのATR
                tr=np.copy(tr),
                er=np.copy(er),
                dynamic_period=np.copy(dynamic_period)
            )
            
            self._values = percent_atr_values  # 標準インジケーターインターフェース用
            return percent_atr_values
            
        except Exception as e:
            self.logger.error(f"AlphaATR計算中にエラー: {str(e)}")
            # エラー時は前回の結果を維持
            if self._result is None:
                return np.array([])
            return self._result.values
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range (TR)の値を取得する
        
        Returns:
            np.ndarray: TRの値
        """
        if self._result is None:
            return np.array([])
        return self._result.tr
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的ATR期間の値を取得する
        
        Returns:
            np.ndarray: 動的ATR期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_period
    
    def get_percent_atr(self) -> np.ndarray:
        """
        %ベースのATRを取得する
        
        Returns:
            np.ndarray: %ベースのATR値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            return np.array([])
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_atr(self) -> np.ndarray:
        """
        金額ベースのATRを取得する
        
        Returns:
            np.ndarray: 金額ベースのATR値
        """
        if self._result is None:
            return np.array([])
        return self._result.absolute_values
    
    def get_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのATR × 倍数
        """
        atr = self.get_percent_atr()
        return atr * multiplier
    
    def get_absolute_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        金額ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 金額ベースのATR × 倍数
        """
        abs_atr = self.get_absolute_atr()
        return abs_atr * multiplier
        
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None 