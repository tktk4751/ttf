#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy.stats import norm
from numba import jit, njit, prange, float64, int64

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class KernelMAResult:
    """KernelMAの計算結果"""
    values: np.ndarray        # KernelMAの値
    er: np.ndarray            # 効率比
    bandwidth: np.ndarray     # 動的バンド幅
    upper_band: np.ndarray    # 上側バンド
    lower_band: np.ndarray    # 下側バンド
    slope: np.ndarray         # 傾き（トレンド方向）


@njit(fastmath=True)
def gaussian_kernel(x: float) -> float:
    """
    ガウシアンカーネル関数
    
    Args:
        x: 入力値
    
    Returns:
        カーネル値
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


@njit(fastmath=True)
def epanechnikov_kernel(x: float) -> float:
    """
    エパネチニコフカーネル関数
    
    Args:
        x: 入力値
    
    Returns:
        カーネル値
    """
    if abs(x) <= 1:
        return 0.75 * (1 - x * x)
    return 0.0


@njit(fastmath=True, parallel=True)
def calculate_dynamic_bandwidth(er: np.ndarray, max_bandwidth: float, min_bandwidth: float) -> np.ndarray:
    """
    効率比に基づいて動的なバンド幅を計算する
    
    Args:
        er: 効率比の配列
        max_bandwidth: 最大バンド幅
        min_bandwidth: 最小バンド幅
    
    Returns:
        動的なバンド幅の配列
    """
    # ERが高い（トレンドが強い）ほどバンド幅は小さく（より反応的に）、
    # ERが低い（トレンドが弱い）ほどバンド幅は大きくなる（より平滑に）
    bandwidth = min_bandwidth + (1.0 - er) * (max_bandwidth - min_bandwidth)
    return bandwidth


@njit(fastmath=True)
def apply_kernel_weights(x_diff: np.ndarray, h: float, kernel_type: int) -> np.ndarray:
    """
    カーネル重みを計算する（ベクトル化バージョン）
    
    Args:
        x_diff: x_pred[i] - x の配列
        h: バンド幅
        kernel_type: カーネルタイプ (0=ガウシアン, 1=エパネチニコフ)
    
    Returns:
        重み配列
    """
    scaled_diff = x_diff / h
    
    if kernel_type == 0:  # ガウシアン
        weights = np.exp(-0.5 * scaled_diff * scaled_diff) / np.sqrt(2 * np.pi)
    else:  # エパネチニコフ
        weights = np.zeros_like(scaled_diff)
        mask = np.abs(scaled_diff) <= 1.0
        weights[mask] = 0.75 * (1.0 - scaled_diff[mask] * scaled_diff[mask])
    
    # 正規化
    sum_weights = np.sum(weights)
    if sum_weights > 1e-10:
        weights = weights / sum_weights
    
    return weights


@njit(fastmath=True, parallel=True)
def kernel_regression_no_lookahead(x: np.ndarray, y: np.ndarray, bandwidth: np.ndarray, kernel_type: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    未来データを使用しないカーネル回帰を実行する
    
    各時点iでは、時点i以前のデータポイントのみを使用します。
    
    Args:
        x: 説明変数（時間など）
        y: 目的変数（価格など）
        bandwidth: バンド幅の配列
        kernel_type: カーネルタイプ (0=ガウシアン, 1=エパネチニコフ)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 予測値、上側バンド、下側バンド
    """
    n = len(x)
    y_pred = np.zeros(n)
    y_upper = np.zeros(n)
    y_lower = np.zeros(n)
    
    # 最初の数ポイントは計算に十分なデータがないため、単純な移動平均で初期化
    min_points = 5  # 最低限必要なデータポイント数
    
    # 並列処理のためにprangeを使用
    for i in prange(n):
        # 現在のバンド幅を取得
        h = bandwidth[i]
        
        # 時点i以前のデータポイントのみを使用
        if i < min_points:
            # 十分なデータがない場合は単純な平均を使用
            if i > 0:
                y_pred[i] = np.mean(y[:i+1])
                # 標準偏差を使用した簡易的な信頼区間
                std = np.std(y[:i+1]) if i > 1 else 0.0
                y_upper[i] = y_pred[i] + 1.96 * std
                y_lower[i] = y_pred[i] - 1.96 * std
            else:
                y_pred[i] = y[i]
                y_upper[i] = y[i]
                y_lower[i] = y[i]
            continue
        
        # 過去のデータのみを使用
        x_past = x[:i+1]
        y_past = y[:i+1]
        
        # x_diffを計算（過去データのみ）
        x_diff = x[i] - x_past
        
        # カーネル重みを計算
        weights = np.zeros(i+1)
        for j in range(i+1):
            scaled_diff = x_diff[j] / h
            
            if kernel_type == 0:  # ガウシアン
                weights[j] = np.exp(-0.5 * scaled_diff * scaled_diff) / np.sqrt(2 * np.pi)
            else:  # エパネチニコフ
                if np.abs(scaled_diff) <= 1.0:
                    weights[j] = 0.75 * (1.0 - scaled_diff * scaled_diff)
        
        # 正規化
        weights_sum = np.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
            
            # 加重平均を計算
            y_pred[i] = np.sum(weights * y_past)
            
            # 信頼区間の計算（加重標準偏差を使用）
            weighted_var = np.sum(weights * (y_past - y_pred[i])**2)
            std_dev = np.sqrt(weighted_var)
            y_upper[i] = y_pred[i] + 1.96 * std_dev  # 95%信頼区間
            y_lower[i] = y_pred[i] - 1.96 * std_dev
        else:
            # 重みの合計が極めて小さい場合は直近の値を使用
            y_pred[i] = y[i]
            y_upper[i] = y[i]
            y_lower[i] = y[i]
    
    return y_pred, y_upper, y_lower


@njit(fastmath=True, parallel=True)
def calculate_kernel_ma_slope(values: np.ndarray, period: int = 5) -> np.ndarray:
    """
    KernelMAの傾きを計算する（最適化版）
    
    Args:
        values: KernelMAの値
        period: 傾きを計算する期間
    
    Returns:
        傾きの配列
    """
    length = len(values)
    slope = np.zeros(length)
    
    # 事前計算: x配列と関連する統計量
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_diff = x - x_mean
    denominator = np.sum(x_diff ** 2)
    
    # 並列処理のためにprangeを使用
    for i in prange(period, length):
        # 現在のウィンドウのデータを取得
        y = values[i-period:i]
        
        # NaNをチェック
        if np.any(np.isnan(y)):
            slope[i] = slope[i-1] if i > 0 else 0
            continue
            
        # 傾きを計算
        y_mean = np.mean(y)
        numerator = np.sum(x_diff * (y - y_mean))
        
        if denominator != 0:
            slope[i] = numerator / denominator
        else:
            slope[i] = 0
    
    return slope


class KernelMA(Indicator):
    """
    カーネル回帰法を用いた適応型移動平均線インジケーター
    
    効率比（ER）に基づいてカーネルのバンド幅を動的に調整し、
    市場状況に適応する高度な移動平均線。
    
    特徴:
    - 非パラメトリックな平滑化手法によるノイズ除去
    - 効率比に基づく動的バンド幅調整
    - 上下バンドによる価格変動の範囲予測
    - 傾きによるトレンド方向の判定
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_bandwidth: float = 10.0,
        min_bandwidth: float = 2.0,
        kernel_type: str = 'gaussian',
        confidence_level: float = 0.95,
        slope_period: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_bandwidth: バンド幅の最大値（デフォルト: 10.0）
            min_bandwidth: バンド幅の最小値（デフォルト: 2.0）
            kernel_type: カーネルの種類（'gaussian'または'epanechnikov'）
            confidence_level: 信頼区間のレベル（デフォルト: 0.95）
            slope_period: 傾きを計算する期間（デフォルト: 5）
        """
        super().__init__(
            f"KernelMA({er_period}, {max_bandwidth}, {min_bandwidth}, {kernel_type})"
        )
        self.er_period = er_period
        self.max_bandwidth = max_bandwidth
        self.min_bandwidth = min_bandwidth
        self.kernel_type = kernel_type
        self.confidence_level = confidence_level
        self.slope_period = slope_period
        self._result = None
        
        # カーネル関数の選択（数値コードに変換）
        if kernel_type.lower() == 'gaussian':
            self.kernel_type_code = 0
        elif kernel_type.lower() == 'epanechnikov':
            self.kernel_type_code = 1
        else:
            self.logger.warning(f"未知のカーネルタイプ: {kernel_type}、ガウシアンカーネルを使用します")
            self.kernel_type_code = 0
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        KernelMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            KernelMAの値
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
            
            # 動的なバンド幅の計算
            dynamic_bandwidth = calculate_dynamic_bandwidth(
                er,
                self.max_bandwidth,
                self.min_bandwidth
            )
            
            # カーネル回帰の実行（GPU/CPU選択）
            x = np.arange(data_length, dtype=np.float64)
            
            # 時系列制約を導入したカーネル回帰を使用
            kernel_ma_values, upper_band, lower_band = kernel_regression_no_lookahead(
                x, close, dynamic_bandwidth, self.kernel_type_code
            )
            
            # 傾きの計算（最適化版）
            slope = calculate_kernel_ma_slope(kernel_ma_values, self.slope_period)
            
            # 結果の保存
            self._result = KernelMAResult(
                values=kernel_ma_values,
                er=er,
                bandwidth=dynamic_bandwidth,
                upper_band=upper_band,
                lower_band=lower_band,
                slope=slope
            )
            
            self._values = kernel_ma_values
            return kernel_ma_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"KernelMA計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に初期化済みの配列を返す
            if 'close' in locals():
                self._values = np.zeros_like(close)
                return self._values
            return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_bandwidth(self) -> np.ndarray:
        """
        動的バンド幅の値を取得する
        
        Returns:
            np.ndarray: バンド幅の値
        """
        if self._result is None:
            return np.array([])
        return self._result.bandwidth
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        上下バンドの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上側バンド, 下側バンド)の値
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        return self._result.upper_band, self._result.lower_band
    
    def get_slope(self) -> np.ndarray:
        """
        傾きの値を取得する
        
        Returns:
            np.ndarray: 傾きの値
        """
        if self._result is None:
            return np.array([])
        return self._result.slope
    
    def get_trend_direction(self) -> np.ndarray:
        """
        トレンド方向を取得する
        
        Returns:
            np.ndarray: トレンド方向（1=上昇、-1=下降、0=横ばい）
        """
        if self._result is None:
            return np.array([])
            
        slope = self._result.slope
        direction = np.zeros_like(slope)
        
        # 傾きの符号に基づいてトレンド方向を決定
        direction[slope > 0.0001] = 1    # 上昇トレンド
        direction[slope < -0.0001] = -1  # 下降トレンド
        
        return direction 