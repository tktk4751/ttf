#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange, vectorize
import math

from .indicator import Indicator


@dataclass
class HurstExponentResult:
    """ハースト指数の計算結果"""
    values: np.ndarray            # ハースト指数値（0-1の範囲）
    rs_values: np.ndarray         # R/S統計量
    trend_strength: np.ndarray    # トレンド強度（0.5からの距離）


@njit(fastmath=True)
def calculate_rs(data: np.ndarray, lag: int) -> float:
    """
    特定のラグでのR/S（Range/Standard Deviation）統計量を計算する
    
    Args:
        data: 価格データの配列
        lag: ラグ（期間）
    
    Returns:
        R/S統計量
    """
    # データを差分に変換
    returns = np.diff(data[-lag:])
    if len(returns) <= 1:
        return np.nan
    
    # 平均値を引いた累積偏差を計算
    mean_return = np.mean(returns)
    mean_adjusted = returns - mean_return
    cumulative = np.cumsum(mean_adjusted)
    
    # 範囲R（最大値 - 最小値）を計算
    r = np.max(cumulative) - np.min(cumulative)
    
    # 標準偏差Sを計算
    s = np.std(returns)
    
    # R/Sが0または無限大になるのを防ぐ
    if s == 0 or r == 0:
        return np.nan
    
    # R/S統計量を返す
    return r / s


@njit(fastmath=True)
def calculate_hurst_for_point(data: np.ndarray, min_lag: int = 10, max_lag: int = 100, step: int = 10) -> float:
    """
    単一のポイントに対するハースト指数を計算する
    
    Args:
        data: 価格データの配列
        min_lag: 最小ラグ（期間）
        max_lag: 最大ラグ（期間）
        step: ラグの増分
    
    Returns:
        ハースト指数（0-1の範囲）
    """
    if len(data) < max_lag:
        return np.nan
    
    # 各ラグでのR/S統計量を計算
    lags = np.arange(min_lag, min(max_lag+1, len(data)), step)
    if len(lags) < 2:
        return np.nan
    
    rs_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        rs_values[i] = calculate_rs(data, lag)
    
    # 無効な値を除外
    valid_indices = ~np.isnan(rs_values)
    if np.sum(valid_indices) < 2:
        return np.nan
    
    lag_logs = np.log(lags[valid_indices])
    rs_logs = np.log(rs_values[valid_indices])
    
    # 線形回帰で傾きを計算（ハースト指数）
    n = len(lag_logs)
    sum_x = np.sum(lag_logs)
    sum_y = np.sum(rs_logs)
    sum_xy = np.sum(lag_logs * rs_logs)
    sum_xx = np.sum(lag_logs * lag_logs)
    
    if n * sum_xx - sum_x * sum_x == 0:
        return np.nan
    
    # 傾き = ハースト指数
    h = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    
    # 0-1の範囲に収める
    return min(max(h, 0.0), 1.0)


@njit(fastmath=True, parallel=True)
def calculate_hurst_exponent(
    data: np.ndarray,
    window: int = 100,
    min_lag: int = 10,
    max_lag: int = 50,
    step: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ローリングウィンドウを使用してハースト指数を計算する
    
    Args:
        data: 価格データの配列
        window: 分析ウィンドウのサイズ
        min_lag: 最小ラグ（期間）
        max_lag: 最大ラグ（期間）
        step: ラグの増分
    
    Returns:
        (ハースト指数の配列, R/S統計量の配列, トレンド強度の配列)のタプル
    """
    size = len(data)
    hurst_values = np.zeros(size)
    rs_values = np.zeros(size)
    trend_strength = np.zeros(size)
    
    # ウィンドウサイズよりも小さいインデックスではNaNを設定
    hurst_values[:window] = np.nan
    rs_values[:window] = np.nan
    trend_strength[:window] = np.nan
    
    for i in prange(window, size):
        window_data = data[i-window:i]
        h = calculate_hurst_for_point(window_data, min_lag, max_lag, step)
        
        hurst_values[i] = h
        rs_values[i] = calculate_rs(window_data, min_lag)
        
        # トレンド強度：0.5からの距離（絶対値）
        if not np.isnan(h):
            trend_strength[i] = abs(h - 0.5)
    
    return hurst_values, rs_values, trend_strength


class HurstExponent(Indicator):
    """
    ハースト指数インジケーター
    
    ハースト指数は、時系列データの長期記憶性（持続性）を測定します。
    - H = 0.5 : ランダムウォーク（無相関）
    - 0.5 < H < 1 : 持続的な時系列（トレンドが続く傾向）
    - 0 < H < 0.5 : 反持続的な時系列（平均回帰の傾向）
    
    特徴:
    - R/S分析を使用してハースト指数を計算
    - ローリングウィンドウ方式で時間変化するハースト指数を提供
    - トレンド強度の指標としても使用可能
    """
    
    def __init__(
        self,
        window: int = 100,          # 分析ウィンドウサイズ
        min_lag: int = 10,          # 最小ラグ（期間）
        max_lag: int = 50,          # 最大ラグ（期間）
        step: int = 5               # ラグの増分
    ):
        """
        コンストラクタ
        
        Args:
            window: 分析ウィンドウのサイズ（デフォルト: 100）
            min_lag: 最小ラグ（期間）（デフォルト: 10）
            max_lag: 最大ラグ（期間）（デフォルト: 50）
            step: ラグの増分（デフォルト: 5）
        """
        super().__init__(
            f"HurstExponent({window})"
        )
        
        self.window = window
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.step = step
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            if data.ndim == 2 and data.shape[1] >= 4:
                data_hash = hash(tuple(data[:, 3]))  # close column
            else:
                data_hash = hash(tuple(data.flatten()))
        
        # パラメータ値を含める
        param_str = f"{self.window}_{self.min_lag}_{self.max_lag}_{self.step}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ハースト指数を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ハースト指数値の配列（0-1の範囲）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    raise ValueError("DataFrameには'close'カラムが必要です")
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    prices = data[:, 3]  # close column
                elif data.ndim == 1:
                    prices = data
                else:
                    raise ValueError("サポートされていないデータ形式です")
            
            # データ長の検証
            data_length = len(prices)
            if data_length < self.window:
                raise ValueError(f"データ長({data_length})がウィンドウサイズ({self.window})より小さいです")
            
            # ハースト指数の計算
            hurst_values, rs_values, trend_strength = calculate_hurst_exponent(
                prices,
                self.window,
                self.min_lag,
                self.max_lag,
                self.step
            )
            
            # 結果を保存
            self._result = HurstExponentResult(
                values=hurst_values,
                rs_values=rs_values,
                trend_strength=trend_strength
            )
            
            self._values = hurst_values  # 基底クラスの要件を満たすため
            
            return hurst_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ハースト指数計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_rs_values(self) -> np.ndarray:
        """
        R/S統計量を取得する
        
        Returns:
            R/S統計量の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.rs_values
    
    def get_trend_strength(self) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列 (0-0.5の範囲、大きいほど強いトレンド性)
        """
        if self._result is None:
            return np.array([])
        return self._result.trend_strength
    
    def get_trend_type(self) -> np.ndarray:
        """
        トレンドタイプを取得する
        
        Returns:
            トレンドタイプの配列:
            1: 持続的トレンド (H > 0.5)
            0: ランダムウォーク (H = 0.5)
            -1: 反持続的トレンド (H < 0.5)
        """
        if self._result is None:
            return np.array([])
        
        values = self._result.values
        trend_type = np.zeros_like(values)
        
        # トレンドタイプを判定
        # 0.5より大きい: 持続的トレンド
        # 0.5未満: 反持続的トレンド
        # 誤差を考慮して0.5±0.01の範囲はランダムウォークとする
        trend_type = np.where(np.isnan(values), np.nan, 
                     np.where(values > 0.51, 1, 
                     np.where(values < 0.49, -1, 0)))
        
        return trend_type
    
    def get_signals(self, threshold: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
        """
        トレンドシグナルを取得する
        
        Args:
            threshold: シグナル生成のためのしきい値（デフォルト: 0.65）
        
        Returns:
            (上昇トレンドシグナル, 下降トレンドシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            empty = np.array([])
            return empty, empty
        
        # 前回値との差分を計算（傾き）
        prev_values = np.roll(self._values, 1)
        prev_values[0] = self._values[0]
        
        slope = self._values - prev_values
        
        # ハースト指数が閾値を超え、傾きが正なら上昇トレンド
        uptrend = np.where(
            (self._values > threshold) & (slope > 0),
            1, 0
        )
        
        # ハースト指数が閾値を超え、傾きが負なら下降トレンド
        downtrend = np.where(
            (self._values > threshold) & (slope < 0),
            1, 0
        )
        
        return uptrend, downtrend
    
    def get_mean_reversion_signals(self, threshold: float = 0.35) -> np.ndarray:
        """
        平均回帰シグナルを取得する
        
        Args:
            threshold: シグナル生成のためのしきい値（デフォルト: 0.35）
        
        Returns:
            平均回帰シグナル（値が閾値未満の場合は1、そうでない場合は0）
        """
        if self._values is None:
            return np.array([])
        
        # ハースト指数が閾値未満なら平均回帰傾向
        mean_reversion = np.where(self._values < threshold, 1, 0)
        
        return mean_reversion
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None 