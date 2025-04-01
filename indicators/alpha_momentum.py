#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_bollinger_bands import AlphaBollingerBands
from .alpha_keltner_channel import AlphaKeltnerChannel


@dataclass
class AlphaMomentumResult:
    """AlphaMomentumの計算結果"""
    momentum: np.ndarray     # モメンタム値
    sqz_on: np.ndarray       # スクイーズ状態（True=オン）
    sqz_off: np.ndarray      # スクイーズ解放状態（True=オフ）
    no_sqz: np.ndarray       # 非スクイーズ状態（True=非スクイーズ）
    bb_upper: np.ndarray     # アルファボリンジャー上限
    bb_lower: np.ndarray     # アルファボリンジャー下限
    kc_upper: np.ndarray     # アルファケルトナー上限
    kc_lower: np.ndarray     # アルファケルトナー下限


@jit(nopython=True)
def calculate_rolling_max_min(high: np.ndarray, low: np.ndarray, length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的期間内の最高値と最安値を計算（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        length: 各位置での期間（動的）
        
    Returns:
        (highest_high, lowest_low) のタプル
    """
    data_length = len(high)
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    
    for i in range(data_length):
        current_length = int(length[i])
        if current_length < 2:
            current_length = 2
            
        start_idx = max(0, i - current_length + 1)
        
        # 直接ループで最大値と最小値を計算（スライスを避ける）
        max_val = high[start_idx]
        min_val = low[start_idx]
        
        for j in range(start_idx + 1, i + 1):
            if high[j] > max_val:
                max_val = high[j]
            if low[j] < min_val:
                min_val = low[j]
        
        highest_high[i] = max_val
        lowest_low[i] = min_val
    
    return highest_high, lowest_low


@jit(nopython=True)
def calculate_linear_regression(x: np.ndarray, length: np.ndarray) -> np.ndarray:
    """
    動的期間での線形回帰を計算（高速化版）
    
    Args:
        x: 入力配列
        length: 各位置での期間（動的）
        
    Returns:
        線形回帰の結果
    """
    data_length = len(x)
    result = np.zeros_like(x)
    
    for i in range(data_length):
        current_length = int(length[i])
        if current_length < 2:
            current_length = 2
            
        start_idx = max(0, i - current_length + 1)
        window_size = i - start_idx + 1
        
        # 最低2点必要
        if window_size < 2:
            result[i] = 0
            continue
        
        # x軸の値（0, 1, 2, ...）
        x_sum = 0
        x_squared_sum = 0
        
        # 事前計算（等差数列の和と二乗和）
        n = window_size
        x_sum = (n - 1) * n / 2  # 0からn-1までの和
        x_squared_sum = (n - 1) * n * (2 * n - 1) / 6  # 0からn-1までの二乗和
        
        # y値の合計とxy積の合計を計算
        y_sum = 0.0
        xy_sum = 0.0
        
        for j in range(window_size):
            y_val = x[start_idx + j]
            y_sum += y_val
            xy_sum += j * y_val
        
        # 平均値
        x_mean = x_sum / window_size
        y_mean = y_sum / window_size
        
        # 線形回帰の係数を計算
        numerator = xy_sum - x_mean * y_sum
        denominator = x_squared_sum - x_mean * x_sum
        
        # ゼロ除算を防止
        if abs(denominator) < 1e-10:
            result[i] = 0
            continue
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 結果を計算（傾きを使用）
        result[i] = slope * (window_size - 1) + intercept
    
    return result


@jit(nopython=True)
def calculate_squeeze_states(
    bb_lower: np.ndarray,
    bb_upper: np.ndarray,
    kc_lower: np.ndarray,
    kc_upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スクイーズ状態を計算（高速化版）
    
    Args:
        bb_lower: BBの下限
        bb_upper: BBの上限
        kc_lower: KCの下限
        kc_upper: KCの上限
        
    Returns:
        (sqz_on, sqz_off, no_sqz) のタプル
    """
    length = len(bb_lower)
    # 整数型を使用（Numba対応）
    sqz_on = np.zeros(length, dtype=np.int32)
    sqz_off = np.zeros(length, dtype=np.int32)
    no_sqz = np.zeros(length, dtype=np.int32)
    
    for i in range(length):
        # NaNをスキップ
        if (np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(kc_upper[i])):
            continue
            
        # スクイーズ状態の判定
        if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
            sqz_on[i] = 1
        elif bb_lower[i] < kc_lower[i] and bb_upper[i] > kc_upper[i]:
            sqz_off[i] = 1
        else:
            no_sqz[i] = 1
    
    return sqz_on, sqz_off, no_sqz


@jit(nopython=True)
def calculate_alpha_momentum(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    alpha_bb_middle: np.ndarray,
    alpha_kc_middle: np.ndarray,
    dynamic_length: np.ndarray
) -> np.ndarray:
    """
    アルファモメンタムを計算（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        alpha_bb_middle: アルファボリンジャーの中心線
        alpha_kc_middle: アルファケルトナーの中心線
        dynamic_length: 動的な期間
        
    Returns:
        モメンタム値の配列
    """
    # 最高値と最安値の計算
    highest_high, lowest_low = calculate_rolling_max_min(high, low, dynamic_length)
    
    # 価格レベルの計算（直接計算）
    length = len(close)
    avg_ref = np.zeros(length)
    diff = np.zeros(length)
    
    for i in range(length):
        # 平均値の計算
        avg_hl = (highest_high[i] + lowest_low[i]) / 2.0
        avg_ref[i] = (avg_hl + alpha_bb_middle[i] + alpha_kc_middle[i]) / 3.0
        
        # デリファレンスされた価格
        diff[i] = close[i] - avg_ref[i]
    
    # 線形回帰によるモメンタムの計算
    momentum = calculate_linear_regression(diff, dynamic_length)
    
    return momentum


class AlphaMomentum(Indicator):
    """
    アルファモメンタム（Alpha Momentum）インジケーター
    
    特徴:
    - アルファボリンジャーバンドとアルファケルトナーチャネルを組み合わせた高度なスクイーズ検出
    - 効率比（ER）に基づくすべてのパラメータの動的最適化
    - 最適化されたモメンタム計算による高精度なシグナル
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - より狭いバンド幅でトレンドをより早く検出
      - 短い期間でモメンタムをより早く検出
    - トレンド弱い（ER低い）:
      - より広いバンド幅でノイズによる誤シグナルを減少
      - 長い期間でモメンタムをより安定させる
    """
    
    def __init__(
        self,
        er_period: int = 21,
        bb_max_period: int = 55,
        bb_min_period: int = 13,
        kc_max_period: int = 55,
        kc_min_period: int = 13,
        bb_max_mult: float = 2,
        bb_min_mult: float = 1.0,
        kc_max_mult: float = 3,
        kc_min_mult: float = 1.0,
        max_length: int = 34,
        min_length: int = 8,
        alma_offset: float = 0.85,
        alma_sigma: float = 6
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            
            bb_max_period: ボリンジャーバンドの標準偏差計算の最大期間（デフォルト: 55）
            bb_min_period: ボリンジャーバンドの標準偏差計算の最小期間（デフォルト: 13）
            
            kc_max_period: ケルトナーチャネルのAlphaATR最大期間（デフォルト: 55）
            kc_min_period: ケルトナーチャネルのAlphaATR最小期間（デフォルト: 13）
            
            bb_max_mult: ボリンジャーバンドの標準偏差乗数の最大値（デフォルト: 2.5）
            bb_min_mult: ボリンジャーバンドの標準偏差乗数の最小値（デフォルト: 1.0）
            
            kc_max_mult: ケルトナーチャネルのATR乗数の最大値（デフォルト: 2.5）
            kc_min_mult: ケルトナーチャネルのATR乗数の最小値（デフォルト: 1.0）
            
            max_length: モメンタム計算の最大期間（デフォルト: 34）
            min_length: モメンタム計算の最小期間（デフォルト: 8）
            
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
        """
        super().__init__(f"AlphaMomentum({er_period})")
        
        self.er_period = er_period
        self.max_length = max_length
        self.min_length = min_length
        
        # アルファボリンジャーバンドの初期化
        self.alpha_bb = AlphaBollingerBands(
            er_period=er_period,
            max_kama_period=bb_max_period,
            min_kama_period=bb_min_period,
            std_dev_period=bb_max_period,  # 標準偏差期間は最大期間を使用
            max_multiplier=bb_max_mult,
            min_multiplier=bb_min_mult
        )
        
        # アルファケルトナーチャネルの初期化
        self.alpha_kc = AlphaKeltnerChannel(
            er_period=er_period,
            max_kama_period=kc_max_period,
            min_kama_period=kc_min_period,
            max_atr_period=kc_max_period,
            min_atr_period=kc_min_period,
            max_multiplier=kc_max_mult,
            min_multiplier=kc_min_mult,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファモメンタムを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            モメンタム値の配列
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
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
            self._validate_period(self.er_period, data_length)
            
            # アルファボリンジャーバンドの計算
            bb_middle = self.alpha_bb.calculate(data)
            bb_middle, bb_upper, bb_lower = self.alpha_bb.get_bands()
            
            # アルファケルトナーチャネルの計算
            kc_middle = self.alpha_kc.calculate(data)
            kc_middle, kc_upper, kc_lower = self.alpha_kc.get_bands()
            
            # 効率比の取得（どちらかのインジケーターから）
            er = self.alpha_bb.get_efficiency_ratio()
            
            # 動的なモメンタム計算期間の決定
            dynamic_length = self.min_length + (1.0 - er) * (self.max_length - self.min_length)
            dynamic_length = np.round(dynamic_length).astype(np.int32)
            
            # アルファモメンタムの計算
            momentum = calculate_alpha_momentum(
                high, low, close,
                bb_middle, kc_middle,
                dynamic_length
            )
            
            # スクイーズ状態の判定
            sqz_on, sqz_off, no_sqz = calculate_squeeze_states(
                bb_lower, bb_upper,
                kc_lower, kc_upper
            )
            
            # 結果の保存
            self._result = AlphaMomentumResult(
                momentum=momentum,
                sqz_on=sqz_on,
                sqz_off=sqz_off,
                no_sqz=no_sqz,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                kc_upper=kc_upper,
                kc_lower=kc_lower
            )
            
            # 基底クラスの要件を満たすため
            self._values = momentum
            return momentum
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaMomentum計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時でも最低限の結果を初期化
            if 'close' in locals():
                empty_array = np.zeros_like(close)
                self._values = empty_array
                return empty_array
            return np.array([])
    
    def get_squeeze_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スクイーズ状態を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (sqz_on, sqz_off, no_sqz)のタプル
                - sqz_on: スクイーズオン状態（True=スクイーズ中）
                - sqz_off: スクイーズオフ状態（True=スクイーズ解放直後）
                - no_sqz: 非スクイーズ状態（True=通常状態）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.sqz_on, self._result.sqz_off, self._result.no_sqz
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        バンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (bb_upper, bb_lower, kc_upper, kc_lower)のタプル
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.bb_upper,
            self._result.bb_lower,
            self._result.kc_upper,
            self._result.kc_lower
        )
    
    def get_momentum(self) -> np.ndarray:
        """
        モメンタム値を取得する
        
        Returns:
            np.ndarray: モメンタム値の配列
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.momentum
    
    def get_bollinger(self) -> AlphaBollingerBands:
        """
        アルファボリンジャーバンドのインスタンスを取得する
        
        Returns:
            AlphaBollingerBands: アルファボリンジャーバンドのインスタンス
        """
        return self.alpha_bb
    
    def get_keltner(self) -> AlphaKeltnerChannel:
        """
        アルファケルトナーチャネルのインスタンスを取得する
        
        Returns:
            AlphaKeltnerChannel: アルファケルトナーチャネルのインスタンス
        """
        return self.alpha_kc 