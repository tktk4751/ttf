#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, prange, float64, int32

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@jit(nopython=True)
def calculate_dft_dc_numba(
    price: np.ndarray,
    window: int = 50,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1,
    min_period: int = 8,
    max_period: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    離散フーリエ変換を使ったドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        window: 変換ウィンドウサイズ
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        min_period: 計算する最小周期
        max_period: 計算する最大周期
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    hp = np.zeros(n)
    cleaned_data = np.zeros(n)
    period_power = np.zeros((n, max_period + 1))
    db = np.zeros((n, max_period + 1))
    dom_period = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    # CosinePart/SinePart保存用の配列
    # これらは各時点で再計算される
    cosine_part = np.zeros(max_period + 1)
    sine_part = np.zeros(max_period + 1)
    power = np.zeros(max_period + 1)
    
    # メインループ
    for i in range(n):
        # 十分なデータがない場合は初期値を設定
        if i <= 5:
            hp[i] = price[i]
            cleaned_data[i] = price[i]
            dom_cycle[i] = min_output  # 最小出力値に設定
            continue
        
        # ハイパスフィルター（カットオフ期間40）を適用してデータをデトレンド
        per = 2 * pi / 40
        cos_per = np.cos(per)
        alpha1 = 0
        if cos_per != 0:
            alpha1 = (1 - np.sin(per)) / cos_per
        
        hp[i] = 0.5 * (1 + alpha1) * (price[i] - price[i-1]) + alpha1 * hp[i-1]
        
        # 単純なFIRフィルターでデータをスムージング
        cleaned_data[i] = (hp[i] + 2 * hp[i-1] + 3 * hp[i-2] + 3 * hp[i-3] + 2 * hp[i-4] + hp[i-5]) / 12
        
        # ウィンドウサイズ分のデータが揃っていない場合は計算をスキップ
        if i < window:
            dom_cycle[i] = min_output
            continue
        
        # DFT計算
        # 各周期ごとにコサイン成分とサイン成分を計算
        for period in range(min_period, max_period + 1):
            cosine_part[period] = 0.0
            sine_part[period] = 0.0
            
            for k in range(window):
                if i - k >= 0:
                    cycle_per = 2 * pi * k / period
                    cosine_part[period] += cleaned_data[i-k] * np.cos(cycle_per)
                    sine_part[period] += cleaned_data[i-k] * np.sin(cycle_per)
            
            # パワースペクトル計算
            power[period] = cosine_part[period]**2 + sine_part[period]**2
            period_power[i, period] = power[period]
        
        # 最大パワーを求めて正規化
        max_power = 0.0
        for period in range(min_period, max_period + 1):
            if power[period] > max_power:
                max_power = power[period]
        
        # パワーレベルを正規化してdBに変換
        for period in range(min_period, max_period + 1):
            if max_power > 0 and power[period] > 0:
                # dB変換：10 * log10(0.01 / (1 - 0.99 * power / max_power))
                db_val = -10 * np.log10(0.01 / (1 - 0.99 * power[period] / max_power))
                if db_val > 20:
                    db_val = 20
                db[i, period] = db_val
        
        # 重心法（CG）を使用して優勢周期を計算
        numerator = 0.0
        denominator = 0.0
        
        for period in range(min_period, max_period + 1):
            if db[i, period] < 3:
                three_minus = 3 - db[i, period]
                numerator += period * three_minus
                denominator += three_minus
        
        if denominator > 0:
            dom_period[i] = numerator / denominator
        else:
            # 前回の値を使用
            if i > 0:
                dom_period[i] = dom_period[i-1]
            else:
                dom_period[i] = (min_period + max_period) / 2  # デフォルト値
        
        # 周期値をCycPartで調整し、出力値を計算
        cycle_value = np.ceil(dom_period[i] * cycle_part)
        
        # 出力値を制限
        if cycle_value > max_output:
            dom_cycle[i] = max_output
        elif cycle_value < min_output:
            dom_cycle[i] = min_output
        else:
            dom_cycle[i] = cycle_value
    
    # 生の周期値と平滑化周期値（この場合は同じ）を保存
    raw_period = np.copy(dom_period)
    smooth_period = np.copy(dom_period)
    
    return dom_cycle, raw_period, smooth_period


class EhlersDFTDC(EhlersDominantCycle):
    """
    エーラーズの離散フーリエ変換（Discrete Fourier Transform）を使ったドミナントサイクル検出
    
    このアルゴリズムは離散フーリエ変換を使用して周波数領域で周期を検出します。
    高度な数学的手法により、非常に正確な周期検出を実現しています。
    
    特徴:
    - 離散フーリエ変換による高精度な周波数解析
    - 複数の周期が混在する場合でも優勢な周期を抽出可能
    - 重心法を用いた優勢周期の計算
    """
    
    def __init__(
        self,
        window: int = 50,
        cycle_part: float = 0.5,
        min_period: int = 8,
        max_period: int = 50,
        max_output: int = 34,
        min_output: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            window: 変換ウィンドウサイズ（デフォルト: 50）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            min_period: 計算する最小周期（デフォルト: 8）
            max_period: 計算する最大周期（デフォルト: 50）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
        """
        super().__init__(
            f"EhlersDFTDC({window}, {cycle_part}, {min_period}, {max_period})",
            cycle_part,
            max_period,
            min_period,
            max_output,
            min_output
        )
        self.window = window
        self.min_period = min_period
        self.max_period = max_period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        離散フーリエ変換を使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                price = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    price = data[:, 3]  # close
                else:
                    price = data  # 1次元配列として扱う
            
            # Numba関数を使用してドミナントサイクルを計算
            dom_cycle, raw_period, smooth_period = calculate_dft_dc_numba(
                price,
                self.window,
                self.cycle_part,
                self.max_output,
                self.min_output,
                self.min_period,
                self.max_period
            )
            
            # 結果を保存
            self._result = DominantCycleResult(
                values=dom_cycle,
                raw_period=raw_period,
                smooth_period=smooth_period
            )
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersDFTDC計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 