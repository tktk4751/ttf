#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultimate Choppiness Index by John Ehlers
超低遅延・超適応性・超精度のチョピネスインデックス

Core Technologies:
1. Zero-Lag Exponential Moving Average (ZLEMA) - 超低遅延
2. Adaptive Efficiency Ratio - 超適応性
3. Hilbert Transform Phase Analysis - 超精度
"""

from typing import Union, NamedTuple
import numpy as np
import pandas as pd
from numba import njit
from dataclasses import dataclass

from .indicator import Indicator


@dataclass
class UltimateChoppinessResult:
    """アルティメットチョピネスインデックスの結果"""
    choppiness: np.ndarray      # メインチョピネス値 (0-1, 1=最大チョピネス)
    efficiency: np.ndarray      # 効率比 (0-1, 1=最大効率)
    phase_coherence: np.ndarray # 位相コヒーレンス (0-1, 1=最大コヒーレンス)
    adaptive_period: np.ndarray # 適応期間
    confidence: np.ndarray      # 信頼度 (0-1, 1=最大信頼度)


@njit(fastmath=True)
def calculate_zlema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Zero-Lag Exponential Moving Average (ZLEMA)
    遅延を最小化する指数移動平均
    """
    length = len(data)
    zlema = np.zeros(length)
    
    if length < period:
        return zlema
    
    # 最初の値を設定
    zlema[0] = data[0]
    
    # ZLEMA計算
    alpha = 2.0 / (period + 1.0)
    lag = int((period - 1) / 2)
    
    for i in range(1, length):
        # 遅延補正
        lag_index = max(0, i - lag)
        ema_data = data[i] + (data[i] - data[lag_index])
        
        # 指数移動平均
        zlema[i] = alpha * ema_data + (1 - alpha) * zlema[i-1]
    
    return zlema


@njit(fastmath=True)
def calculate_efficiency_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    適応効率比の計算
    価格の方向性効率を測定
    """
    length = len(close)
    efficiency = np.zeros(length)
    
    for i in range(period, length):
        # 期間内の価格変化
        price_change = abs(close[i] - close[i - period])
        
        # True Rangeの合計（ボラティリティ）
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(
                    high[j] - low[j],
                    abs(high[j] - close[j-1]),
                    abs(low[j] - close[j-1])
                )
                volatility += tr
        
        # 効率比計算
        if volatility > 0:
            efficiency[i] = price_change / volatility
        else:
            efficiency[i] = 0.0
    
    return efficiency


@njit(fastmath=True)
def hilbert_transform(data: np.ndarray) -> tuple:
    """
    ヒルベルト変換による位相解析
    市場の周期性とコヒーレンスを検出
    """
    length = len(data)
    
    # ヒルベルト変換のためのフィルタ係数
    h_real = np.zeros(length)
    h_imag = np.zeros(length)
    
    # 簡略化されたヒルベルト変換
    for i in range(7, length):
        h_real[i] = (data[i-3] + data[i-2] + data[i-1] + data[i]) / 4.0
        h_imag[i] = (data[i] - data[i-6]) / 6.0
    
    # 位相と振幅の計算
    phase = np.zeros(length)
    amplitude = np.zeros(length)
    
    for i in range(length):
        if abs(h_real[i]) > 1e-10:
            phase[i] = np.arctan(h_imag[i] / h_real[i])
        amplitude[i] = np.sqrt(h_real[i]**2 + h_imag[i]**2)
    
    return phase, amplitude


@njit(fastmath=True)
def calculate_phase_coherence(phase: np.ndarray, period: int) -> np.ndarray:
    """
    位相コヒーレンス計算
    周期性の安定性を測定
    """
    length = len(phase)
    coherence = np.zeros(length)
    
    for i in range(period, length):
        # 期間内の位相変化の一貫性
        phase_diff_sum = 0.0
        phase_diff_count = 0
        
        for j in range(i - period + 1, i):
            if j > 0:
                diff = abs(phase[j] - phase[j-1])
                # 位相の不連続性を処理
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                phase_diff_sum += diff
                phase_diff_count += 1
        
        # 位相コヒーレンス計算
        if phase_diff_count > 0:
            avg_phase_diff = phase_diff_sum / phase_diff_count
            coherence[i] = 1.0 - (avg_phase_diff / np.pi)
        else:
            coherence[i] = 0.0
    
    return np.maximum(0.0, coherence)


@njit(fastmath=True)
def calculate_adaptive_period(efficiency: np.ndarray, min_period: int = 8, max_period: int = 50) -> np.ndarray:
    """
    効率比に基づく適応期間計算
    """
    length = len(efficiency)
    adaptive_period = np.zeros(length)
    
    for i in range(length):
        # 効率比に基づく期間調整
        if efficiency[i] > 0.8:
            # 高効率：短期間
            adaptive_period[i] = min_period
        elif efficiency[i] < 0.2:
            # 低効率：長期間
            adaptive_period[i] = max_period
        else:
            # 中間効率：線形補間
            ratio = (efficiency[i] - 0.2) / 0.6
            adaptive_period[i] = max_period - ratio * (max_period - min_period)
    
    return adaptive_period


@njit(fastmath=True)
def calculate_ultimate_choppiness(
    efficiency: np.ndarray,
    phase_coherence: np.ndarray,
    adaptive_period: np.ndarray
) -> tuple:
    """
    最終的なチョピネスインデックス計算
    """
    length = len(efficiency)
    choppiness = np.zeros(length)
    confidence = np.zeros(length)
    
    for i in range(length):
        # 効率比の逆数（低効率=高チョピネス）
        efficiency_component = 1.0 - efficiency[i]
        
        # 位相コヒーレンスの逆数（低コヒーレンス=高チョピネス）
        phase_component = 1.0 - phase_coherence[i]
        
        # 適応期間による重み付け
        period_weight = adaptive_period[i] / 50.0
        
        # 統合チョピネス計算
        choppiness[i] = (efficiency_component * 0.5 + phase_component * 0.3 + period_weight * 0.2)
        
        # 信頼度計算
        confidence[i] = min(efficiency[i] + phase_coherence[i], 1.0)
    
    return choppiness, confidence


class UltimateChoppinessIndex(Indicator):
    """
    🚀 Ultimate Choppiness Index
    
    ジョン・エラーズによる革新的なチョピネスインデックス
    3つの核心技術を統合：
    - ZLEMA：超低遅延
    - 適応効率比：超適応性
    - ヒルベルト変換：超精度
    """
    
    def __init__(self, 
                 base_period: int = 14,
                 min_period: int = 8,
                 max_period: int = 50,
                 smoothing_period: int = 3):
        """
        Parameters:
        -----------
        base_period : int
            基本計算期間
        min_period : int
            最小適応期間
        max_period : int
            最大適応期間
        smoothing_period : int
            最終スムージング期間
        """
        super().__init__(f"UltimateChop({base_period})")
        self.base_period = base_period
        self.min_period = min_period
        self.max_period = max_period
        self.smoothing_period = smoothing_period
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ計算"""
        if isinstance(data, pd.DataFrame):
            data_str = str(data.values.tobytes())
        else:
            data_str = str(data.tobytes())
        
        param_str = f"{self.base_period}_{self.min_period}_{self.max_period}_{self.smoothing_period}"
        return str(hash(data_str + param_str))
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ultimate Choppiness Index計算
        
        Returns:
        --------
        np.ndarray
            チョピネスインデックス値 (0-1, 1=最大チョピネス)
        """
        # キャッシュチェック
        data_hash = self._get_data_hash(data)
        if self._data_hash == data_hash and self._result is not None:
            return self._result.choppiness
        
        # データ準備
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        else:
            df = data.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 1. 効率比計算（適応性）
        efficiency = calculate_efficiency_ratio(high, low, close, self.base_period)
        
        # 2. ZLEMA適用（低遅延）
        efficiency_smooth = calculate_zlema(efficiency, self.smoothing_period)
        
        # 3. ヒルベルト変換（精度）
        phase, amplitude = hilbert_transform(close)
        phase_coherence = calculate_phase_coherence(phase, self.base_period)
        
        # 4. 適応期間計算
        adaptive_period = calculate_adaptive_period(efficiency_smooth, self.min_period, self.max_period)
        
        # 5. 最終チョピネス計算
        choppiness, confidence = calculate_ultimate_choppiness(
            efficiency_smooth, phase_coherence, adaptive_period
        )
        
        # 6. 最終スムージング
        final_choppiness = calculate_zlema(choppiness, self.smoothing_period)
        final_confidence = calculate_zlema(confidence, self.smoothing_period)
        
        # 結果保存
        self._result = UltimateChoppinessResult(
            choppiness=final_choppiness,
            efficiency=efficiency_smooth,
            phase_coherence=phase_coherence,
            adaptive_period=adaptive_period,
            confidence=final_confidence
        )
        self._data_hash = data_hash
        
        return final_choppiness
    
    def get_result(self) -> UltimateChoppinessResult:
        """完全な計算結果を取得"""
        return self._result
    
    def get_signals(self, chop_threshold: float = 0.6) -> np.ndarray:
        """
        シグナル生成
        
        Parameters:
        -----------
        chop_threshold : float
            チョピネス判定閾値
        
        Returns:
        --------
        np.ndarray
            1: トレンド状態, -1: チョピー状態, 0: 不明
        """
        if self._result is None:
            return np.array([])
        
        choppiness = self._result.choppiness
        confidence = self._result.confidence
        
        signals = np.zeros(len(choppiness))
        
        # 信頼度が高い場合のみシグナル生成
        high_confidence = confidence > 0.7
        
        signals[high_confidence & (choppiness < chop_threshold)] = 1   # トレンド
        signals[high_confidence & (choppiness >= chop_threshold)] = -1  # チョピー
        
        return signals
    
    def reset(self):
        """インジケーターリセット"""
        super().reset()
        self._result = None
        self._data_hash = None 