#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math

# Base classes import
try:
    from .indicator import Indicator
    from .atr import ATR
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class ATR:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return None
        def get_values(self): return np.array([])
        def get_dynamic_periods(self): return np.array([])
        def reset(self): pass
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 10.0)
        def reset(self): pass


class UltraChopTrendResult(NamedTuple):
    """ウルトラCHOPトレンド計算結果"""
    values: np.ndarray                    # メインのウルトラチョップ値（0-1の範囲）
    trend_signals: np.ndarray             # 高精度トレンド信号（1=up, -1=down, 0=range）
    current_trend: str                    # 現在のトレンド状態
    current_trend_value: int              # 現在のトレンド値
    confidence: np.ndarray                # 信頼度スコア（0-1の範囲）  
    quantum_volatility: np.ndarray        # 量子ボラティリティ指標
    fractal_dimension: np.ndarray         # フラクタル次元
    adaptive_threshold: np.ndarray        # 適応的しきい値
    multi_cycle_strength: np.ndarray      # マルチサイクル強度
    regime_state: np.ndarray              # 市場レジーム状態（0=bear, 1=bull, 2=sideways）
    latency_optimized_signal: np.ndarray  # 低遅延最適化信号
    precision_score: np.ndarray           # 精度スコア


@njit(fastmath=True, cache=True)
def calculate_quantum_volatility(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    volume: np.ndarray = None
) -> np.ndarray:
    """
    量子ボラティリティを計算 - 価格とボリュームの相関を考慮した高度なボラティリティ指標
    """
    length = len(high)
    qvol = np.zeros(length)
    
    for i in range(14, length):  # 最小14期間
        # 価格レンジの標準化
        price_ranges = high[i-13:i+1] - low[i-13:i+1]
        normalized_ranges = price_ranges / np.mean(price_ranges) if np.mean(price_ranges) > 0 else price_ranges
        
        # 変化率の計算
        returns = np.diff(close[i-13:i+1])
        vol_factor = np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 1.0
        
        # ボリューム調整（提供されている場合）
        volume_factor = 1.0
        if volume is not None and i >= 14:
            volume_ma = np.mean(volume[i-13:i+1])
            current_volume = volume[i]
            volume_factor = current_volume / volume_ma if volume_ma > 0 else 1.0
            volume_factor = np.sqrt(volume_factor)  # 非線形調整
        
        # 量子ボラティリティの計算
        range_factor = np.mean(normalized_ranges[-7:])  # 直近7期間の平均
        qvol[i] = range_factor * vol_factor * volume_factor
    
    # 0-1の範囲に正規化
    if np.max(qvol) > np.min(qvol):
        qvol = (qvol - np.min(qvol)) / (np.max(qvol) - np.min(qvol))
    
    return qvol


@njit(fastmath=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    フラクタル次元を計算 - 市場の複雑さを測定
    """
    length = len(prices)
    fractal_dim = np.full(length, 1.5)  # デフォルト値
    
    for i in range(window, length):
        price_window = prices[i-window+1:i+1]
        
        # 価格系列の長さを計算
        total_length = 0.0
        for j in range(1, len(price_window)):
            total_length += abs(price_window[j] - price_window[j-1])
        
        # 直線距離
        straight_distance = abs(price_window[-1] - price_window[0])
        
        if straight_distance > 1e-10:
            # フラクタル次元の計算（ハウスドルフ次元の近似）
            ratio = total_length / straight_distance
            if ratio > 1.0:
                fractal_dim[i] = 1.0 + math.log(ratio) / math.log(float(window))
                fractal_dim[i] = min(fractal_dim[i], 2.0)  # 上限を2に制限
            else:
                fractal_dim[i] = 1.0
    
    return fractal_dim


@njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(
    values: np.ndarray,
    volatility: np.ndarray,
    fractal_dim: np.ndarray,
    base_threshold: float = 0.5
) -> np.ndarray:
    """
    適応的しきい値の計算 - 市場状況に応じて動的に調整
    """
    length = len(values)
    threshold = np.full(length, base_threshold)
    
    for i in range(21, length):  # 21期間のルックバック
        # ボラティリティ調整
        vol_factor = volatility[i] if volatility[i] > 0 else 0.5
        
        # フラクタル次元調整
        fractal_factor = (fractal_dim[i] - 1.0) if fractal_dim[i] > 1.0 else 0.0
        
        # 過去の値の分散を考慮
        recent_values = values[i-20:i+1]
        value_std = np.std(recent_values)
        
        # 適応的調整
        volatility_adj = vol_factor * 0.3  # ボラティリティによる調整
        fractal_adj = fractal_factor * 0.2  # フラクタル次元による調整
        variance_adj = value_std * 0.2      # 分散による調整
        
        # しきい値の計算
        threshold[i] = base_threshold + volatility_adj - fractal_adj + variance_adj
        threshold[i] = max(0.1, min(0.9, threshold[i]))  # 0.1-0.9の範囲に制限
    
    return threshold


@njit(fastmath=True, cache=True)
def calculate_multi_cycle_strength(
    prices: np.ndarray,
    short_cycle: int = 8,
    medium_cycle: int = 21,
    long_cycle: int = 55
) -> np.ndarray:
    """
    マルチサイクル強度の計算 - 複数の時間軸でのトレンド強度を統合
    """
    length = len(prices)
    cycle_strength = np.zeros(length)
    
    # 各サイクルでの移動平均を計算
    for i in range(long_cycle, length):
        # 短期、中期、長期の移動平均
        short_ma = np.mean(prices[i-short_cycle+1:i+1])
        medium_ma = np.mean(prices[i-medium_cycle+1:i+1])
        long_ma = np.mean(prices[i-long_cycle+1:i+1])
        
        current_price = prices[i]
        
        # 各サイクルでの位置を計算（-1 to 1）
        short_pos = (current_price - short_ma) / short_ma if short_ma != 0 else 0
        medium_pos = (current_price - medium_ma) / medium_ma if medium_ma != 0 else 0
        long_pos = (current_price - long_ma) / long_ma if long_ma != 0 else 0
        
        # 加重平均でサイクル強度を計算
        cycle_strength[i] = (short_pos * 0.5 + medium_pos * 0.3 + long_pos * 0.2)
    
    # -1から1の範囲を0から1に正規化
    cycle_strength = (cycle_strength + 1.0) / 2.0
    cycle_strength = np.clip(cycle_strength, 0.0, 1.0)
    
    return cycle_strength


@njit(fastmath=True, cache=True)
def calculate_regime_state(
    prices: np.ndarray,
    volatility: np.ndarray,
    trend_strength: np.ndarray,
    window: int = 34
) -> np.ndarray:
    """
    市場レジーム状態の判定 - 強気、弱気、横ばい市場の識別
    """
    length = len(prices)
    regime = np.zeros(length, dtype=np.int32)  # 0=bear, 1=bull, 2=sideways
    
    for i in range(window, length):
        # 価格トレンドの計算
        price_change = (prices[i] - prices[i-window]) / prices[i-window] if prices[i-window] != 0 else 0
        
        # ボラティリティの平均
        avg_volatility = np.mean(volatility[i-window+1:i+1])
        
        # トレンド強度の平均
        avg_trend_strength = np.mean(trend_strength[i-window+1:i+1])
        
        # レジーム判定
        if price_change > 0.02 and avg_trend_strength > 0.6:  # 強気市場
            regime[i] = 1
        elif price_change < -0.02 and avg_trend_strength > 0.6:  # 弱気市場
            regime[i] = 0
        else:  # 横ばい市場
            regime[i] = 2
    
    return regime


@njit(fastmath=True, cache=True)
def calculate_low_latency_signal(
    values: np.ndarray,
    quantum_vol: np.ndarray,
    cycle_strength: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    低遅延最適化信号の計算 - 遅延を最小化しつつ精度を保つ
    """
    length = len(values)
    signal = np.zeros(length)
    
    if length < 2:
        return signal
    
    # 初期値設定
    signal[0] = values[0]
    
    for i in range(1, length):
        # 適応的アルファの計算
        vol_factor = quantum_vol[i] if quantum_vol[i] > 0 else 0.5
        trend_factor = cycle_strength[i] if cycle_strength[i] > 0 else 0.5
        
        # ボラティリティが高い時は反応を早く、低い時は遅く
        adaptive_alpha = alpha + (vol_factor - 0.5) * 0.2
        adaptive_alpha = max(0.1, min(0.7, adaptive_alpha))
        
        # トレンド強度による調整
        trend_adjustment = (trend_factor - 0.5) * 0.1
        adaptive_alpha += trend_adjustment
        adaptive_alpha = max(0.05, min(0.8, adaptive_alpha))
        
        # 低遅延EMAの計算
        signal[i] = adaptive_alpha * values[i] + (1 - adaptive_alpha) * signal[i-1]
    
    return signal


@njit(fastmath=True, cache=True)
def calculate_precision_score(
    main_signal: np.ndarray,
    quantum_vol: np.ndarray,
    fractal_dim: np.ndarray,
    cycle_strength: np.ndarray
) -> np.ndarray:
    """
    精度スコアの計算 - シグナルの信頼性を評価
    """
    length = len(main_signal)
    precision = np.zeros(length)
    
    for i in range(21, length):
        # 各要素の一貫性をチェック
        signal_consistency = 1.0 - abs(main_signal[i] - main_signal[i-1])
        vol_stability = 1.0 - abs(quantum_vol[i] - np.mean(quantum_vol[i-5:i]))
        fractal_stability = 1.0 - abs(fractal_dim[i] - np.mean(fractal_dim[i-5:i])) / 2.0
        cycle_consistency = 1.0 - abs(cycle_strength[i] - np.mean(cycle_strength[i-5:i]))
        
        # 加重平均で精度スコアを計算
        precision[i] = (signal_consistency * 0.4 + 
                       vol_stability * 0.2 + 
                       fractal_stability * 0.2 + 
                       cycle_consistency * 0.2)
        
        precision[i] = max(0.0, min(1.0, precision[i]))
    
    return precision


@njit(fastmath=True, cache=True)
def ultra_chop_core_calculation(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ウルトラチョップのコア計算 - すべての主要指標を統合計算
    """
    length = len(close)
    
    # 1. 量子ボラティリティの計算
    quantum_vol = calculate_quantum_volatility(high, low, close, volume)
    
    # 2. フラクタル次元の計算
    fractal_dim = calculate_fractal_dimension(close, 21)
    
    # 3. マルチサイクル強度の計算
    cycle_strength = calculate_multi_cycle_strength(close, 8, 21, 55)
    
    # 4. メインシグナルの計算（改良されたチョップインデックス）
    main_signal = np.zeros(length)
    for i in range(21, length):
        # 価格変動の複雑さ
        price_complexity = fractal_dim[i] - 1.0  # 0-1の範囲に調整
        
        # ボラティリティの正規化された値
        vol_normalized = quantum_vol[i]
        
        # サイクル強度
        cycle_factor = cycle_strength[i]
        
        # 統合計算
        # トレンド = 1 - (複雑さ * ボラティリティ調整 - サイクル強度)
        main_signal[i] = 1.0 - (price_complexity * (1.0 + vol_normalized) * 0.5 - cycle_factor * 0.3)
        main_signal[i] = max(0.0, min(1.0, main_signal[i]))
    
    # 5. 適応的しきい値の計算
    adaptive_threshold = calculate_adaptive_threshold(main_signal, quantum_vol, fractal_dim, 0.5)
    
    # 6. 市場レジーム状態の判定
    regime_state = calculate_regime_state(close, quantum_vol, main_signal, 34)
    
    # 7. 低遅延シグナルの計算
    low_latency_signal = calculate_low_latency_signal(main_signal, quantum_vol, cycle_strength, 0.3)
    
    # 8. 精度スコアの計算
    precision_score = calculate_precision_score(main_signal, quantum_vol, fractal_dim, cycle_strength)
    
    return (main_signal, quantum_vol, fractal_dim, cycle_strength, 
            adaptive_threshold, regime_state, low_latency_signal, precision_score)


@njit(fastmath=True, cache=True)
def calculate_ultra_trend_signals(
    main_signal: np.ndarray,
    adaptive_threshold: np.ndarray,
    precision_score: np.ndarray,
    regime_state: np.ndarray,
    min_precision: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ウルトラ高精度トレンドシグナルの計算
    """
    length = len(main_signal)
    trend_signals = np.zeros(length, dtype=np.int8)
    confidence = np.zeros(length)
    
    for i in range(1, length):
        if precision_score[i] < min_precision:
            trend_signals[i] = 0  # 精度が低い場合はニュートラル
            confidence[i] = 0.0
            continue
        
        current_value = main_signal[i]
        threshold = adaptive_threshold[i]
        regime = regime_state[i]
        
        # トレンド判定の基本ロジック
        if current_value > threshold:
            base_signal = 1  # 上昇トレンド
        elif current_value < (1.0 - threshold):
            base_signal = -1  # 下降トレンド
        else:
            base_signal = 0  # レンジ
        
        # レジーム状態による調整
        if regime == 1 and base_signal >= 0:  # 強気市場
            trend_signals[i] = 1
            confidence[i] = precision_score[i] * 0.9
        elif regime == 0 and base_signal <= 0:  # 弱気市場
            trend_signals[i] = -1
            confidence[i] = precision_score[i] * 0.9
        elif regime == 2:  # 横ばい市場
            trend_signals[i] = 0
            confidence[i] = precision_score[i] * 0.7
        else:
            trend_signals[i] = base_signal
            confidence[i] = precision_score[i] * 0.8
        
        # 信頼度の調整
        confidence[i] = max(0.0, min(1.0, confidence[i]))
    
    return trend_signals, confidence


class UltraChopTrend(Indicator):
    """
    ウルトラチョップトレンドインジケーター
    
    革新的な低遅延・高精度トレンド検出システム：
    - 量子ボラティリティ分析
    - フラクタル市場構造解析
    - マルチサイクル統合アプローチ
    - 適応的しきい値システム
    - リアルタイム市場レジーム認識
    - 低遅延最適化シグナル処理
    - 高精度信頼度評価システム
    """
    
    def __init__(
        self,
        quantum_sensitivity: float = 1.0,
        fractal_window: int = 21,
        cycle_short: int = 8,
        cycle_medium: int = 21,
        cycle_long: int = 55,
        base_threshold: float = 0.5,
        min_precision: float = 0.6,
        low_latency_alpha: float = 0.3,
        regime_window: int = 34
    ):
        """
        ウルトラチョップトレンドの初期化
        
        Args:
            quantum_sensitivity: 量子ボラティリティの感度
            fractal_window: フラクタル分析ウィンドウ
            cycle_short: 短期サイクル期間
            cycle_medium: 中期サイクル期間
            cycle_long: 長期サイクル期間
            base_threshold: ベースしきい値
            min_precision: 最小精度要求
            low_latency_alpha: 低遅延フィルターのアルファ
            regime_window: レジーム分析ウィンドウ
        """
        super().__init__(f"UltraChopTrend(q={quantum_sensitivity},f={fractal_window},c={cycle_short}-{cycle_medium}-{cycle_long})")
        
        self.quantum_sensitivity = quantum_sensitivity
        self.fractal_window = fractal_window
        self.cycle_short = cycle_short
        self.cycle_medium = cycle_medium
        self.cycle_long = cycle_long
        self.base_threshold = base_threshold
        self.min_precision = min_precision
        self.low_latency_alpha = low_latency_alpha
        self.regime_window = regime_window
        
        self._cache = {}
        self._result: Optional[UltraChopTrendResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltraChopTrendResult:
        """
        ウルトラチョップトレンドを計算
        """
        if len(data) == 0:
            return self._empty_result()
        
        try:
            # データの準備
            if isinstance(data, pd.DataFrame):
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                close = data['close'].values.astype(np.float64)
                volume = data['volume'].values.astype(np.float64) if 'volume' in data.columns else None
            else:
                high, low, close = data[:, 1], data[:, 2], data[:, 3]
                volume = data[:, 4] if data.shape[1] > 4 else None
            
            # コア計算の実行
            (main_signal, quantum_vol, fractal_dim, cycle_strength,
             adaptive_threshold, regime_state, low_latency_signal, precision_score) = ultra_chop_core_calculation(
                high, low, close, volume
            )
            
            # トレンドシグナルと信頼度の計算
            trend_signals, confidence = calculate_ultra_trend_signals(
                main_signal, adaptive_threshold, precision_score, regime_state, self.min_precision
            )
            
            # 現在のトレンド状態を決定
            current_trend_value = trend_signals[-1] if len(trend_signals) > 0 else 0
            trend_names = {-1: 'down', 0: 'range', 1: 'up'}
            current_trend = trend_names.get(current_trend_value, 'range')
            
            # 結果の作成
            result = UltraChopTrendResult(
                values=main_signal,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=current_trend_value,
                confidence=confidence,
                quantum_volatility=quantum_vol,
                fractal_dimension=fractal_dim,
                adaptive_threshold=adaptive_threshold,
                multi_cycle_strength=cycle_strength,
                regime_state=regime_state,
                latency_optimized_signal=low_latency_signal,
                precision_score=precision_score
            )
            
            self._result = result
            self._values = main_signal
            
            return result
            
        except Exception as e:
            self.logger.error(f"ウルトラチョップトレンド計算エラー: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> UltraChopTrendResult:
        """空の結果を返す"""
        return UltraChopTrendResult(
            values=np.array([]),
            trend_signals=np.array([], dtype=np.int8),
            current_trend='range',
            current_trend_value=0,
            confidence=np.array([]),
            quantum_volatility=np.array([]),
            fractal_dimension=np.array([]),
            adaptive_threshold=np.array([]),
            multi_cycle_strength=np.array([]),
            regime_state=np.array([], dtype=np.int32),
            latency_optimized_signal=np.array([]),
            precision_score=np.array([])
        )
    
    # Getter methods
    def get_values(self) -> Optional[np.ndarray]:
        """メインシグナル値を取得"""
        return self._result.values.copy() if self._result else None
    
    def get_confidence(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        return self._result.confidence.copy() if self._result else None
    
    def get_quantum_volatility(self) -> Optional[np.ndarray]:
        """量子ボラティリティを取得"""
        return self._result.quantum_volatility.copy() if self._result else None
    
    def get_fractal_dimension(self) -> Optional[np.ndarray]:
        """フラクタル次元を取得"""
        return self._result.fractal_dimension.copy() if self._result else None
    
    def get_low_latency_signal(self) -> Optional[np.ndarray]:
        """低遅延シグナルを取得"""
        return self._result.latency_optimized_signal.copy() if self._result else None
    
    def get_regime_state(self) -> Optional[np.ndarray]:
        """市場レジーム状態を取得"""
        return self._result.regime_state.copy() if self._result else None
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result = None
        self._cache = {} 