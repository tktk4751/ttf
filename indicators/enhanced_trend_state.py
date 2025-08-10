#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class EnhancedTrendStateResult:
    """Enhanced Trend State結果"""
    trend_state: np.ndarray          # 1=トレンド, 0=レンジ (バイナリ出力)
    confidence: np.ndarray           # 信頼度 (0-1)
    efficiency_ratio: np.ndarray     # 強化効率比
    choppiness_index: np.ndarray     # 強化チョピネス指数
    composite_score: np.ndarray      # 複合スコア (0-1)
    dynamic_periods: np.ndarray      # 動的期間
    volatility_factor: np.ndarray    # ボラティリティ調整係数


@njit(fastmath=True, cache=True)
def calculate_enhanced_efficiency_ratio(
    prices: np.ndarray, 
    periods: np.ndarray,
    volatility_adjustment: bool = True
) -> np.ndarray:
    """
    強化版効率比を計算（動的期間対応、ボラティリティ調整付き）
    
    Args:
        prices: 価格配列
        periods: 動的期間配列
        volatility_adjustment: ボラティリティ調整を適用するか
    
    Returns:
        強化効率比配列（0-1、高いほどトレンド）
    """
    n = len(prices)
    enhanced_er = np.zeros(n)
    
    for i in range(n):
        period = int(periods[i])
        if period < 2:
            period = 2
        
        if i >= period:
            # 方向性の変化（開始点から終了点への直線的変化）
            directional_change = abs(prices[i] - prices[i - period])
            
            # 実際の価格変動の累積（ボラティリティ）
            volatility_sum = 0.0
            for j in range(i - period, i):
                volatility_sum += abs(prices[j + 1] - prices[j])
            
            if volatility_sum > 1e-12:
                base_er = directional_change / volatility_sum
                
                if volatility_adjustment:
                    # ボラティリティ調整：低ボラティリティ時は効率比を向上
                    recent_volatility = volatility_sum / period
                    avg_price = (prices[i] + prices[i - period]) / 2.0
                    normalized_vol = recent_volatility / (avg_price + 1e-12)
                    
                    # ボラティリティが低いほど効率比にボーナス
                    vol_bonus = 1.0 / (1.0 + normalized_vol * 100.0)
                    enhanced_er[i] = min(1.0, base_er * (1.0 + vol_bonus))
                else:
                    enhanced_er[i] = min(1.0, base_er)
    
    return enhanced_er


@njit(fastmath=True, cache=True)
def calculate_enhanced_choppiness(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    periods: np.ndarray,
    atr_smoothing: bool = True
) -> np.ndarray:
    """
    強化版チョピネス指数を計算（動的期間対応、ATRスムージング付き）
    
    Args:
        high: 高値配列
        low: 安値配列
        close: 終値配列
        periods: 動的期間配列
        atr_smoothing: ATRスムージングを適用するか
    
    Returns:
        強化チョピネス指数配列（0-100、高いほどレンジ）
    """
    n = len(high)
    enhanced_chop = np.zeros(n)
    
    # True Rangeを計算
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # ATRスムージング用の配列
    atr = np.zeros(n)
    if atr_smoothing:
        # Wilder's スムージング（14期間）
        atr_period = 14
        if n > atr_period:
            atr[atr_period - 1] = np.mean(tr[:atr_period])
            for i in range(atr_period, n):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    else:
        atr = tr
    
    # 強化チョピネス指数を計算
    for i in range(n):
        period = int(periods[i])
        if period < 2:
            period = 2
        
        if i >= period:
            # ATRスムージングされたTrue Rangeの合計
            if atr_smoothing:
                tr_sum = 0.0
                for j in range(i - period + 1, i + 1):
                    tr_sum += atr[j]
            else:
                tr_sum = 0.0
                for j in range(i - period + 1, i + 1):
                    tr_sum += tr[j]
            
            # 期間内の最高値と最安値
            period_high = high[i - period + 1]
            period_low = low[i - period + 1]
            for j in range(i - period + 2, i + 1):
                period_high = max(period_high, high[j])
                period_low = min(period_low, low[j])
            
            price_range = period_high - period_low
            
            if price_range > 1e-12 and tr_sum > 1e-12 and period > 1:
                log_period = np.log10(float(period))
                chop_value = 100.0 * np.log10(tr_sum / price_range) / log_period
                enhanced_chop[i] = max(0.0, min(100.0, chop_value))
    
    return enhanced_chop


@njit(fastmath=True, cache=True)
def calculate_volatility_factor(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    ボラティリティ調整係数を計算
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        ボラティリティ係数配列（市場状況に応じた調整係数）
    """
    n = len(prices)
    vol_factor = np.ones(n)
    
    for i in range(window, n):
        # 価格変化率を計算
        returns = np.zeros(window - 1)
        for j in range(window - 1):
            if prices[i - window + j] != 0:
                returns[j] = (prices[i - window + j + 1] - prices[i - window + j]) / prices[i - window + j]
        
        # 標準偏差を計算
        mean_return = np.mean(returns)
        variance = 0.0
        for ret in returns:
            variance += (ret - mean_return) ** 2
        variance /= len(returns)
        volatility = np.sqrt(variance)
        
        # ボラティリティに基づく調整係数
        # 低ボラティリティ時はトレンド検出を強化
        # 高ボラティリティ時はレンジ検出を強化
        if volatility > 0:
            # 正規化されたボラティリティ（0.02を基準とする）
            normalized_vol = volatility / 0.02
            vol_factor[i] = 1.0 / (1.0 + normalized_vol)
        
    return vol_factor


@njit(fastmath=True, cache=True)
def calculate_enhanced_trend_state(
    enhanced_er: np.ndarray,
    enhanced_chop: np.ndarray,
    vol_factor: np.ndarray,
    er_weight: float = 0.6,
    chop_weight: float = 0.4,
    threshold: float = 0.6
) -> tuple:
    """
    強化トレンド状態を計算
    
    Args:
        enhanced_er: 強化効率比
        enhanced_chop: 強化チョピネス指数
        vol_factor: ボラティリティ係数
        er_weight: 効率比の重み
        chop_weight: チョピネス指数の重み
        threshold: 判定閾値
    
    Returns:
        (trend_state, confidence, composite_score)
    """
    n = len(enhanced_er)
    trend_state = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n)
    composite_score = np.zeros(n)
    
    for i in range(n):
        # 効率比スコア（高いほどトレンド）
        er_score = enhanced_er[i]
        
        # チョピネススコア（低いほどトレンド）
        chop_score = 1.0 - (enhanced_chop[i] / 100.0)
        
        # ボラティリティ調整を適用
        adjusted_er = er_score * vol_factor[i]
        adjusted_chop = chop_score * vol_factor[i]
        
        # 重み付き複合スコア
        composite_score[i] = (adjusted_er * er_weight + adjusted_chop * chop_weight)
        
        # 信頼度を計算（両指標の一致度）
        score_diff = abs(adjusted_er - adjusted_chop)
        confidence[i] = 1.0 - score_diff
        
        # バイナリ判定
        trend_state[i] = 1 if composite_score[i] >= threshold else 0
    
    return trend_state, confidence, composite_score


class EnhancedTrendState(Indicator):
    """
    Enhanced Trend State Indicator
    
    既存のEfficiency RatioとChoppiness Indexをベースとした
    超高精度・超低遅延・超動的適応性のトレンド状態判別インジケーター
    
    特徴:
    - 動的期間調整による適応性
    - ボラティリティ調整による精度向上
    - ATRスムージングによる安定性
    - シンプルで洗練されたアルゴリズム
    - バイナリ出力（1=トレンド、0=レンジ）
    """
    
    def __init__(
        self,
        base_period: int = 20,
        threshold: float = 0.5,
        src_type: str = 'hlc3',
        use_dynamic_period: bool = True,
        volatility_adjustment: bool = True,
        atr_smoothing: bool = True,
        er_weight: float = 0.6,
        chop_weight: float = 0.4,
        detector_type: str = 'absolute_ultimate',
        max_cycle: int = 50,
        min_cycle: int = 8
    ):
        """
        コンストラクタ
        
        Args:
            base_period: ベース期間
            threshold: トレンド判定閾値 (0.5-0.8推奨)
            src_type: 価格ソースタイプ
            use_dynamic_period: 動的期間を使用するか
            volatility_adjustment: ボラティリティ調整を使用するか
            atr_smoothing: ATRスムージングを使用するか
            er_weight: 効率比の重み
            chop_weight: チョピネス指数の重み
            detector_type: サイクル検出器タイプ
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        vol_str = "_vol" if volatility_adjustment else ""
        atr_str = "_atr" if atr_smoothing else ""
        super().__init__(f"EnhancedTrendState(p={base_period},th={threshold:.2f}{dynamic_str}{vol_str}{atr_str})")
        
        self.base_period = base_period
        self.threshold = threshold
        self.src_type = src_type
        self.use_dynamic_period = use_dynamic_period
        self.volatility_adjustment = volatility_adjustment
        self.atr_smoothing = atr_smoothing
        self.er_weight = er_weight
        self.chop_weight = chop_weight
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        
        # 動的期間用サイクル検出器
        self.cycle_detector = None
        if self.use_dynamic_period:
            self.cycle_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                src_type=self.src_type
            )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 3
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    first_val = float(data[0, -1]) if data.ndim > 1 else float(data[0])
                    last_val = float(data[-1, -1]) if data.ndim > 1 else float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            params_sig = f"{self.base_period}_{self.threshold}_{self.use_dynamic_period}_{self.volatility_adjustment}_{self.atr_smoothing}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.base_period}_{self.threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EnhancedTrendStateResult:
        """
        Enhanced Trend Stateを計算
        
        Args:
            data: 価格データ
        
        Returns:
            EnhancedTrendStateResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # データ準備
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
                close = np.asarray(data['close'].values, dtype=np.float64)
                prices = PriceSource.calculate_source(data, self.src_type)
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1].astype(np.float64)
                low = data[:, 2].astype(np.float64)
                close = data[:, 3].astype(np.float64)
                prices = close
            
            prices = np.asarray(prices, dtype=np.float64)
            
            # 動的期間の計算
            if self.use_dynamic_period and self.cycle_detector is not None:
                dynamic_cycles = self.cycle_detector.calculate(data)
                periods = np.asarray(dynamic_cycles, dtype=np.float64)
                # 無効な値を基本期間で置換
                periods = np.where(np.isnan(periods) | (periods < self.min_cycle), 
                                 self.base_period, periods)
                periods = np.clip(periods, self.min_cycle, self.max_cycle)
            else:
                periods = np.full(len(prices), self.base_period, dtype=np.float64)
            
            # ボラティリティ係数の計算
            vol_factor = calculate_volatility_factor(prices, min(20, self.base_period))
            
            # 強化効率比の計算
            enhanced_er = calculate_enhanced_efficiency_ratio(
                prices, periods, self.volatility_adjustment
            )
            
            # 強化チョピネス指数の計算
            enhanced_chop = calculate_enhanced_choppiness(
                high, low, close, periods, self.atr_smoothing
            )
            
            # 最終判定
            trend_state, confidence, composite_score = calculate_enhanced_trend_state(
                enhanced_er, enhanced_chop, vol_factor,
                self.er_weight, self.chop_weight, self.threshold
            )
            
            # 結果を作成
            result = EnhancedTrendStateResult(
                trend_state=trend_state,
                confidence=confidence,
                efficiency_ratio=enhanced_er,
                choppiness_index=enhanced_chop,
                composite_score=composite_score,
                dynamic_periods=periods,
                volatility_factor=vol_factor
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_state
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced Trend State計算エラー: {e}")
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            return EnhancedTrendStateResult(
                trend_state=np.zeros(n, dtype=np.int32),
                confidence=np.zeros(n),
                efficiency_ratio=np.zeros(n),
                choppiness_index=np.zeros(n),
                composite_score=np.zeros(n),
                dynamic_periods=np.full(n, self.base_period),
                volatility_factor=np.ones(n)
            )
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態（バイナリ出力）を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.trend_state.copy()
    
    def get_confidence(self) -> Optional[np.ndarray]:
        """信頼度を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.confidence.copy()
    
    def get_composite_score(self) -> Optional[np.ndarray]:
        """複合スコアを取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.composite_score.copy()
    
    def is_trending(self) -> bool:
        """現在がトレンド状態かを判定"""
        trend_state = self.get_trend_state()
        if trend_state is None or len(trend_state) == 0:
            return False
        return bool(trend_state[-1])
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if self.cycle_detector is not None:
            self.cycle_detector.reset()