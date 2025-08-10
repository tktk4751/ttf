#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit
import traceback

try:
    from .indicator import Indicator
    from .str import STR
    from .price_source import PriceSource
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from str import STR
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class UltimateChoppinessResult:
    """Ultimate Choppiness Index計算結果"""
    values: np.ndarray              # チョピネス値（0-100）
    str_values: np.ndarray          # STR値
    true_range: np.ndarray          # True Range値
    range_values: np.ndarray        # 期間内の価格レンジ
    trend_state: np.ndarray         # トレンド状態（1=トレンド、0=レンジ）
    dynamic_periods: np.ndarray     # 動的期間


@njit(fastmath=True, cache=True)
def calculate_ultimate_choppiness(
    str_values: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    periods: np.ndarray,
    trend_threshold: float = 0.5,
    range_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ultimate Choppiness Indexを計算
    
    Args:
        str_values: STR（Smooth True Range）値
        high: 高値配列
        low: 安値配列
        periods: 動的期間配列
        trend_threshold: トレンド判定閾値（デフォルト: 0.5）
        range_threshold: レンジ判定閾値（デフォルト: 0.5）
    
    Returns:
        (choppiness_values, range_values, trend_state)
    """
    n = len(high)
    choppiness = np.zeros(n)
    range_values = np.zeros(n)
    trend_state = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        period = int(periods[i])
        if period < 2:
            period = 2
        
        if i >= period:
            # STRの合計（Ultimate Smootherで平滑化済み）
            str_sum = 0.0
            for j in range(i - period + 1, i + 1):
                str_sum += str_values[j]
            
            # 期間内の最高値と最安値
            period_high = high[i - period + 1]
            period_low = low[i - period + 1]
            for j in range(i - period + 2, i + 1):
                period_high = max(period_high, high[j])
                period_low = min(period_low, low[j])
            
            price_range = period_high - period_low
            range_values[i] = price_range
            
            # チョピネス計算
            if price_range > 1e-12 and str_sum > 1e-12 and period > 1:
                log_period = np.log10(float(period))
                chop_value = 100.0 * np.log10(str_sum / price_range) / log_period
                choppiness[i] = max(0.0, min(100.0, chop_value))
                
                # トレンド状態判定（0-1に正規化してから判定）
                normalized_chop = choppiness[i] / 100.0
                if normalized_chop <= trend_threshold:
                    trend_state[i] = 1  # トレンド
                else:
                    trend_state[i] = 0  # レンジ
            else:
                choppiness[i] = 50.0  # デフォルト値（中立）
                if i > 0:
                    trend_state[i] = trend_state[i-1]
    
    return choppiness, range_values, trend_state


@njit(fastmath=True, cache=True)
def smooth_choppiness_values(
    choppiness: np.ndarray,
    smooth_period: int = 3
) -> np.ndarray:
    """
    チョピネス値を平滑化（ノイズ除去）
    
    Args:
        choppiness: チョピネス値配列
        smooth_period: 平滑化期間
    
    Returns:
        平滑化されたチョピネス値
    """
    n = len(choppiness)
    smoothed = np.zeros(n)
    
    for i in range(n):
        if i < smooth_period:
            # 初期値は単純移動平均
            sum_val = 0.0
            count = 0
            for j in range(i + 1):
                sum_val += choppiness[j]
                count += 1
            if count > 0:
                smoothed[i] = sum_val / count
        else:
            # 指数移動平均（EMA）で平滑化
            alpha = 2.0 / (smooth_period + 1.0)
            smoothed[i] = alpha * choppiness[i] + (1.0 - alpha) * smoothed[i-1]
    
    return smoothed


class UltimateChoppinessIndex(Indicator):
    """
    Ultimate Choppiness Index
    
    従来のChoppiness IndexのATRをSTR（Smooth True Range）に置き換えた
    超低遅延・高精度版チョピネスインデックス
    
    特徴:
    - STRによる超低遅延計算
    - 動的期間調整対応
    - トレンド/レンジの明確な判定
    - ノイズ除去のための平滑化機能
    
    判定基準:
    - 0.5以下: トレンド相場（正規化後）
    - 0.5超: レンジ相場（正規化後）
    """
    
    def __init__(
        self,
        period: float = 14.0,
        src_type: str = 'hlc3',
        period_mode: str = 'dynamic',
        trend_threshold: float = 0.5,
        range_threshold: float = 0.5,
        smooth_period: int = 3,
        # STRパラメータ
        ukf_params: Optional[Dict] = None,
        # サイクル検出器パラメータ
        cycle_detector_type: str = 'absolute_ultimate',
        cycle_detector_cycle_part: float = 0.5,
        cycle_detector_max_cycle: int = 55,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本期間
            src_type: 価格ソースタイプ
            period_mode: 期間モード ('fixed' or 'dynamic')
            trend_threshold: トレンド判定閾値（デフォルト: 0.5）
            range_threshold: レンジ判定閾値（デフォルト: 0.5）
            smooth_period: 平滑化期間（デフォルト: 3）
            ukf_params: UKFパラメータ（STR用）
            cycle_detector_type: サイクル検出器タイプ
            cycle_detector_cycle_part: サイクル部分倍率
            cycle_detector_max_cycle: 最大サイクル期間
            cycle_detector_min_cycle: 最小サイクル期間
            cycle_period_multiplier: サイクル期間の乗数
            cycle_detector_period_range: サイクル検出器の周期範囲
        """
        indicator_name = f"UltimateChoppiness({period}, {period_mode}, smooth={smooth_period})"
        super().__init__(indicator_name)
        
        self.period = period
        self.src_type = src_type
        self.period_mode = period_mode
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.smooth_period = smooth_period
        
        # パラメータ検証
        if self.trend_threshold > 1.0 or self.trend_threshold < 0.0:
            raise ValueError("trend_thresholdは0.0から1.0の範囲である必要があります")
        
        # STRインジケーターの初期化
        self.str_indicator = STR(
            period=period,
            src_type=src_type,
            ukf_params=ukf_params,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            cycle_detector_cycle_part=cycle_detector_cycle_part,
            cycle_detector_max_cycle=cycle_detector_max_cycle,
            cycle_detector_min_cycle=cycle_detector_min_cycle,
            cycle_period_multiplier=cycle_period_multiplier,
            cycle_detector_period_range=cycle_detector_period_range
        )
        
        # 動的期間用サイクル検出器（STRと共有）
        self.cycle_detector = None
        if self.period_mode == 'dynamic':
            self.cycle_detector = EhlersUnifiedDC(
                detector_type=cycle_detector_type,
                cycle_part=cycle_detector_cycle_part,
                max_cycle=cycle_detector_max_cycle,
                min_cycle=cycle_detector_min_cycle,
                src_type=src_type,
                period_range=cycle_detector_period_range
            )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
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
            
            params_sig = f"{self.period}_{self.period_mode}_{self.trend_threshold}_{self.range_threshold}_{self.smooth_period}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.period_mode}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateChoppinessResult:
        """
        Ultimate Choppiness Indexを計算
        
        Args:
            data: 価格データ（high, low, closeが必要）
        
        Returns:
            UltimateChoppinessResult: 計算結果
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
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1].astype(np.float64)
                low = data[:, 2].astype(np.float64)
                close = data[:, 3].astype(np.float64)
            
            # STRを計算
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            true_range = str_result.true_range
            
            # 動的期間の取得
            if self.period_mode == 'dynamic' and self.cycle_detector is not None:
                dynamic_cycles = self.cycle_detector.calculate(data)
                periods = np.asarray(dynamic_cycles, dtype=np.float64)
                # 無効な値を基本期間で置換
                periods = np.where(np.isnan(periods) | (periods < self.str_indicator.cycle_detector_min_cycle), 
                                 self.period, periods)
                periods = np.clip(periods, self.str_indicator.cycle_detector_min_cycle, 
                                self.str_indicator.cycle_detector_max_cycle)
            else:
                periods = np.full(len(close), self.period, dtype=np.float64)
            
            # Ultimate Choppiness計算
            choppiness, range_values, trend_state = calculate_ultimate_choppiness(
                str_values, high, low, periods,
                self.trend_threshold, self.range_threshold
            )
            
            # 平滑化（オプション）
            if self.smooth_period > 1:
                choppiness = smooth_choppiness_values(choppiness, self.smooth_period)
            
            # 結果作成
            result = UltimateChoppinessResult(
                values=choppiness,
                str_values=str_values,
                true_range=true_range,
                range_values=range_values,
                trend_state=trend_state,
                dynamic_periods=periods
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = choppiness
            
            self.logger.debug(f"Ultimate Choppiness計算完了 - 平均値: {np.mean(choppiness[~np.isnan(choppiness)]):.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ultimate Choppiness計算エラー: {e}")
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            return UltimateChoppinessResult(
                values=np.zeros(n),
                str_values=np.zeros(n),
                true_range=np.zeros(n),
                range_values=np.zeros(n),
                trend_state=np.zeros(n, dtype=np.int32),
                dynamic_periods=np.full(n, self.period)
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """チョピネス値を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.values.copy()
    
    def get_str_values(self) -> Optional[np.ndarray]:
        """STR値を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.str_values.copy()
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.trend_state.copy()
    
    def is_trending(self) -> bool:
        """現在がトレンド状態かを判定"""
        trend_state = self.get_trend_state()
        if trend_state is None or len(trend_state) == 0:
            return False
        return bool(trend_state[-1] == 1)
    
    def is_ranging(self) -> bool:
        """現在がレンジ状態かを判定"""
        return not self.is_trending()
    
    def get_current_value(self) -> float:
        """現在のチョピネス値を取得"""
        values = self.get_values()
        if values is None or len(values) == 0:
            return 50.0  # デフォルト値（中立）
        return float(values[-1])
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self.str_indicator.reset()
        if self.cycle_detector is not None:
            self.cycle_detector.reset()