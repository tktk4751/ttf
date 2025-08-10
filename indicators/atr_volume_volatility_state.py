#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.ultimate_smoother import UltimateSmoother


@dataclass
class ATRVolumeVolatilityStateResult:
    """ATR + Volume ボラティリティ状態判別結果"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    atr_values: np.ndarray                # ATR値
    atr_percentile: np.ndarray            # ATRパーセンタイル
    volume_ratio: np.ndarray              # ボリューム比率
    volume_percentile: np.ndarray         # ボリュームパーセンタイル
    combined_signal: np.ndarray           # ATR + Volume 複合シグナル


@njit(fastmath=True, cache=True)
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range (ATR) 計算
    """
    length = len(close)
    atr_values = np.zeros(length)
    tr_values = np.zeros(length)
    
    # True Range計算
    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr_values[i] = max(tr1, max(tr2, tr3))
    
    # ATR計算（指数移動平均）
    if length > period:
        # 初期値は最初のperiod期間の平均TR
        atr_values[period-1] = np.mean(tr_values[1:period])
        
        # 指数移動平均でATRを計算
        alpha = 2.0 / (period + 1)
        for i in range(period, length):
            atr_values[i] = alpha * tr_values[i] + (1 - alpha) * atr_values[i-1]
    
    return atr_values


@njit(fastmath=True, cache=True)
def calculate_volume_ratio(volume: np.ndarray, period: int) -> np.ndarray:
    """
    ボリューム比率計算（現在のボリューム / 過去平均ボリューム）
    """
    length = len(volume)
    volume_ratio = np.zeros(length)
    
    for i in range(period, length):
        # 過去period期間の平均ボリューム
        avg_volume = 0.0
        for j in range(period):
            avg_volume += volume[i-j-1]
        avg_volume /= period
        
        # 現在のボリューム比率
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / avg_volume
        else:
            volume_ratio[i] = 1.0
    
    return volume_ratio


@njit(fastmath=True, cache=True)
def calculate_percentile_rank(values: np.ndarray, lookback_period: int) -> np.ndarray:
    """
    パーセンタイルランク計算
    """
    length = len(values)
    percentiles = np.zeros(length)
    
    for i in range(lookback_period, length):
        # 過去の値を取得
        historical_values = values[i-lookback_period:i]
        
        # ソート
        sorted_values = np.sort(historical_values)
        
        current_value = values[i]
        
        # パーセンタイル計算
        count_below = 0
        for val in sorted_values:
            if val <= current_value:
                count_below += 1
        
        percentiles[i] = count_below / len(sorted_values)
    
    return percentiles


@njit(fastmath=True, cache=True)
def calculate_combined_signal(atr_values: np.ndarray, volume_ratio: np.ndarray, period: int) -> np.ndarray:
    """
    ATR + Volume 複合シグナル計算
    """
    length = len(atr_values)
    combined_signal = np.zeros(length)
    
    for i in range(period, length):
        # ATRの正規化（過去期間の最大値で正規化）
        max_atr = 0.0
        for j in range(period):
            if atr_values[i-j] > max_atr:
                max_atr = atr_values[i-j]
        
        atr_normalized = atr_values[i] / max_atr if max_atr > 0 else 0
        
        # ボリューム比率の調整（上限制限）
        vol_adjusted = min(volume_ratio[i], 3.0) / 3.0  # 3倍を上限として正規化
        
        # 複合シグナル（ATRとボリュームの幾何平均）
        if atr_normalized > 0 and vol_adjusted > 0:
            combined_signal[i] = np.sqrt(atr_normalized * vol_adjusted)
        else:
            combined_signal[i] = 0.0
    
    return combined_signal


@njit(fastmath=True, parallel=True, cache=True)
def atr_volume_volatility_fusion(
    atr_percentile: np.ndarray,
    volume_percentile: np.ndarray,
    combined_signal: np.ndarray,
    atr_weight: float = 0.4,
    volume_weight: float = 0.3,
    combined_weight: float = 0.3,
    high_vol_threshold: float = 0.7,
    low_vol_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ATR + Volume ボラティリティ融合アルゴリズム
    """
    length = len(atr_percentile)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 各指標のスコア
        atr_score = atr_percentile[i] if not np.isnan(atr_percentile[i]) else 0.5
        volume_score = volume_percentile[i] if not np.isnan(volume_percentile[i]) else 0.5
        
        # 複合シグナルスコア（0-1範囲に正規化）
        combined_score = min(combined_signal[i], 1.0) if not np.isnan(combined_signal[i]) else 0.5
        
        # 重み付き融合
        score = (atr_weight * atr_score + 
                 volume_weight * volume_score + 
                 combined_weight * combined_score)
        
        raw_score[i] = score
        
        # 確率計算（シグモイド変換）
        k = 6.0  # 急峻さパラメータ
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - 0.5)))
        
        # ヒステリシス判定
        if i > 0:
            prev_state = state[i-1]
            
            if prev_state == 0:  # 前回が低ボラティリティ
                state[i] = 1 if score > high_vol_threshold else 0
            else:  # 前回が高ボラティリティ
                state[i] = 0 if score < low_vol_threshold else 1
        else:
            # 初回判定
            state[i] = 1 if score > (high_vol_threshold + low_vol_threshold) / 2 else 0
    
    return state, probability, raw_score


class ATRVolumeVolatilityState(Indicator):
    """
    ATR + Volume ボラティリティ状態判別インジケーター
    
    技術指標ベースのアプローチ:
    1. ATR（Average True Range）- 価格ボラティリティ測定
    2. ボリューム比率 - 取引活動の活発度測定
    3. ATR + Volume 複合シグナル - 価格と出来高の相乗効果
    
    特徴:
    - シンプルで理解しやすい
    - 実績ある技術指標の組み合わせ
    - 価格と出来高の両面からボラティリティを評価
    - リアルタイム判定に適している
    """
    
    def __init__(
        self,
        atr_period: int = 14,                 # ATR計算期間
        volume_period: int = 20,              # ボリューム比率計算期間
        percentile_lookback: int = 100,       # パーセンタイル計算期間
        atr_weight: float = 0.4,              # ATR重み
        volume_weight: float = 0.3,           # ボリューム重み
        combined_weight: float = 0.3,         # 複合シグナル重み
        high_vol_threshold: float = 0.7,      # 高ボラティリティ閾値
        low_vol_threshold: float = 0.3,       # 低ボラティリティ閾値
        smoothing: bool = True                # スムージングの有効化
    ):
        """
        コンストラクタ
        
        Args:
            atr_period: ATR計算期間
            volume_period: ボリューム比率計算期間
            percentile_lookback: パーセンタイル計算の振り返り期間
            atr_weight: ATRの重み
            volume_weight: ボリュームの重み
            combined_weight: 複合シグナルの重み
            high_vol_threshold: 高ボラティリティ判定閾値
            low_vol_threshold: 低ボラティリティ判定閾値
            smoothing: 最終結果のスムージング
        """
        super().__init__(f"ATRVolumeVolatilityState(atr={atr_period}, vol={volume_period}, weights={atr_weight:.1f}-{volume_weight:.1f}-{combined_weight:.1f})")
        
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.percentile_lookback = percentile_lookback
        self.atr_weight = atr_weight
        self.volume_weight = volume_weight
        self.combined_weight = combined_weight
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.smoothing = smoothing
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=3, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ATRVolumeVolatilityStateResult:
        """
        ATR + Volume ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLCV必須)
            
        Returns:
            ATRVolumeVolatilityStateResult: 判定結果
        """
        try:
            # データ準備
            if isinstance(data, pd.DataFrame):
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
                volume = data['volume'].to_numpy() if 'volume' in data.columns else np.ones(len(data))
            else:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
                volume = data[:, 4] if data.shape[1] > 4 else np.ones(len(data))
            
            length = len(close)
            min_required = max(self.atr_period, self.volume_period, self.percentile_lookback // 4)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. ATR計算
            atr_values = calculate_atr(high, low, close, self.atr_period)
            
            # 2. ボリューム比率計算
            volume_ratio = calculate_volume_ratio(volume, self.volume_period)
            
            # 3. ATR + Volume 複合シグナル
            combined_signal = calculate_combined_signal(atr_values, volume_ratio, max(self.atr_period, self.volume_period))
            
            # 4. パーセンタイル計算
            atr_percentile = calculate_percentile_rank(atr_values, self.percentile_lookback)
            volume_percentile = calculate_percentile_rank(volume_ratio, self.percentile_lookback)
            
            # 5. ATR + Volume 融合
            state, probability, raw_score = atr_volume_volatility_fusion(
                atr_percentile, volume_percentile, combined_signal,
                self.atr_weight, self.volume_weight, self.combined_weight,
                self.high_vol_threshold, self.low_vol_threshold
            )
            
            # 6. オプショナルスムージング
            if self.smoothing:
                # 状態のスムージング
                state_df = pd.DataFrame({'close': state.astype(np.float64)})
                smoothed_state_result = self.smoother.calculate(state_df)
                smoothed_state = (smoothed_state_result.values > 0.5).astype(np.int8)
                
                # 確率のスムージング
                prob_df = pd.DataFrame({'close': probability})
                smoothed_prob_result = self.smoother.calculate(prob_df)
                smoothed_probability = smoothed_prob_result.values
            else:
                smoothed_state = state
                smoothed_probability = probability
            
            # 結果作成
            result = ATRVolumeVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                atr_values=atr_values,
                atr_percentile=atr_percentile,
                volume_ratio=volume_ratio,
                volume_percentile=volume_percentile,
                combined_signal=combined_signal
            )
            
            # キャッシュ管理
            data_hash = self._get_data_hash(data)
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"ATR + Volume ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> ATRVolumeVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return ATRVolumeVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            raw_score=empty_array,
            atr_values=empty_array,
            atr_percentile=empty_array,
            volume_ratio=empty_array,
            volume_percentile=empty_array,
            combined_signal=empty_array
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0]['close']) if length > 0 else 0.0
                last_val = float(data.iloc[-1]['close']) if length > 0 else 0.0
            else:
                length = len(data)
                first_val = float(data[0, 3]) if length > 0 else 0.0
                last_val = float(data[-1, 3]) if length > 0 else 0.0
            
            params_sig = f"{self.atr_period}_{self.volume_period}_{self.high_vol_threshold}_{self.low_vol_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.atr_period}_{self.volume_period}"
    
    def get_state(self) -> Optional[np.ndarray]:
        """現在のボラティリティ状態を取得"""
        if self._values is not None:
            return self._values.astype(np.int8)
        return None
    
    def get_detailed_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """詳細分析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return {
                'atr_values': latest_result.atr_values,
                'atr_percentile': latest_result.atr_percentile,
                'volume_ratio': latest_result.volume_ratio,
                'volume_percentile': latest_result.volume_percentile,
                'combined_signal': latest_result.combined_signal
            }
        return None
    
    def is_high_volatility(self) -> bool:
        """現在が高ボラティリティかどうか"""
        state = self.get_state()
        if state is not None and len(state) > 0:
            return bool(state[-1] == 1)
        return False
    
    def is_low_volatility(self) -> bool:
        """現在が低ボラティリティかどうか"""
        state = self.get_state()
        if state is not None and len(state) > 0:
            return bool(state[-1] == 0)
        return False
    
    def get_current_atr_ratio(self) -> Optional[float]:
        """現在のATR比率を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.atr_percentile) > 0:
                return float(latest_result.atr_percentile[-1])
        return None
    
    def get_current_volume_ratio(self) -> Optional[float]:
        """現在のボリューム比率を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.volume_ratio) > 0:
                return float(latest_result.volume_ratio[-1])
        return None
    
    def get_current_combined_signal(self) -> Optional[float]:
        """現在の複合シグナルを取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.combined_signal) > 0:
                return float(latest_result.combined_signal[-1])
        return None
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        if self.smoothing:
            self.smoother.reset()