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
from .str import STR
from .smoother.ultimate_smoother import UltimateSmoother


@dataclass
class DualSTRVolatilityStateResult:
    """Dual STR ボラティリティ状態判別結果"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    short_str: np.ndarray                 # 短期STR値（20期間）
    long_str: np.ndarray                  # 長期STR値（100期間）
    str_ratio: np.ndarray                 # STR比率（短期/長期）
    str_difference: np.ndarray            # STR差分（短期-長期）
    trend_strength: np.ndarray            # トレンド強度


@njit(fastmath=True, cache=True)
def calculate_str_ratio_and_difference(short_str: np.ndarray, long_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    STR比率と差分を計算
    """
    length = len(short_str)
    str_ratio = np.zeros(length)
    str_difference = np.zeros(length)
    
    for i in range(length):
        if long_str[i] > 1e-8:  # ゼロ除算回避
            str_ratio[i] = short_str[i] / long_str[i]
        else:
            str_ratio[i] = 1.0
        
        str_difference[i] = short_str[i] - long_str[i]
    
    return str_ratio, str_difference


@njit(fastmath=True, cache=True)
def calculate_trend_strength(str_ratio: np.ndarray, period: int = 10) -> np.ndarray:
    """
    トレンド強度を計算（STR比率の移動平均から安定性を測定）
    """
    length = len(str_ratio)
    trend_strength = np.zeros(length)
    
    for i in range(period, length):
        # 過去period期間のSTR比率の標準偏差（安定性の逆指標）
        sum_ratio = 0.0
        for j in range(period):
            sum_ratio += str_ratio[i-j]
        mean_ratio = sum_ratio / period
        
        # 分散計算
        variance = 0.0
        for j in range(period):
            diff = str_ratio[i-j] - mean_ratio
            variance += diff * diff
        variance /= period
        
        # トレンド強度 = 1 / (1 + 標準偏差)
        # 標準偏差が小さい（安定）ほど、トレンド強度が高い
        std_dev = np.sqrt(variance)
        trend_strength[i] = 1.0 / (1.0 + std_dev)
    
    return trend_strength


@njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(str_ratio: np.ndarray, lookback_period: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応的閾値計算（市場状況に応じて動的に調整）
    """
    length = len(str_ratio)
    upper_threshold = np.zeros(length)
    lower_threshold = np.zeros(length)
    
    for i in range(lookback_period, length):
        # 過去期間のSTR比率統計
        historical_ratios = str_ratio[i-lookback_period:i]
        
        # ソートして四分位点を計算
        sorted_ratios = np.sort(historical_ratios)
        q1_idx = len(sorted_ratios) // 4
        q3_idx = 3 * len(sorted_ratios) // 4
        median_idx = len(sorted_ratios) // 2
        
        if q3_idx < len(sorted_ratios) and q1_idx >= 0:
            q1 = sorted_ratios[q1_idx]
            q3 = sorted_ratios[q3_idx]
            median = sorted_ratios[median_idx]
            iqr = q3 - q1
            
            # 適応的閾値（IQRベース）
            upper_threshold[i] = median + 0.5 * iqr
            lower_threshold[i] = median - 0.5 * iqr
        else:
            upper_threshold[i] = 1.1  # デフォルト上限
            lower_threshold[i] = 0.9  # デフォルト下限
    
    return upper_threshold, lower_threshold


@njit(fastmath=True, parallel=True, cache=True)
def dual_str_volatility_fusion(
    short_str: np.ndarray,
    long_str: np.ndarray,
    str_ratio: np.ndarray,
    str_difference: np.ndarray,
    trend_strength: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray,
    ratio_weight: float = 0.6,
    difference_weight: float = 0.25,
    trend_weight: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dual STR ボラティリティ融合アルゴリズム
    """
    length = len(short_str)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 基本判定: 短期STR > 長期STR なら高ボラティリティ
        basic_signal = 1.0 if str_ratio[i] > 1.0 else 0.0
        
        # 適応的閾値による詳細判定
        if i < len(upper_threshold) and upper_threshold[i] > 0:
            if str_ratio[i] > upper_threshold[i]:
                ratio_score = 1.0
            elif str_ratio[i] < lower_threshold[i]:
                ratio_score = 0.0
            else:
                # 閾値内の場合は線形補間
                range_size = upper_threshold[i] - lower_threshold[i]
                if range_size > 0:
                    ratio_score = (str_ratio[i] - lower_threshold[i]) / range_size
                else:
                    ratio_score = basic_signal
        else:
            ratio_score = basic_signal
        
        # 差分スコア（正規化）
        if not np.isnan(str_difference[i]):
            # 差分を[-1, 1]範囲に正規化（tanh使用）
            normalized_diff = np.tanh(str_difference[i] * 10)  # 10は感度調整
            difference_score = (normalized_diff + 1.0) / 2.0  # [0, 1]範囲に変換
        else:
            difference_score = 0.5
        
        # トレンド強度スコア
        trend_score = trend_strength[i] if not np.isnan(trend_strength[i]) else 0.5
        
        # 重み付き融合
        score = (ratio_weight * ratio_score + 
                 difference_weight * difference_score + 
                 trend_weight * trend_score)
        
        raw_score[i] = score
        
        # 確率計算（シグモイド変換でスムーズに）
        k = 4.0  # 急峻さパラメータ
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - 0.5)))
        
        # 状態判定（ヒステリシス付き）
        if i > 0:
            prev_state = state[i-1]
            
            # ヒステリシス閾値
            high_threshold = 0.65
            low_threshold = 0.35
            
            if prev_state == 0:  # 前回が低ボラティリティ
                state[i] = 1 if score > high_threshold else 0
            else:  # 前回が高ボラティリティ
                state[i] = 0 if score < low_threshold else 1
        else:
            # 初回判定
            state[i] = 1 if score > 0.5 else 0
    
    return state, probability, raw_score


class DualSTRVolatilityState(Indicator):
    """
    Dual STR ボラティリティ状態判別インジケーター
    
    STRベースのシンプルで効果的なアプローチ:
    1. 短期STR（20期間）- 短期的なボラティリティ変化を捉える
    2. 長期STR（100期間）- 長期的なボラティリティベースラインを捉える
    3. STR比率判定 - 短期STR > 長期STR なら高ボラティリティ
    4. 適応的閾値 - 市場状況に応じて動的に調整
    
    特徴:
    - 超低遅延（STRベース）
    - シンプルで直感的なロジック
    - 適応的閾値で市場変化に対応
    - トレンド強度で安定性を評価
    - ノイズに対する頑健性
    """
    
    def __init__(
        self,
        short_period: int = 20,               # 短期STR期間
        long_period: int = 100,               # 長期STR期間
        lookback_period: int = 50,            # 適応的閾値計算期間
        trend_period: int = 10,               # トレンド強度計算期間
        ratio_weight: float = 0.6,            # STR比率重み
        difference_weight: float = 0.25,      # STR差分重み
        trend_weight: float = 0.15,           # トレンド強度重み
        src_type: str = 'hlc3',               # 価格ソース
        smoothing: bool = True                # スムージングの有効化
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期STR計算期間
            long_period: 長期STR計算期間
            lookback_period: 適応的閾値計算の振り返り期間
            trend_period: トレンド強度計算期間
            ratio_weight: STR比率の重み
            difference_weight: STR差分の重み
            trend_weight: トレンド強度の重み
            src_type: 価格ソースタイプ
            smoothing: 最終結果のスムージング
        """
        super().__init__(f"DualSTRVolatilityState(short={short_period}, long={long_period}, weights={ratio_weight:.1f}-{difference_weight:.1f}-{trend_weight:.1f})")
        
        self.short_period = short_period
        self.long_period = long_period
        self.lookback_period = lookback_period
        self.trend_period = trend_period
        self.ratio_weight = ratio_weight
        self.difference_weight = difference_weight
        self.trend_weight = trend_weight
        self.src_type = src_type.lower()
        self.smoothing = smoothing
        
        # STRインジケーター
        self.short_str = STR(
            period=short_period,
            src_type=src_type,
            period_mode='fixed'
        )
        
        self.long_str = STR(
            period=long_period,
            src_type=src_type,
            period_mode='fixed'
        )
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=3, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> DualSTRVolatilityStateResult:
        """
        Dual STR ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            DualSTRVolatilityStateResult: 判定結果
        """
        try:
            # データ準備
            length = len(data)
            min_required = max(self.short_period, self.long_period, self.lookback_period)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. 短期・長期STR計算
            short_str_result = self.short_str.calculate(data)
            long_str_result = self.long_str.calculate(data)
            
            short_str_values = short_str_result.values
            long_str_values = long_str_result.values
            
            # 2. STR比率と差分計算
            str_ratio, str_difference = calculate_str_ratio_and_difference(short_str_values, long_str_values)
            
            # 3. トレンド強度計算
            trend_strength = calculate_trend_strength(str_ratio, self.trend_period)
            
            # 4. 適応的閾値計算
            upper_threshold, lower_threshold = calculate_adaptive_threshold(str_ratio, self.lookback_period)
            
            # 5. Dual STR融合
            state, probability, raw_score = dual_str_volatility_fusion(
                short_str_values, long_str_values, str_ratio, str_difference, 
                trend_strength, upper_threshold, lower_threshold,
                self.ratio_weight, self.difference_weight, self.trend_weight
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
            result = DualSTRVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                short_str=short_str_values,
                long_str=long_str_values,
                str_ratio=str_ratio,
                str_difference=str_difference,
                trend_strength=trend_strength
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
            self.logger.error(f"Dual STR ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> DualSTRVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return DualSTRVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            raw_score=empty_array,
            short_str=empty_array,
            long_str=empty_array,
            str_ratio=empty_array,
            str_difference=empty_array,
            trend_strength=empty_array
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
            
            params_sig = f"{self.short_period}_{self.long_period}_{self.ratio_weight}_{self.difference_weight}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.short_period}_{self.long_period}"
    
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
                'short_str': latest_result.short_str,
                'long_str': latest_result.long_str,
                'str_ratio': latest_result.str_ratio,
                'str_difference': latest_result.str_difference,
                'trend_strength': latest_result.trend_strength
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
    
    def get_current_str_ratio(self) -> Optional[float]:
        """現在のSTR比率を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.str_ratio) > 0:
                return float(latest_result.str_ratio[-1])
        return None
    
    def get_current_str_difference(self) -> Optional[float]:
        """現在のSTR差分を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.str_difference) > 0:
                return float(latest_result.str_difference[-1])
        return None
    
    def get_current_trend_strength(self) -> Optional[float]:
        """現在のトレンド強度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.trend_strength) > 0:
                return float(latest_result.trend_strength[-1])
        return None
    
    def get_current_short_str(self) -> Optional[float]:
        """現在の短期STR値を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.short_str) > 0:
                return float(latest_result.short_str[-1])
        return None
    
    def get_current_long_str(self) -> Optional[float]:
        """現在の長期STR値を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.long_str) > 0:
                return float(latest_result.long_str[-1])
        return None
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.short_str.reset()
        self.long_str.reset()
        if self.smoothing:
            self.smoother.reset()