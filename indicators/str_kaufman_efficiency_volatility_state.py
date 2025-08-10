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
class STRKaufmanEfficiencyVolatilityStateResult:
    """STR + Kaufman効率比 ボラティリティ状態判別結果"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    str_values: np.ndarray                # STR値
    kaufman_efficiency: np.ndarray        # カウフマン効率比
    directional_movement: np.ndarray      # 方向性動き（分子）
    volatility_movement: np.ndarray       # ボラティリティ動き（分母）
    efficiency_trend: np.ndarray          # 効率比のトレンド
    signal_strength: np.ndarray           # シグナル強度


@njit(fastmath=True, cache=True)
def calculate_kaufman_efficiency_ratio(str_values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STRベースのカウフマン効率比計算
    
    効率比 = |方向性変化| / Σ|期間内変化|
    
    Args:
        str_values: STR値配列
        period: 計算期間
        
    Returns:
        efficiency_ratio: 効率比 (0-1)
        directional_movement: 方向性動き（分子）
        volatility_movement: ボラティリティ動き（分母）
    """
    length = len(str_values)
    efficiency_ratio = np.zeros(length)
    directional_movement = np.zeros(length)
    volatility_movement = np.zeros(length)
    
    for i in range(period, length):
        # 方向性動き（期間の始点から終点への変化）
        direction = abs(str_values[i] - str_values[i - period])
        directional_movement[i] = direction
        
        # ボラティリティ動き（期間内の累積変化）
        volatility = 0.0
        for j in range(period):
            if i - j > 0:
                volatility += abs(str_values[i - j] - str_values[i - j - 1])
        volatility_movement[i] = volatility
        
        # 効率比計算
        if volatility > 1e-8:  # ゼロ除算回避
            efficiency_ratio[i] = direction / volatility
        else:
            efficiency_ratio[i] = 0.0
        
        # 効率比は0-1範囲にクリップ
        if efficiency_ratio[i] > 1.0:
            efficiency_ratio[i] = 1.0
        elif efficiency_ratio[i] < 0.0:
            efficiency_ratio[i] = 0.0
    
    return efficiency_ratio, directional_movement, volatility_movement


@njit(fastmath=True, cache=True)
def calculate_efficiency_trend(efficiency_ratio: np.ndarray, trend_period: int = 5) -> np.ndarray:
    """
    効率比のトレンド計算（移動平均による平滑化）
    """
    length = len(efficiency_ratio)
    efficiency_trend = np.zeros(length)
    
    for i in range(trend_period, length):
        # 過去trend_period期間の効率比平均
        sum_efficiency = 0.0
        for j in range(trend_period):
            sum_efficiency += efficiency_ratio[i - j]
        efficiency_trend[i] = sum_efficiency / trend_period
    
    return efficiency_trend


@njit(fastmath=True, cache=True)
def calculate_signal_strength(efficiency_ratio: np.ndarray, str_values: np.ndarray, period: int = 10) -> np.ndarray:
    """
    シグナル強度計算（効率比とSTR変動の組み合わせ）
    """
    length = len(efficiency_ratio)
    signal_strength = np.zeros(length)
    
    for i in range(period, length):
        # 効率比の安定性（標準偏差の逆数）
        sum_eff = 0.0
        for j in range(period):
            sum_eff += efficiency_ratio[i - j]
        mean_eff = sum_eff / period
        
        variance_eff = 0.0
        for j in range(period):
            diff = efficiency_ratio[i - j] - mean_eff
            variance_eff += diff * diff
        variance_eff /= period
        std_eff = np.sqrt(variance_eff)
        
        # STRレベルの正規化
        current_str = str_values[i]
        max_str = 0.0
        for j in range(period):
            if str_values[i - j] > max_str:
                max_str = str_values[i - j]
        
        str_level = current_str / max_str if max_str > 0 else 0
        
        # シグナル強度 = 効率比 * STRレベル * 安定性
        stability = 1.0 / (1.0 + std_eff * 10)  # 安定性（標準偏差が小さいほど高い）
        signal_strength[i] = efficiency_ratio[i] * str_level * stability
        
        # 0-1範囲にクリップ
        if signal_strength[i] > 1.0:
            signal_strength[i] = 1.0
        elif signal_strength[i] < 0.0:
            signal_strength[i] = 0.0
    
    return signal_strength


@njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(efficiency_ratio: np.ndarray, lookback_period: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応的閾値計算（効率比の動的閾値）
    """
    length = len(efficiency_ratio)
    upper_threshold = np.zeros(length)
    lower_threshold = np.zeros(length)
    
    for i in range(lookback_period, length):
        # 過去期間の効率比統計
        historical_ratios = efficiency_ratio[i-lookback_period:i]
        
        # ソートして統計量を計算
        sorted_ratios = np.sort(historical_ratios)
        median_idx = len(sorted_ratios) // 2
        q1_idx = len(sorted_ratios) // 4
        q3_idx = 3 * len(sorted_ratios) // 4
        
        if q3_idx < len(sorted_ratios) and median_idx >= 0:
            median = sorted_ratios[median_idx]
            q1 = sorted_ratios[q1_idx]
            q3 = sorted_ratios[q3_idx]
            
            # 適応的閾値（中央値ベース）
            iqr = q3 - q1
            upper_threshold[i] = median + 0.3 * iqr  # より保守的
            lower_threshold[i] = median - 0.3 * iqr
            
            # 最低限の閾値設定
            if upper_threshold[i] < 0.6:
                upper_threshold[i] = 0.6
            if lower_threshold[i] > 0.4:
                lower_threshold[i] = 0.4
        else:
            upper_threshold[i] = 0.6  # デフォルト上限
            lower_threshold[i] = 0.4  # デフォルト下限
    
    return upper_threshold, lower_threshold


@njit(fastmath=True, parallel=True, cache=True)
def str_kaufman_efficiency_volatility_fusion(
    efficiency_ratio: np.ndarray,
    efficiency_trend: np.ndarray,
    signal_strength: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray,
    efficiency_weight: float = 0.6,
    trend_weight: float = 0.25,
    strength_weight: float = 0.15,
    base_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STR + カウフマン効率比 ボラティリティ融合アルゴリズム
    """
    length = len(efficiency_ratio)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 基本効率比スコア
        efficiency_score = efficiency_ratio[i] if not np.isnan(efficiency_ratio[i]) else 0.5
        
        # トレンドスコア（平滑化効率比）
        trend_score = efficiency_trend[i] if not np.isnan(efficiency_trend[i]) else 0.5
        
        # シグナル強度スコア
        strength_score = signal_strength[i] if not np.isnan(signal_strength[i]) else 0.5
        
        # 重み付き融合
        score = (efficiency_weight * efficiency_score + 
                 trend_weight * trend_score + 
                 strength_weight * strength_score)
        
        raw_score[i] = score
        
        # 確率計算（シグモイド変換）
        k = 5.0  # 急峻さパラメータ
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - 0.5)))
        
        # 適応的閾値による状態判定
        if i < len(upper_threshold) and upper_threshold[i] > 0:
            high_thresh = upper_threshold[i]
            low_thresh = lower_threshold[i]
        else:
            # フォールバック閾値
            high_thresh = 0.6
            low_thresh = 0.4
        
        # ヒステリシス付き状態判定
        if i > 0:
            prev_state = state[i-1]
            
            if prev_state == 0:  # 前回が低ボラティリティ
                # 基本閾値 + 適応的調整
                effective_threshold = max(base_threshold, high_thresh)
                state[i] = 1 if score > effective_threshold else 0
            else:  # 前回が高ボラティリティ
                # 基本閾値 + 適応的調整
                effective_threshold = min(base_threshold, low_thresh)
                state[i] = 0 if score < effective_threshold else 1
        else:
            # 初回判定（基本閾値使用）
            state[i] = 1 if score > base_threshold else 0
    
    return state, probability, raw_score


class STRKaufmanEfficiencyVolatilityState(Indicator):
    """
    STR + カウフマン効率比 ボラティリティ状態判別インジケーター
    
    カウフマン効率比ベースのアプローチ:
    1. STRをソースとしてカウフマン効率比を計算
    2. 効率比 > 0.5 なら高ボラティリティ（トレンド性が強い）
    3. 効率比 <= 0.5 なら低ボラティリティ（ノイズが多い）
    4. 効率比トレンドで平滑化
    5. シグナル強度で信頼性評価
    
    カウフマン効率比の特徴:
    - 0に近い: ノイズが多い（横ばい、高ボラティリティ）
    - 1に近い: トレンドが強い（方向性がある、低ボラティリティ）
    
    注意: この実装では効率比の解釈を逆転
    - 高効率比 = 強いトレンド = 高ボラティリティ状態として扱う
    """
    
    def __init__(
        self,
        str_period: int = 14,                 # STR計算期間
        efficiency_period: int = 10,          # 効率比計算期間
        trend_period: int = 5,                # トレンド計算期間
        lookback_period: int = 100,           # 適応的閾値計算期間
        efficiency_weight: float = 0.6,       # 効率比重み
        trend_weight: float = 0.25,           # トレンド重み
        strength_weight: float = 0.15,        # 強度重み
        base_threshold: float = 0.5,          # 基本閾値
        src_type: str = 'hlc3',               # 価格ソース
        smoothing: bool = True                # スムージングの有効化
    ):
        """
        コンストラクタ
        
        Args:
            str_period: STR計算期間
            efficiency_period: カウフマン効率比計算期間
            trend_period: 効率比トレンド計算期間
            lookback_period: 適応的閾値計算の振り返り期間
            efficiency_weight: 効率比の重み
            trend_weight: トレンドの重み
            strength_weight: シグナル強度の重み
            base_threshold: 基本判定閾値（0.5推奨）
            src_type: 価格ソースタイプ
            smoothing: 最終結果のスムージング
        """
        super().__init__(f"STRKaufmanEfficiencyVolatilityState(str={str_period}, eff={efficiency_period}, threshold={base_threshold})")
        
        self.str_period = str_period
        self.efficiency_period = efficiency_period
        self.trend_period = trend_period
        self.lookback_period = lookback_period
        self.efficiency_weight = efficiency_weight
        self.trend_weight = trend_weight
        self.strength_weight = strength_weight
        self.base_threshold = base_threshold
        self.src_type = src_type.lower()
        self.smoothing = smoothing
        
        # STRインジケーター
        self.str_indicator = STR(
            period=str_period,
            src_type=src_type,
            period_mode='fixed'
        )
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=3, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> STRKaufmanEfficiencyVolatilityStateResult:
        """
        STR + カウフマン効率比 ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            STRKaufmanEfficiencyVolatilityStateResult: 判定結果
        """
        try:
            # データ準備
            length = len(data)
            min_required = max(self.str_period, self.efficiency_period, self.lookback_period // 4)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. STR計算
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            # 2. カウフマン効率比計算
            kaufman_efficiency, directional_movement, volatility_movement = calculate_kaufman_efficiency_ratio(
                str_values, self.efficiency_period
            )
            
            # 3. 効率比トレンド計算
            efficiency_trend = calculate_efficiency_trend(kaufman_efficiency, self.trend_period)
            
            # 4. シグナル強度計算
            signal_strength = calculate_signal_strength(kaufman_efficiency, str_values, self.trend_period)
            
            # 5. 適応的閾値計算
            upper_threshold, lower_threshold = calculate_adaptive_threshold(kaufman_efficiency, self.lookback_period)
            
            # 6. STR + カウフマン効率比融合
            state, probability, raw_score = str_kaufman_efficiency_volatility_fusion(
                kaufman_efficiency, efficiency_trend, signal_strength,
                upper_threshold, lower_threshold,
                self.efficiency_weight, self.trend_weight, self.strength_weight,
                self.base_threshold
            )
            
            # 7. オプショナルスムージング
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
            result = STRKaufmanEfficiencyVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                str_values=str_values,
                kaufman_efficiency=kaufman_efficiency,
                directional_movement=directional_movement,
                volatility_movement=volatility_movement,
                efficiency_trend=efficiency_trend,
                signal_strength=signal_strength
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
            self.logger.error(f"STR + カウフマン効率比ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> STRKaufmanEfficiencyVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return STRKaufmanEfficiencyVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            raw_score=empty_array,
            str_values=empty_array,
            kaufman_efficiency=empty_array,
            directional_movement=empty_array,
            volatility_movement=empty_array,
            efficiency_trend=empty_array,
            signal_strength=empty_array
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
            
            params_sig = f"{self.str_period}_{self.efficiency_period}_{self.base_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.str_period}_{self.efficiency_period}"
    
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
                'str_values': latest_result.str_values,
                'kaufman_efficiency': latest_result.kaufman_efficiency,
                'directional_movement': latest_result.directional_movement,
                'volatility_movement': latest_result.volatility_movement,
                'efficiency_trend': latest_result.efficiency_trend,
                'signal_strength': latest_result.signal_strength
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
    
    def get_current_efficiency_ratio(self) -> Optional[float]:
        """現在のカウフマン効率比を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.kaufman_efficiency) > 0:
                return float(latest_result.kaufman_efficiency[-1])
        return None
    
    def get_current_efficiency_trend(self) -> Optional[float]:
        """現在の効率比トレンドを取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.efficiency_trend) > 0:
                return float(latest_result.efficiency_trend[-1])
        return None
    
    def get_current_signal_strength(self) -> Optional[float]:
        """現在のシグナル強度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.signal_strength) > 0:
                return float(latest_result.signal_strength[-1])
        return None
    
    def get_efficiency_statistics(self) -> Optional[Dict[str, float]]:
        """効率比統計を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            efficiency = latest_result.kaufman_efficiency
            valid_efficiency = efficiency[efficiency > 0]
            
            if len(valid_efficiency) > 0:
                return {
                    'mean': float(np.mean(valid_efficiency)),
                    'std': float(np.std(valid_efficiency)),
                    'min': float(np.min(valid_efficiency)),
                    'max': float(np.max(valid_efficiency)),
                    'above_threshold': float(np.sum(valid_efficiency > self.base_threshold) / len(valid_efficiency) * 100)
                }
        return None
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        if self.smoothing:
            self.smoother.reset()