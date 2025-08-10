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
class PracticalVolatilityStateResult:
    """実践的ボラティリティ状態判別結果 - STR専用版"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    str_values: np.ndarray                # STR値
    str_percentile: np.ndarray            # STRパーセンタイル
    str_velocity: np.ndarray              # STR変化率（速度）
    str_acceleration: np.ndarray          # STR変化加速度


@njit(fastmath=True, cache=True)
def calculate_str_percentile(str_values: np.ndarray, lookback_period: int) -> np.ndarray:
    """
    STRパーセンタイル計算 - 高精度版
    """
    length = len(str_values)
    percentiles = np.zeros(length)
    
    for i in range(lookback_period, length):
        # 過去のSTR値を取得
        historical_values = str_values[i-lookback_period:i]
        
        # 現在値との比較
        current_value = str_values[i]
        
        # パーセンタイル計算（高精度）
        count_below = 0
        count_equal = 0
        
        for val in historical_values:
            if val < current_value:
                count_below += 1
            elif val == current_value:
                count_equal += 1
        
        # より正確なパーセンタイル計算
        if len(historical_values) > 0:
            percentiles[i] = (count_below + count_equal * 0.5) / len(historical_values)
        else:
            percentiles[i] = 0.5
    
    return percentiles


@njit(fastmath=True, cache=True)
def calculate_str_velocity(str_values: np.ndarray, period: int = 3) -> np.ndarray:
    """
    STR変化率（速度）計算 - 動的適応版
    """
    length = len(str_values)
    velocity = np.zeros(length)
    
    for i in range(period, length):
        # 短期変化率
        if str_values[i-period] > 0:
            velocity[i] = (str_values[i] - str_values[i-period]) / str_values[i-period]
        else:
            velocity[i] = 0.0
    
    return velocity


@njit(fastmath=True, cache=True)
def calculate_str_acceleration(str_velocity: np.ndarray, period: int = 3) -> np.ndarray:
    """
    STR変化加速度計算 - 体制変化検出用
    """
    length = len(str_velocity)
    acceleration = np.zeros(length)
    
    for i in range(period, length):
        # 速度の変化率
        if abs(str_velocity[i-period]) > 1e-10:  # ゼロ除算防止
            acceleration[i] = str_velocity[i] - str_velocity[i-period]
        else:
            acceleration[i] = 0.0
    
    return acceleration


@njit(fastmath=True, parallel=True, cache=True)
def ultimate_str_volatility_fusion(
    str_values: np.ndarray,
    str_percentile: np.ndarray,
    str_velocity: np.ndarray,
    str_acceleration: np.ndarray,
    high_vol_threshold: float = 0.75,
    low_vol_threshold: float = 0.25,
    velocity_threshold: float = 0.05,
    acceleration_threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最強のSTRベース・ボラティリティ融合アルゴリズム
    
    シンプルで洗練された多角的判定:
    1. STRパーセンタイル（相対的位置）
    2. STR変化率（動的変化）
    3. STR加速度（体制変化）
    4. 動的適応閾値
    """
    length = len(str_values)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 1. STRパーセンタイルスコア（最も重要）
        percentile_score = str_percentile[i] if not np.isnan(str_percentile[i]) else 0.5
        
        # 2. STR変化率スコア（動的変化検出）
        velocity_score = 0.5
        if not np.isnan(str_velocity[i]):
            # 正規化された速度スコア
            velocity_score = min(max(abs(str_velocity[i]) / velocity_threshold, 0.0), 1.0)
        
        # 3. STR加速度スコア（体制変化検出）
        acceleration_score = 0.5
        if not np.isnan(str_acceleration[i]):
            # 正規化された加速度スコア
            acceleration_score = min(max(abs(str_acceleration[i]) / acceleration_threshold, 0.0), 1.0)
        
        # 4. 動的適応重み計算
        # パーセンタイルが極端な場合は重要度を上げる
        percentile_weight = 0.6 + 0.2 * abs(percentile_score - 0.5) * 2
        velocity_weight = 0.25 - 0.05 * abs(percentile_score - 0.5) * 2
        acceleration_weight = 0.15 - 0.05 * abs(percentile_score - 0.5) * 2
        
        # 重み正規化
        total_weight = percentile_weight + velocity_weight + acceleration_weight
        if total_weight > 0:
            percentile_weight /= total_weight
            velocity_weight /= total_weight
            acceleration_weight /= total_weight
        
        # 5. 最終スコア計算
        score = (percentile_weight * percentile_score + 
                 velocity_weight * velocity_score + 
                 acceleration_weight * acceleration_score)
        
        raw_score[i] = score
        
        # 6. 確率計算（シグモイド関数で滑らかに）
        # シンプルな確率変換
        probability[i] = score
        
        # 7. 高精度ヒステリシス判定
        if i > 0:
            prev_state = state[i-1]
            
            # 動的閾値調整
            dynamic_high_threshold = high_vol_threshold
            dynamic_low_threshold = low_vol_threshold
            
            # 速度と加速度による閾値調整
            if velocity_score > 0.7:  # 高速変化時
                dynamic_high_threshold -= 0.05  # 高ボラ判定しやすく
                dynamic_low_threshold += 0.05   # 低ボラ判定しにくく
            
            if acceleration_score > 0.7:  # 高加速度時
                dynamic_high_threshold -= 0.03  # さらに高ボラ判定しやすく
                dynamic_low_threshold += 0.03   # さらに低ボラ判定しにくく
            
            # 境界値制限
            dynamic_high_threshold = max(dynamic_high_threshold, 0.5)
            dynamic_low_threshold = min(dynamic_low_threshold, 0.5)
            
            # 状態判定
            if prev_state == 0:  # 前回が低ボラティリティ
                state[i] = 1 if score > dynamic_high_threshold else 0
            else:  # 前回が高ボラティリティ
                state[i] = 0 if score < dynamic_low_threshold else 1
        else:
            # 初回判定
            state[i] = 1 if score > (high_vol_threshold + low_vol_threshold) / 2 else 0
    
    return state, probability, raw_score


class PracticalVolatilityState(Indicator):
    """
    実践的ボラティリティ状態判別インジケーター - STR専用版
    
    最強のSTRベース・アプローチ:
    1. STRによる高精度ボラティリティ測定
    2. STR変化率による動的変化検出
    3. STR加速度による体制変化検出
    4. 動的適応閾値による高精度判定
    5. 低遅延・高精度のヒステリシス判定
    
    特徴:
    - 超高精度: STR単体による洗練された判定
    - 低遅延: Ultimate Smootherによる最小遅延
    - 動的適応: 市場状況に応じた閾値調整
    - シンプル: 複雑さを排除した効果的アルゴリズム
    - 高性能: 並列処理による高速計算
    """
    
    def __init__(
        self,
        str_period: float = 20.0,             # STR計算期間
        percentile_lookback: int = 252,       # パーセンタイル計算期間（約1年）
        high_vol_threshold: float = 0.75,     # 高ボラティリティ閾値
        low_vol_threshold: float = 0.25,      # 低ボラティリティ閾値
        velocity_threshold: float = 0.05,     # 速度閾値
        acceleration_threshold: float = 0.02, # 加速度閾値
        velocity_period: int = 3,             # 速度計算期間
        acceleration_period: int = 3,         # 加速度計算期間
        src_type: str = 'ukf_hlc3',           # 価格ソース
        smoothing: bool = True,               # スムージングの有効化
        # 動的適応パラメータ
        dynamic_adaptation: bool = True,      # 動的適応の有効化
        cycle_detector_type: str = 'absolute_ultimate'  # サイクル検出器タイプ
    ):
        """
        コンストラクタ
        
        Args:
            str_period: STR計算期間
            percentile_lookback: パーセンタイル計算の振り返り期間
            high_vol_threshold: 高ボラティリティ判定閾値
            low_vol_threshold: 低ボラティリティ判定閾値
            velocity_threshold: STR変化率の正規化閾値
            acceleration_threshold: STR加速度の正規化閾値
            velocity_period: 速度計算期間
            acceleration_period: 加速度計算期間
            src_type: 価格ソースタイプ
            smoothing: 最終結果のスムージング
            dynamic_adaptation: 動的適応の有効化
            cycle_detector_type: サイクル検出器タイプ
        """
        super().__init__(f"PracticalVolatilityState_STR(period={str_period}, thresholds={low_vol_threshold}-{high_vol_threshold})")
        
        self.str_period = str_period
        self.percentile_lookback = percentile_lookback
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.velocity_period = velocity_period
        self.acceleration_period = acceleration_period
        self.src_type = src_type.lower()
        self.smoothing = smoothing
        self.dynamic_adaptation = dynamic_adaptation
        self.cycle_detector_type = cycle_detector_type
        
        # STRインジケーターの初期化
        self.str_indicator = STR(
            period=self.str_period,
            src_type=self.src_type,
            period_mode='dynamic' if self.dynamic_adaptation else 'fixed',
            cycle_detector_type=self.cycle_detector_type
        )
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=3, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
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
            
            params_sig = f"{self.str_period}_{self.percentile_lookback}_{self.high_vol_threshold}_{self.low_vol_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.str_period}_{self.percentile_lookback}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> PracticalVolatilityStateResult:
        """
        実践的ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            PracticalVolatilityStateResult: 判定結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                return self._result_cache[data_hash]
            
            # データ長チェック
            length = len(data)
            min_required = max(self.str_period, self.percentile_lookback // 4)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. STR計算（動的適応対応）
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            if len(str_values) == 0:
                return self._create_empty_result(length)
            
            # 2. STRパーセンタイル計算
            str_percentile = calculate_str_percentile(str_values, self.percentile_lookback)
            
            # 3. STR変化率（速度）計算
            str_velocity = calculate_str_velocity(str_values, self.velocity_period)
            
            # 4. STR加速度計算
            str_acceleration = calculate_str_acceleration(str_velocity, self.acceleration_period)
            
            # 5. 最強STR融合アルゴリズム
            state, probability, raw_score = ultimate_str_volatility_fusion(
                str_values, str_percentile, str_velocity, str_acceleration,
                self.high_vol_threshold, self.low_vol_threshold,
                self.velocity_threshold, self.acceleration_threshold
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
            result = PracticalVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                str_values=str_values,
                str_percentile=str_percentile,
                str_velocity=str_velocity,
                str_acceleration=str_acceleration
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"実践的ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> PracticalVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return PracticalVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            raw_score=empty_array,
            str_values=empty_array,
            str_percentile=empty_array,
            str_velocity=empty_array,
            str_acceleration=empty_array
        )
    
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
                'str_percentile': latest_result.str_percentile,
                'str_velocity': latest_result.str_velocity,
                'str_acceleration': latest_result.str_acceleration
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
    
    def get_current_regime(self) -> str:
        """現在の市場体制を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.str_acceleration) > 0:
                acceleration_val = latest_result.str_acceleration[-1]
                velocity_val = latest_result.str_velocity[-1] if len(latest_result.str_velocity) > 0 else 0
                
                # STR加速度と速度による体制判定
                if abs(acceleration_val) > self.acceleration_threshold:
                    if velocity_val > self.velocity_threshold:
                        return "Accelerating_High"
                    elif velocity_val < -self.velocity_threshold:
                        return "Accelerating_Low"
                    else:
                        return "Accelerating"
                elif abs(velocity_val) > self.velocity_threshold:
                    return "Dynamic"
                else:
                    return "Stable"
        return "Unknown"
    
    def get_str_metrics(self) -> Optional[Dict[str, float]]:
        """STR関連メトリクスを取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.str_values) > 0:
                return {
                    'current_str': float(latest_result.str_values[-1]),
                    'str_percentile': float(latest_result.str_percentile[-1]) if len(latest_result.str_percentile) > 0 else 0.0,
                    'str_velocity': float(latest_result.str_velocity[-1]) if len(latest_result.str_velocity) > 0 else 0.0,
                    'str_acceleration': float(latest_result.str_acceleration[-1]) if len(latest_result.str_acceleration) > 0 else 0.0,
                    'volatility_score': float(latest_result.raw_score[-1]) if len(latest_result.raw_score) > 0 else 0.0,
                    'volatility_probability': float(latest_result.probability[-1]) if len(latest_result.probability) > 0 else 0.0
                }
        return None
    
    def is_volatility_expanding(self) -> bool:
        """ボラティリティが拡大しているかどうか"""
        metrics = self.get_str_metrics()
        if metrics:
            return metrics['str_velocity'] > self.velocity_threshold
        return False
    
    def is_volatility_contracting(self) -> bool:
        """ボラティリティが収縮しているかどうか"""
        metrics = self.get_str_metrics()
        if metrics:
            return metrics['str_velocity'] < -self.velocity_threshold
        return False
    
    def get_volatility_strength(self) -> str:
        """ボラティリティの強さを取得"""
        metrics = self.get_str_metrics()
        if metrics:
            percentile = metrics['str_percentile']
            if percentile > 0.9:
                return "Extreme_High"
            elif percentile > 0.75:
                return "High"
            elif percentile > 0.6:
                return "Moderate_High"
            elif percentile < 0.1:
                return "Extreme_Low"
            elif percentile < 0.25:
                return "Low"
            elif percentile < 0.4:
                return "Moderate_Low"
            else:
                return "Normal"
        return "Unknown"

    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        if self.smoothing:
            self.smoother.reset()