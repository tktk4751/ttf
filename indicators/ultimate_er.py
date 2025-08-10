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
    from .efficiency_ratio import calculate_efficiency_ratio_for_period, calculate_trend_signals_with_range
    from .smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from .smoother.ultimate_smoother import UltimateSmoother
    from .price_source import PriceSource
    from .ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from efficiency_ratio import calculate_efficiency_ratio_for_period, calculate_trend_signals_with_range
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class UltimateERResult:
    """Ultimate Efficiency Ratio計算結果"""
    values: np.ndarray              # Ultimate ER値（0-1の範囲）
    raw_er: np.ndarray              # 元のER値
    ukf_filtered: np.ndarray        # UKFフィルター後のER値
    trend_signals: np.ndarray       # トレンド信号（1=up, -1=down, 0=range）
    confidence_scores: np.ndarray   # 信頼度スコア
    dynamic_periods: np.ndarray     # 動的期間
    current_trend: str              # 現在のトレンド状態


@njit(fastmath=True, cache=True)
def calculate_dynamic_er_with_periods(
    prices: np.ndarray,
    periods: np.ndarray
) -> np.ndarray:
    """
    動的期間でEfficiency Ratioを計算
    
    Args:
        prices: 価格データ
        periods: 各時点での期間
    
    Returns:
        ER値の配列
    """
    n = len(prices)
    er_values = np.zeros(n)
    
    for i in range(n):
        period = int(periods[i])
        if period < 2:
            period = 2
        
        if i >= period:
            # 価格変化
            change = abs(prices[i] - prices[i-period])
            
            # ボラティリティ（価格変化の絶対値の合計）
            volatility = 0.0
            for j in range(i-period+1, i+1):
                volatility += abs(prices[j] - prices[j-1])
            
            # ER計算
            if volatility > 1e-10:
                er_values[i] = change / volatility
            else:
                er_values[i] = 0.0
        else:
            er_values[i] = 0.5  # デフォルト値
    
    return er_values


@njit(fastmath=True, cache=True)
def apply_confidence_weighting(
    er_values: np.ndarray,
    confidence_scores: np.ndarray
) -> np.ndarray:
    """
    信頼度スコアによる重み付けを適用
    
    Args:
        er_values: ER値
        confidence_scores: 信頼度スコア（0-1）
    
    Returns:
        重み付けされたER値
    """
    n = len(er_values)
    weighted_er = np.zeros(n)
    
    for i in range(n):
        # 信頼度が低い場合は中央値（0.5）に近づける
        weighted_er[i] = er_values[i] * confidence_scores[i] + 0.5 * (1.0 - confidence_scores[i])
    
    return weighted_er


class UltimateER(Indicator):
    """
    Ultimate Efficiency Ratio インジケーター
    
    5期間のERをUKF（Unscented Kalman Filter）でフィルタリングし、
    20期間のUltimate Smootherで平滑化した高精度ER
    
    特徴:
    - 短期間(5期間)ERによる高感度な変化検出
    - UKFによるノイズ除去と状態推定
    - Ultimate Smoother(20期間)による超低遅延平滑化
    - 信頼度スコアによる適応的重み付け
    - 高精度なトレンド/レンジ判定
    """
    
    def __init__(
        self,
        er_period: float = 5.0,  # ER計算期間（固定5期間）
        src_type: str = 'hlc3',
        # UKFパラメータ
        ukf_alpha: float = 0.001,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        ukf_process_noise: float = 0.001,
        ukf_volatility_window: int = 10,
        ukf_adaptive_noise: bool = True,
        # Ultimate Smootherパラメータ（固定20期間）
        smoother_period: float = 20.0,
        # トレンド判定パラメータ
        slope_index: int = 3,
        range_threshold: float = 0.005,
        # 信頼度重み付け
        use_confidence_weighting: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            er_period: ER計算期間（デフォルト: 5.0）
            src_type: 価格ソースタイプ
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            ukf_process_noise: UKFプロセスノイズスケール
            ukf_volatility_window: UKFボラティリティ計算窓
            ukf_adaptive_noise: UKF適応的ノイズ推定
            smoother_period: Ultimate Smoother期間（デフォルト: 20.0）
            slope_index: トレンド判定期間
            range_threshold: レンジ判定閾値
            use_confidence_weighting: 信頼度重み付けを使用するか
        """
        indicator_name = f"UltimateER(er={er_period}, ukf={ukf_alpha}, smooth={smoother_period})"
        super().__init__(indicator_name)
        
        self.er_period = er_period
        self.src_type = src_type
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        self.use_confidence_weighting = use_confidence_weighting
        
        # UKFパラメータ
        self.ukf_alpha = ukf_alpha
        self.ukf_beta = ukf_beta
        self.ukf_kappa = ukf_kappa
        self.ukf_process_noise = ukf_process_noise
        self.ukf_volatility_window = ukf_volatility_window
        self.ukf_adaptive_noise = ukf_adaptive_noise
        
        # Ultimate Smootherパラメータ
        self.smoother_period = smoother_period
        
        # パラメータ検証
        if self.er_period < 2:
            raise ValueError("er_periodは2以上である必要があります")
        
        # UKFインジケーターの初期化
        self.ukf_indicator = UnscentedKalmanFilter(
            src_type='close',  # ER値に対して適用
            alpha=ukf_alpha,
            beta=ukf_beta,
            kappa=ukf_kappa,
            process_noise_scale=ukf_process_noise,
            volatility_window=ukf_volatility_window,
            adaptive_noise=ukf_adaptive_noise
        )
        
        # Ultimate Smootherの初期化（固定20期間）
        self.ultimate_smoother = UltimateSmoother(
            period=smoother_period,
            src_type='close',  # UKF済みER値に対して適用
            period_mode='fixed'  # 固定期間モード
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
            
            params_sig = f"{self.er_period}_{self.src_type}_{self.ukf_alpha}_{self.smoother_period}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.er_period}_{self.smoother_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateERResult:
        """
        Ultimate Efficiency Ratioを計算
        
        Args:
            data: 価格データ
        
        Returns:
            UltimateERResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # 価格データの抽出
            prices = PriceSource.calculate_source(data, self.src_type)
            prices = np.asarray(prices, dtype=np.float64)
            
            n = len(prices)
            if n < self.er_period + 5:
                return self._create_empty_result(n)
            
            # 1. 5期間の固定ERを計算
            raw_er = calculate_efficiency_ratio_for_period(prices, int(self.er_period))
            
            # 動的期間は使用しないので固定値を設定
            dynamic_periods = np.full(n, self.er_period)
            
            # 2. UKFによるフィルタリング
            # ER値をDataFrameに変換してUKFに渡す
            er_df = pd.DataFrame({'close': raw_er})
            ukf_result = self.ukf_indicator.calculate(er_df)
            ukf_filtered = ukf_result.filtered_values
            confidence_scores = ukf_result.confidence_scores
            
            # 3. 信頼度重み付け（オプション）
            if self.use_confidence_weighting:
                ukf_weighted = apply_confidence_weighting(ukf_filtered, confidence_scores)
            else:
                ukf_weighted = ukf_filtered
            
            # 4. Ultimate Smootherによる平滑化（20期間）
            # UKF済みER値をDataFrameに変換
            smooth_df = pd.DataFrame({'close': ukf_weighted})
            smoother_result = self.ultimate_smoother.calculate(smooth_df)
            ultimate_er = smoother_result.values
            
            # 5. 値を0-1の範囲にクリップ
            ultimate_er = np.clip(ultimate_er, 0.0, 1.0)
            
            # 6. トレンド信号の計算
            trend_signals = calculate_trend_signals_with_range(
                ultimate_er, self.slope_index, self.range_threshold
            )
            
            # 7. 現在のトレンド判定
            if len(trend_signals) > 0:
                current_signal = trend_signals[-1]
                if current_signal == 1:
                    current_trend = 'up'
                elif current_signal == -1:
                    current_trend = 'down'
                else:
                    current_trend = 'range'
            else:
                current_trend = 'range'
            
            # 結果作成
            result = UltimateERResult(
                values=ultimate_er,
                raw_er=raw_er,
                ukf_filtered=ukf_filtered,
                trend_signals=trend_signals,
                confidence_scores=confidence_scores,
                dynamic_periods=dynamic_periods,
                current_trend=current_trend
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = ultimate_er  # 基底クラス要件
            
            self.logger.debug(f"Ultimate ER計算完了 - 平均値: {np.mean(ultimate_er[~np.isnan(ultimate_er)]):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ultimate ER計算エラー: {e}")
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            return self._create_empty_result(n)
    
    def _create_empty_result(self, length: int) -> UltimateERResult:
        """空の結果を作成"""
        return UltimateERResult(
            values=np.full(length, np.nan),
            raw_er=np.full(length, np.nan),
            ukf_filtered=np.full(length, np.nan),
            trend_signals=np.zeros(length, dtype=np.int8),
            confidence_scores=np.full(length, 0.5),
            dynamic_periods=np.full(length, self.er_period),
            current_trend='range'
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """Ultimate ER値を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.values.copy()
    
    def get_raw_er(self) -> Optional[np.ndarray]:
        """元のER値を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.raw_er.copy()
    
    def get_ukf_filtered(self) -> Optional[np.ndarray]:
        """UKFフィルター後のER値を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.ukf_filtered.copy()
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.trend_signals.copy()
    
    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得"""
        if not self._result_cache or not self._cache_keys:
            return 'range'
        result = self._result_cache[self._cache_keys[-1]]
        return result.current_trend
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.confidence_scores.copy()
    
    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """動的期間を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.dynamic_periods.copy()
    
    def is_trending(self) -> bool:
        """現在がトレンド状態かを判定"""
        return self.get_current_trend() in ['up', 'down']
    
    def is_ranging(self) -> bool:
        """現在がレンジ状態かを判定"""
        return self.get_current_trend() == 'range'
    
    def get_efficiency_score(self) -> float:
        """現在の効率スコアを取得（0-1）"""
        values = self.get_values()
        if values is None or len(values) == 0:
            return 0.5
        return float(values[-1]) if not np.isnan(values[-1]) else 0.5
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self.ukf_indicator.reset()
        self.ultimate_smoother.reset()