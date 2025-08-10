#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Percentile Analysis Utils - パーセンタイル分析ユーティリティ** 🎯

インジケーター値のパーセンタイル分析とトレンド・ボラティリティ分類のための
共通ユーティリティ関数群。

🌟 **主要機能:**
1. **パーセンタイル計算**: 高精度パーセンタイル計算
2. **分類ロジック**: パーセンタイル値に基づくトレンド/ボラティリティ分類
3. **統計的分析**: パーセンタイル分布の要約統計

📊 **用途:**
- トレンドフィルターのパーセンタイル分析
- ボラティリティインジケーターの相対評価
- マーケット状態の分類とシグナル生成
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def calculate_percentile(
    values: np.ndarray, 
    lookback_period: int
) -> np.ndarray:
    """
    高精度パーセンタイル計算（汎用版）
    
    Args:
        values: 計算対象の値の配列
        lookback_period: ルックバック期間
        
    Returns:
        パーセンタイル値の配列（0-1の範囲）
    """
    length = len(values)
    percentiles = np.zeros(length, dtype=np.float64)
    
    for i in range(lookback_period, length):
        # 過去の値を取得
        historical_values = values[i-lookback_period:i]
        
        # 現在値との比較
        current_value = values[i]
        
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
def calculate_trend_classification(
    percentiles: np.ndarray,
    indicator_values: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> tuple:
    """
    パーセンタイルに基づくトレンド分類
    
    Args:
        percentiles: パーセンタイル値
        indicator_values: インジケーター値
        low_threshold: 低トレンド閾値
        high_threshold: 高トレンド閾値
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (トレンド状態, トレンド強度)
    """
    length = len(percentiles)
    trend_state = np.full(length, np.nan, dtype=np.float64)
    trend_intensity = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(percentiles[i]):
            percentile = percentiles[i]
            
            # トレンド状態の分類
            if percentile <= low_threshold:
                trend_state[i] = -1.0  # 低トレンド/レンジ状態
            elif percentile >= high_threshold:
                trend_state[i] = 1.0   # 高トレンド状態
            else:
                trend_state[i] = 0.0   # 中トレンド状態
            
            # トレンド強度（0-1の範囲で正規化）
            if percentile <= 0.5:
                # 低トレンド側の強度
                trend_intensity[i] = (0.5 - percentile) / 0.5
            else:
                # 高トレンド側の強度
                trend_intensity[i] = (percentile - 0.5) / 0.5
    
    return trend_state, trend_intensity


@njit(fastmath=True, cache=True)
def calculate_volatility_classification(
    percentiles: np.ndarray,
    indicator_values: np.ndarray,
    low_threshold: float = 0.25,
    high_threshold: float = 0.75
) -> tuple:
    """
    パーセンタイルに基づくボラティリティ分類
    
    Args:
        percentiles: パーセンタイル値
        indicator_values: インジケーター値
        low_threshold: 低ボラティリティ閾値
        high_threshold: 高ボラティリティ閾値
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ボラティリティ状態, ボラティリティ強度)
    """
    length = len(percentiles)
    volatility_state = np.full(length, np.nan, dtype=np.float64)
    volatility_intensity = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(percentiles[i]):
            percentile = percentiles[i]
            
            # ボラティリティ状態の分類
            if percentile <= low_threshold:
                volatility_state[i] = 1.0  # 低ボラティリティ
            elif percentile >= high_threshold:
                volatility_state[i] = -1.0  # 高ボラティリティ
            else:
                volatility_state[i] = 0.0  # 中ボラティリティ
            
            # ボラティリティ強度（0-1の範囲で正規化）
            if percentile <= 0.5:
                # 低ボラティリティ側の強度
                volatility_intensity[i] = (0.5 - percentile) / 0.5
            else:
                # 高ボラティリティ側の強度
                volatility_intensity[i] = (percentile - 0.5) / 0.5
    
    return volatility_state, volatility_intensity


def calculate_percentile_summary(
    percentiles: np.ndarray,
    state_values: np.ndarray = None,
    enable_percentile_analysis: bool = True,
    lookback_period: int = 50,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    パーセンタイル分析の要約情報を計算
    
    Args:
        percentiles: パーセンタイル値
        state_values: 分類状態値（オプション）
        enable_percentile_analysis: パーセンタイル分析が有効か
        lookback_period: ルックバック期間
        low_threshold: 低閾値
        high_threshold: 高閾値
        
    Returns:
        パーセンタイル分析の要約辞書
    """
    summary = {
        'percentile_analysis_enabled': enable_percentile_analysis,
        'lookback_period': lookback_period,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }
    
    if percentiles is not None:
        valid_percentiles = percentiles[~np.isnan(percentiles)]
        
        if len(valid_percentiles) > 0:
            summary.update({
                'percentile_mean': np.mean(valid_percentiles),
                'percentile_std': np.std(valid_percentiles),
                'percentile_min': np.min(valid_percentiles),
                'percentile_max': np.max(valid_percentiles),
                'current_percentile': percentiles[-1] if not np.isnan(percentiles[-1]) else None
            })
    
    if state_values is not None:
        valid_state = state_values[~np.isnan(state_values)]
        
        if len(valid_state) > 0:
            # 各状態の分布を計算
            low_count = np.sum(valid_state == 1.0)
            mid_count = np.sum(valid_state == 0.0) 
            high_count = np.sum(valid_state == -1.0)
            total_count = len(valid_state)
            
            summary.update({
                'state_distribution': {
                    'low': low_count / total_count,
                    'medium': mid_count / total_count,
                    'high': high_count / total_count
                },
                'current_state': state_values[-1] if not np.isnan(state_values[-1]) else None
            })
    
    return summary


class PercentileAnalysisMixin:
    """
    パーセンタイル分析機能を提供するMixinクラス
    インジケータークラスに組み込んで使用
    """
    
    def _add_percentile_analysis_params(self, **kwargs):
        """パーセンタイル分析パラメータを初期化"""
        self.enable_percentile_analysis = kwargs.get('enable_percentile_analysis', True)
        self.percentile_lookback_period = kwargs.get('percentile_lookback_period', 50)
        self.percentile_low_threshold = kwargs.get('percentile_low_threshold', 0.25)
        self.percentile_high_threshold = kwargs.get('percentile_high_threshold', 0.75)
    
    def _calculate_percentile_analysis(
        self, 
        indicator_values: np.ndarray,
        analysis_type: str = 'trend'  # 'trend' or 'volatility'
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        パーセンタイル分析を実行
        
        Args:
            indicator_values: インジケーター値
            analysis_type: 分析タイプ ('trend' または 'volatility')
            
        Returns:
            Tuple[パーセンタイル値, 状態値, 強度値]
        """
        if not self.enable_percentile_analysis:
            return None, None, None
        
        try:
            # パーセンタイル計算
            percentiles = calculate_percentile(
                indicator_values, self.percentile_lookback_period
            )
            
            # 状態分類
            if analysis_type == 'trend':
                state_values, intensity_values = calculate_trend_classification(
                    percentiles, indicator_values,
                    self.percentile_low_threshold, self.percentile_high_threshold
                )
            else:  # volatility
                state_values, intensity_values = calculate_volatility_classification(
                    percentiles, indicator_values,
                    self.percentile_low_threshold, self.percentile_high_threshold
                )
            
            return percentiles, state_values, intensity_values
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"パーセンタイル分析中にエラー: {e}")
            return None, None, None
    
    def _get_percentile_analysis_summary(
        self, 
        percentiles: np.ndarray, 
        state_values: np.ndarray
    ) -> Dict[str, Any]:
        """パーセンタイル分析要約を取得"""
        return calculate_percentile_summary(
            percentiles, state_values,
            self.enable_percentile_analysis,
            self.percentile_lookback_period,
            self.percentile_low_threshold,
            self.percentile_high_threshold
        )


# 便利関数
def add_percentile_to_convenience_function(
    original_function_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    便利関数にパーセンタイル分析パラメータを追加
    
    Args:
        original_function_kwargs: 元の関数のkwargs
        
    Returns:
        パーセンタイル分析パラメータが追加されたkwargs
    """
    percentile_params = {
        'enable_percentile_analysis': True,
        'percentile_lookback_period': 50,
        'percentile_low_threshold': 0.25,
        'percentile_high_threshold': 0.75
    }
    
    # 既存のパラメータを優先
    return {**percentile_params, **original_function_kwargs}


if __name__ == "__main__":
    """直接実行時のテスト"""
    print("=== パーセンタイル分析ユーティリティのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
    values = np.random.normal(50, 10, length)  # 平均50、標準偏差10の正規分布
    
    # パーセンタイル計算
    percentiles = calculate_percentile(values, 30)
    
    # トレンド分類
    trend_state, trend_intensity = calculate_trend_classification(percentiles, values)
    
    # ボラティリティ分類
    vol_state, vol_intensity = calculate_volatility_classification(percentiles, values)
    
    # 要約統計
    trend_summary = calculate_percentile_summary(percentiles, trend_state)
    vol_summary = calculate_percentile_summary(percentiles, vol_state)
    
    print(f"有効パーセンタイル値数: {np.sum(~np.isnan(percentiles))}")
    print(f"パーセンタイル範囲: {np.nanmin(percentiles):.3f} - {np.nanmax(percentiles):.3f}")
    print(f"トレンド分布: {trend_summary.get('state_distribution', {})}")
    print(f"ボラティリティ分布: {vol_summary.get('state_distribution', {})}")
    
    print("\nテスト完了")