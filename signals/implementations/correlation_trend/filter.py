#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.trend_filter.correlation_trend_indicator import CorrelationTrendIndicator


@njit(fastmath=True, parallel=True)
def generate_signals_numba(trend_signal_values: np.ndarray) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        trend_signal_values: 相関トレンドインジケーターのトレンドシグナル値の配列
    
    Returns:
        シグナルの配列 (1: 上昇トレンド, 0: 横這い, -1: 下降トレンド)
    """
    length = len(trend_signal_values)
    signals = np.zeros(length, dtype=np.float64)
    
    for i in prange(length):
        if np.isnan(trend_signal_values[i]):
            signals[i] = np.nan
        else:
            signals[i] = trend_signal_values[i]  # trend_signalの値をそのまま使用
    
    return signals


class CorrelationTrendFilterSignal(BaseSignal, IFilterSignal):
    """
    相関トレンド分析を使用したフィルターシグナル
    
    特徴:
    - John Ehlersの相関トレンド論文に基づくトレンド判定
    - 価格と正の傾きを持つ直線との相関係数計算
    - -1から+1の範囲でトレンド強度を表現
    - 閾値による明確なトレンド判定
    - 相関期間の約半分のラグで高精度判定
    - オプション平滑化によるノイズ除去
    
    動作:
    - trend_signal = 1: 上昇トレンド（相関値 > +閾値）
    - trend_signal = 0: 横這い（-閾値 ≤ 相関値 ≤ +閾値）
    - trend_signal = -1: 下降トレンド（相関値 < -閾値）
    
    使用方法:
    - トレンド系戦略のフィルター
    - エントリー・エグジット条件の最適化
    - 相場環境に応じた戦略切り替え
    """
    
    def __init__(
        self,
        length: int = 20,                     # 相関計算期間
        src_type: str = 'close',              # ソースタイプ
        trend_threshold: float = 0.3,         # トレンド判定閾値
        enable_smoothing: bool = False,       # 平滑化を有効にするか
        smooth_length: int = 5                # 平滑化期間
    ):
        """
        コンストラクタ
        
        Args:
            length: 相関計算期間（デフォルト: 20）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            trend_threshold: トレンド判定閾値（デフォルト: 0.3）
            enable_smoothing: 平滑化を有効にするか（デフォルト: False）
            smooth_length: 平滑化期間（デフォルト: 5）
        """
        # パラメータの設定
        params = {
            'length': length,
            'src_type': src_type,
            'trend_threshold': trend_threshold,
            'enable_smoothing': enable_smoothing,
            'smooth_length': smooth_length
        }
        
        super().__init__(
            f"CorrelationTrendFilter(length={length}, threshold={trend_threshold:.1f}, smooth={'Y' if enable_smoothing else 'N'})",
            params
        )
        
        # 相関トレンドインジケーターの初期化
        self._filter = CorrelationTrendIndicator(
            length=length,
            src_type=src_type,
            trend_threshold=trend_threshold,
            enable_smoothing=enable_smoothing,
            smooth_length=smooth_length
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{hash(frozenset(self._params.items()))}"
        
        return f"{data_hash}_{param_str}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            シグナルの配列 (1: 上昇トレンド, 0: 横這い, -1: 下降トレンド)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
            elif data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # 相関トレンドインジケーターの計算
            result = self._filter.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if result is None or len(result.trend_signal) == 0:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            # トレンドシグナル値の取得
            trend_signal_values = result.trend_signal
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(trend_signal_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"CorrelationTrendFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_correlation_values(self) -> np.ndarray:
        """
        相関係数値を取得する
        
        Returns:
            相関係数値の配列（-1〜+1の範囲）
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.values
        return np.array([])
    
    def get_trend_signal_values(self) -> np.ndarray:
        """
        トレンドシグナル値を取得する
        
        Returns:
            トレンドシグナル値の配列 (1=上昇、0=横這い、-1=下降)
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.trend_signal
        return np.array([])
    
    def get_trend_strength(self) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列（0〜1の範囲）
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.trend_strength
        return np.array([])
    
    def get_smoothed_values(self) -> np.ndarray:
        """
        平滑化された相関値を取得する
        
        Returns:
            平滑化された相関値の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.smoothed_values
        return np.array([])
    
    def get_trend_direction_confidence(self) -> np.ndarray:
        """
        トレンド方向の信頼度を取得する
        
        Returns:
            方向信頼度の配列（相関値の絶対値）
        """
        correlation_values = self.get_correlation_values()
        if len(correlation_values) > 0:
            return np.abs(correlation_values)
        return np.array([])
    
    def get_uptrend_strength(self) -> np.ndarray:
        """
        上昇トレンド強度を取得する（正の相関値のみ）
        
        Returns:
            上昇トレンド強度の配列
        """
        correlation_values = self.get_correlation_values()
        if len(correlation_values) > 0:
            uptrend_strength = np.maximum(correlation_values, 0.0)
            return uptrend_strength
        return np.array([])
    
    def get_downtrend_strength(self) -> np.ndarray:
        """
        下降トレンド強度を取得する（負の相関値の絶対値）
        
        Returns:
            下降トレンド強度の配列
        """
        correlation_values = self.get_correlation_values()
        if len(correlation_values) > 0:
            downtrend_strength = np.abs(np.minimum(correlation_values, 0.0))
            return downtrend_strength
        return np.array([])
    
    def get_sideways_strength(self) -> np.ndarray:
        """
        横這い強度を取得する（閾値内の相関値の逆数）
        
        Returns:
            横這い強度の配列
        """
        correlation_values = self.get_correlation_values()
        trend_threshold = self._params['trend_threshold']
        
        if len(correlation_values) > 0:
            # 閾値内の場合は横這い強度が高い
            abs_corr = np.abs(correlation_values)
            sideways_strength = np.zeros_like(abs_corr)
            sideways_mask = abs_corr <= trend_threshold
            sideways_strength[sideways_mask] = 1.0 - abs_corr[sideways_mask] / trend_threshold
            return sideways_strength
        return np.array([])
    
    def get_lag_estimate(self) -> int:
        """
        推定ラグを取得する
        
        Returns:
            推定ラグ（相関期間の約半分）
        """
        return self._params['length'] // 2
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._filter, 'reset'):
            self._filter.reset()
        self._signals = None
        self._data_hash = None