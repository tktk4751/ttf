#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.trend_filter.correlation_cycle_indicator import CorrelationCycleIndicator


@njit(fastmath=True, parallel=True)
def generate_signals_numba(state_values: np.ndarray) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        state_values: 相関サイクルインジケーターの状態値の配列
    
    Returns:
        シグナルの配列 (1: 上昇トレンド, 0: サイクルモード, -1: 下降トレンド)
    """
    length = len(state_values)
    signals = np.zeros(length, dtype=np.float64)
    
    for i in prange(length):
        if np.isnan(state_values[i]):
            signals[i] = np.nan
        else:
            signals[i] = state_values[i]  # stateの値をそのまま使用
    
    return signals


class CorrelationCycleFilterSignal(BaseSignal, IFilterSignal):
    """
    相関サイクル分析を使用したフィルターシグナル
    
    特徴:
    - John Ehlersの相関サイクル論文に基づくトレンド・サイクル判定
    - 価格とコサイン/負サイン波の相関による高精度な市場状態検出
    - フェーザー角度と角度変化率による動的判定
    - 直交成分使用による6dBのSN比改善
    - Real vs Imaginary成分比較による追加検証
    
    動作:
    - State = 1: 上昇トレンド（角度変化率<閾値 かつ 角度≥0°）
    - State = 0: サイクルモード（角度変化率≥閾値）
    - State = -1: 下降トレンド（角度変化率<閾値 かつ 角度<0°）
    
    使用方法:
    - トレンド系戦略とサイクル系戦略の自動切り替え
    - エントリー条件の最適化
    - 市場状態に応じたリスク管理
    """
    
    def __init__(
        self,
        period: int = 20,                     # 相関計算期間
        src_type: str = 'close',              # ソースタイプ
        trend_threshold: float = 9.0,         # トレンド判定閾値（角度変化率）
        use_theoretical_input: bool = False,  # 理論的入力を使用するか（テスト用）
        theoretical_period: int = 20          # 理論的サイン波の周期（テスト用）
    ):
        """
        コンストラクタ
        
        Args:
            period: 相関計算期間（デフォルト: 20）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 9.0度）
            use_theoretical_input: 理論的入力を使用するか（デフォルト: False）
            theoretical_period: 理論的サイン波の周期（デフォルト: 20）
        """
        # パラメータの設定
        params = {
            'period': period,
            'src_type': src_type,
            'trend_threshold': trend_threshold,
            'use_theoretical_input': use_theoretical_input,
            'theoretical_period': theoretical_period
        }
        
        super().__init__(
            f"CorrelationCycleFilter(period={period}, threshold={trend_threshold:.1f}, theoretical={'Y' if use_theoretical_input else 'N'})",
            params
        )
        
        # 相関サイクルインジケーターの初期化
        self._filter = CorrelationCycleIndicator(
            period=period,
            src_type=src_type,
            trend_threshold=trend_threshold,
            use_theoretical_input=use_theoretical_input,
            theoretical_period=theoretical_period
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
            シグナルの配列 (1: 上昇トレンド, 0: サイクルモード, -1: 下降トレンド)
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
            
            # 相関サイクルインジケーターの計算
            result = self._filter.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if result is None or len(result.state) == 0:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            # 状態値の取得
            state_values = result.state
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(state_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"CorrelationCycleFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_angle_values(self) -> np.ndarray:
        """
        フェーザー角度を取得する
        
        Returns:
            フェーザー角度の配列（-180°〜+180°）
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.angle
        return np.array([])
    
    def get_real_component(self) -> np.ndarray:
        """
        Real成分（コサイン相関）を取得する
        
        Returns:
            Real成分の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.real_component
        return np.array([])
    
    def get_imag_component(self) -> np.ndarray:
        """
        Imaginary成分（負サイン相関）を取得する
        
        Returns:
            Imaginary成分の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.imag_component
        return np.array([])
    
    def get_state_values(self) -> np.ndarray:
        """
        市場状態値を取得する
        
        Returns:
            市場状態値の配列 (1=上昇トレンド、0=サイクル、-1=下降トレンド)
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.state
        return np.array([])
    
    def get_rate_of_change(self) -> np.ndarray:
        """
        角度変化率を取得する
        
        Returns:
            角度変化率の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.rate_of_change
        return np.array([])
    
    def get_cycle_mode(self) -> np.ndarray:
        """
        サイクルモード判定を取得する
        
        Returns:
            サイクルモード判定の配列 (1=サイクル、0=トレンド)
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.cycle_mode
        return np.array([])
    
    def get_ri_mode(self) -> np.ndarray:
        """
        Real vs Imaginaryモード判定を取得する
        
        Returns:
            Real vs Imaginaryモード判定の配列 (1=トレンド、0=サイクル)
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.ri_mode
        return np.array([])
    
    def get_orthogonal_components(self) -> tuple:
        """
        直交成分（Real, Imaginary）を取得する
        
        Returns:
            (Real, Imaginary)成分のタプル
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.real_component, result.imag_component
        return np.array([]), np.array([])
    
    def get_correlation_strength(self) -> np.ndarray:
        """
        相関強度を取得する（Real^2 + Imaginary^2の平方根）
        
        Returns:
            相関強度の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return np.sqrt(result.real_component**2 + result.imag_component**2)
        return np.array([])
    
    def get_trend_confidence(self) -> np.ndarray:
        """
        トレンド信頼度を取得する（角度変化率の逆数）
        
        Returns:
            トレンド信頼度の配列（値が大きいほど高信頼度）
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            # 角度変化率が小さいほど信頼度が高い
            roc = result.rate_of_change
            confidence = np.zeros_like(roc)
            valid_mask = roc > 0
            confidence[valid_mask] = 1.0 / (1.0 + roc[valid_mask] / 10.0)  # 正規化
            return confidence
        return np.array([])
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._filter, 'reset'):
            self._filter.reset()
        self._signals = None
        self._data_hash = None