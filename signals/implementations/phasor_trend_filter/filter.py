#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.trend_filter.phasor_trend_filter import PhasorTrendFilter


@njit(fastmath=True, parallel=True)
def generate_signals_numba(state_values: np.ndarray) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        state_values: フェーザートレンドフィルターの状態値の配列
    
    Returns:
        シグナルの配列 (1: 上昇トレンド, 0: 中立/サイクリング, -1: 下降トレンド)
    """
    length = len(state_values)
    signals = np.zeros(length, dtype=np.float64)
    
    for i in prange(length):
        if np.isnan(state_values[i]):
            signals[i] = np.nan
        else:
            signals[i] = state_values[i]  # stateの値をそのまま使用
    
    return signals


class PhasorTrendFilterSignal(BaseSignal, IFilterSignal):
    """
    フェーザー分析トレンドフィルターを使用したフィルターシグナル
    
    特徴:
    - John Ehlersのフェーザー分析論文に基づくトレンド判定
    - cos/sin相関による高精度なトレンド・レンジ判定
    - 角度変化率による動的なトレンド強度計算
    - 瞬間周期とサイクル信頼度の統合分析
    
    動作:
    - State = 1: 上昇トレンド（角度が+90度より大きいか-90度より小さい）
    - State = 0: サイクリング/中立（瞬間周期≤60日かつ角度変化率>閾値）
    - State = -1: 下降トレンド（角度が-90度と+90度の間）
    
    使用方法:
    - トレンド系戦略のフィルター
    - エントリー条件の最適化
    - 市場状態に応じた戦略切り替え
    """
    
    def __init__(
        self,
        period: int = 20,                     # フェーザー分析の固定周期
        trend_threshold: float = 6.0,         # トレンド判定閾値（角度変化率）
        src_type: str = 'close',              # ソースタイプ
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,      # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,   # プロセスノイズ
        kalman_observation_noise: float = 0.001 # 観測ノイズ
    ):
        """
        コンストラクタ
        
        Args:
            period: フェーザー分析の固定周期（デフォルト: 20）
            trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 6.0度）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
        """
        # パラメータの設定
        params = {
            'period': period,
            'trend_threshold': trend_threshold,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise
        }
        
        super().__init__(
            f"PhasorTrendFilter(period={period}, threshold={trend_threshold:.1f}, kalman={'Y' if use_kalman_filter else 'N'})",
            params
        )
        
        # フェーザートレンドフィルターインジケーターの初期化
        self._filter = PhasorTrendFilter(
            period=period,
            trend_threshold=trend_threshold,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise
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
            シグナルの配列 (1: 上昇トレンド, 0: 中立/サイクリング, -1: 下降トレンド)
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
            
            # フェーザートレンドフィルターの計算
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
            print(f"PhasorTrendFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_trend_values(self) -> np.ndarray:
        """
        トレンド強度値を取得する
        
        Returns:
            トレンド強度値の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.values
        return np.array([])
    
    def get_phase_angle(self) -> np.ndarray:
        """
        フェーザー角度を取得する
        
        Returns:
            フェーザー角度の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.phase_angle
        return np.array([])
    
    def get_state_values(self) -> np.ndarray:
        """
        トレンド状態値を取得する
        
        Returns:
            トレンド状態値の配列 (1=上昇トレンド、0=中立、-1=下降トレンド)
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.state
        return np.array([])
    
    def get_trend_strength(self) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.trend_strength
        return np.array([])
    
    def get_cycle_confidence(self) -> np.ndarray:
        """
        サイクル信頼度を取得する
        
        Returns:
            サイクル信頼度の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.cycle_confidence
        return np.array([])
    
    def get_instantaneous_period(self) -> np.ndarray:
        """
        瞬間周期を取得する
        
        Returns:
            瞬間周期の配列
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.instantaneous_period
        return np.array([])
    
    def get_phasor_components(self) -> tuple:
        """
        フェーザーのRealとImaginaryコンポーネントを取得する
        
        Returns:
            (Real, Imaginary)コンポーネントのタプル
        """
        if hasattr(self._filter, '_result_cache') and self._filter._cache_keys:
            result = self._filter._result_cache[self._filter._cache_keys[-1]]
            return result.real_component, result.imag_component
        return np.array([]), np.array([])
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._filter, 'reset'):
            self._filter.reset()
        self._signals = None
        self._data_hash = None