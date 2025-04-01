#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.alpha_trend_filter import AlphaTrendFilter


@jit(nopython=True)
def generate_signals_numba(
    filter_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        filter_values: アルファトレンドフィルター値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(filter_values)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in range(length):
        if np.isnan(filter_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif filter_values[i] < threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class AlphaTrendFilterSignal(BaseSignal, IFilterSignal):
    """
    アルファトレンドフィルターを使用したフィルターシグナル
    
    特徴:
    - サイクル効率比（CER）とアルファトレンドインデックスを組み合わせて市場状態を判定
    - 動的しきい値でより正確な市場状態の検出が可能
    - トレンド相場とレンジ相場を高精度に識別
    
    動作:
    - フィルター値が動的しきい値以上：トレンド相場 (1)
    - フィルター値が動的しきい値未満：レンジ相場 (-1)
    
    使用方法:
    - トレンド系/レンジ系ストラテジーの自動切り替え
    - エントリー条件の最適化
    - リスク管理の調整
    """
    
    def __init__(
        self,
        max_chop_period: int = 89,
        min_chop_period: int = 21,
        max_atr_period: int = 89,
        min_atr_period: int = 21,
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        # 組み合わせパラメータ
        combination_weight: float = 0.6,
        combination_method: str = "sigmoid"  # "sigmoid", "rms", "simple"
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_period: チョピネス期間の最大値（デフォルト: 89）
            min_chop_period: チョピネス期間の最小値（デフォルト: 21）
            max_atr_period: ATR期間の最大値（デフォルト: 89）
            min_atr_period: ATR期間の最小値（デフォルト: 21）
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 13）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 5）
            max_lookback_period: 最小標準偏差を探す最大ルックバック期間（デフォルト: 13）
            min_lookback_period: 最小標準偏差を探す最小ルックバック期間（デフォルト: 5）
            max_rms_window: RMS計算の最大ウィンドウサイズ（デフォルト: 13）
            min_rms_window: RMS計算の最小ウィンドウサイズ（デフォルト: 5）
            max_threshold: しきい値の最大値（デフォルト: 0.75）
            min_threshold: しきい値の最小値（デフォルト: 0.55）
            cycle_detector_type: ドミナントサイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 62）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            combination_weight: トレンドインデックスとCERの組み合わせ重み（デフォルト: 0.6）
            combination_method: 組み合わせ方法
                "sigmoid": シグモイド強調（デフォルト）
                "rms": 二乗平均平方根
                "simple": 単純平均
        """
        # パラメータの設定
        params = {
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'combination_weight': combination_weight,
            'combination_method': combination_method
        }
        
        super().__init__(
            f"AlphaTrendFilter({max_chop_period}, {min_chop_period}, {combination_method})",
            params
        )
        
        # アルファトレンドフィルターインジケーターの初期化
        self._filter = AlphaTrendFilter(
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            combination_weight=combination_weight,
            combination_method=combination_method
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
        else:
            if data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            open_prices = data[:, 0]
            high_prices = data[:, 1]
            low_prices = data[:, 2]
            close_prices = data[:, 3]
        
        # アルファトレンドフィルターの計算
        filter_result = self._filter.calculate(open_prices, high_prices, low_prices, close_prices)
        filter_values = filter_result.values
        
        # 動的しきい値の取得
        threshold_values = self._filter.get_dynamic_threshold()
        
        # シグナルの生成（高速化版）
        signals = generate_signals_numba(filter_values, threshold_values)
        
        return signals
    
    def get_filter_values(self) -> np.ndarray:
        """
        アルファトレンドフィルター値を取得する
        
        Returns:
            アルファトレンドフィルター値の配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.values
        return np.array([])
    
    def get_threshold_values(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            動的しきい値の配列
        """
        return self._filter.get_dynamic_threshold()
    
    def get_trend_index(self) -> np.ndarray:
        """
        トレンドインデックス値を取得する
        
        Returns:
            トレンドインデックスの配列
        """
        return self._filter.get_trend_index()
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（CER）を取得する
        
        Returns:
            サイクル効率比の配列
        """
        return self._filter.get_efficiency_ratio()
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._filter, 'reset'):
            self._filter.reset() 