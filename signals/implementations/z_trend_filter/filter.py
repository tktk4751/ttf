#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.z_trend_filter import ZTrendFilter


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    filter_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        filter_values: Zトレンドフィルター値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(filter_values)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in prange(length):
        if np.isnan(filter_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif filter_values[i] < threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class ZTrendFilterSignal(BaseSignal, IFilterSignal):
    """
    Zトレンドフィルターを使用したフィルターシグナル
    
    特徴:
    - サイクル効率比（CER）とZトレンドインデックスを組み合わせて市場状態を判定
    - 動的しきい値でより正確な市場状態の検出が可能
    - トレンド相場とレンジ相場を高精度に識別
    - 3種類の組み合わせ方法によるシグナル最適化
    
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
        # ZTrendIndexのパラメータ

        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        # RMSウィンドウのパラメータ
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        # しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        # 組み合わせパラメータ
        combination_weight: float = 0.6,
        zadx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5,
        smoother_type: str = 'alma'  # 'alma'または'hyper'
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
            zadx_weight: ZADXの重み（デフォルト: 0.4）
            combination_method: 組み合わせ方法
                "sigmoid": シグモイド強調（デフォルト）
                "rms": 二乗平均平方根
                "simple": 単純平均
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル設定
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル設定
            smoother_type: 平滑化アルゴリズム（'alma'または'hyper'）
        """
        # パラメータの設定
        params = {

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
            'zadx_weight': zadx_weight,
            'combination_method': combination_method,
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output,
            'smoother_type': smoother_type
        }
        
        super().__init__(
            f"ZTrendFilter( {combination_method})",
            params
        )
        
        # Zトレンドフィルターインジケーターの初期化
        self._filter = ZTrendFilter(
            # ZTrendIndexのパラメータ
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            # RMSウィンドウのパラメータ
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            # しきい値のパラメータ
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            # 組み合わせパラメータ
            combination_weight=combination_weight,
            zadx_weight=zadx_weight,
            combination_method=combination_method,
            # Zトレンドインデックスの追加パラメータ
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output,
            smoother_type=smoother_type
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
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
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
            
            # Zトレンドフィルターの計算
            filter_result = self._filter.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if filter_result is None:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            filter_values = filter_result.values
            
            # 動的しきい値の取得
            threshold_values = filter_result.dynamic_threshold
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(filter_values, threshold_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZTrendFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        Zトレンドフィルター値を取得する
        
        Returns:
            Zトレンドフィルター値の配列
        """
        if hasattr(self._filter, '_values') and self._filter._values is not None:
            return self._filter._values
        return np.array([])
    
    def get_threshold_values(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            動的しきい値の配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.dynamic_threshold
        return np.array([])
    
    def get_trend_index(self) -> np.ndarray:
        """
        トレンドインデックス値を取得する
        
        Returns:
            トレンドインデックスの配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.trend_index
        return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（CER）を取得する
        
        Returns:
            サイクル効率比の配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.er
        return np.array([])
    
    def get_rms_window(self) -> np.ndarray:
        """
        RMSウィンドウサイズを取得する
        
        Returns:
            RMSウィンドウサイズの配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.rms_window
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