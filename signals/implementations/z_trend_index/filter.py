#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.z_trend_index import ZTrendIndex


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    index_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        index_values: Zトレンドインデックス値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(index_values)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in prange(length):
        if np.isnan(index_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif index_values[i] < threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class ZTrendIndexSignal(BaseSignal, IFilterSignal):
    """
    Zトレンドインデックスを使用したフィルターシグナル
    
    特徴:
    - サイクル効率比（CER）とドミナントサイクル検出を組み合わせて市場状態を判定
    - 動的しきい値でより正確な市場状態の検出が可能
    - トレンド相場とレンジ相場を高精度に識別
    
    動作:
    - インデックス値が動的しきい値以上：トレンド相場 (1)
    - インデックス値が動的しきい値未満：レンジ相場 (-1)
    
    使用方法:
    - トレンド系/レンジ系ストラテジーの自動切り替え
    - エントリー条件の最適化
    - リスク管理の調整
    """
    
    def __init__(
        self,
        # 最大チョピネス期間用ドミナントサイクル設定
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        
        # 最小チョピネス期間用ドミナントサイクル設定
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5,
        
        # 標準偏差と標準偏差ルックバック期間の設定
        max_stddev_period: int = 21,
        min_stddev_period: int = 14,
        max_lookback_period: int = 14,
        min_lookback_period: int = 7,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # ZATR用パラメータ
        smoother_type: str = 'alma',  # 'alma'または'hyper'
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 10）
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 34）
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 13）
            
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 21）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 14）
            max_lookback_period: 標準偏差の最小値を探す期間の最大値（デフォルト: 14）
            min_lookback_period: 標準偏差の最小値を探す期間の最小値（デフォルト: 7）
            
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
            max_threshold: しきい値の最大値（デフォルト: 0.75）
            min_threshold: しきい値の最小値（デフォルト: 0.55）
        """
        # パラメータの設定
        params = {
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
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        super().__init__(
            f"ZTrendIndex({cycle_detector_type}, {max_chop_dc_max_output}, {min_chop_dc_max_output})",
            params
        )
        
        # Zトレンドインデックスインジケーターの初期化
        self._index = ZTrendIndex(
            # 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            
            # 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output,
            
            # 標準偏差と標準偏差ルックバック期間の設定
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            # ZATR用パラメータ
            smoother_type=smoother_type,
            
            # 動的しきい値のパラメータ
            max_threshold=max_threshold,
            min_threshold=min_threshold
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
            
            # Zトレンドインデックスの計算
            index_result = self._index.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if index_result is None:
                self._signals = np.full(len(data), np.nan)
                return self._signals
            
            # インデックス値と動的しきい値の取得
            index_values = index_result.values
            threshold_values = index_result.dynamic_threshold
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(index_values, threshold_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZTrendIndexSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        Zトレンドインデックス値を取得する
        
        Returns:
            Zトレンドインデックス値の配列
        """
        if hasattr(self._index, '_values') and self._index._values is not None:
            return self._index._values
        return np.array([])
    
    def get_threshold_values(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            動的しきい値の配列
        """
        if hasattr(self._index, '_result') and self._index._result is not None:
            return self._index._result.dynamic_threshold
        return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（CER）を取得する
        
        Returns:
            サイクル効率比の配列
        """
        if hasattr(self._index, '_result') and self._index._result is not None:
            return self._index._result.er
        return np.array([])
    
    def get_choppiness_index(self) -> np.ndarray:
        """
        チョピネス指数を取得する
        
        Returns:
            チョピネス指数の配列
        """
        if hasattr(self._index, '_result') and self._index._result is not None:
            return self._index._result.choppiness_index
        return np.array([])
    
    def get_trend_state(self) -> np.ndarray:
        """
        トレンド状態を取得する
        
        Returns:
            トレンド状態の配列 (1=トレンド、0=レンジ、NaN=不明)
        """
        if hasattr(self._index, '_result') and self._index._result is not None:
            return self._index._result.trend_state
        return np.array([])
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._index, 'reset'):
            self._index.reset()
        self._signals = None
        self._data_hash = None 