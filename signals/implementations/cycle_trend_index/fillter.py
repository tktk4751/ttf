#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.cycle_trend_index import CycleTrendIndex


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    filter_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        filter_values: サイクルトレンドインデックス値の配列
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


class CycleTrendIndexFilterSignal(BaseSignal, IFilterSignal):
    """
    サイクルトレンドインデックスを使用したフィルターシグナル
    
    特徴:
    - サイクル効率比（CER）とサイクルチョピネスを組み合わせて市場状態を判定
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
        # 共通パラメータ
        detector_type: str = 'phac_e',
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        lp_period: int = 5,
        hp_period: int = 144,
        
        # CER固有パラメータ
        er_detector_type: Optional[str] = None,
        er_cycle_part: Optional[float] = None,
        er_max_cycle: Optional[int] = None,
        er_min_cycle: Optional[int] = None,
        er_max_output: Optional[int] = None,
        er_min_output: Optional[int] = None,
        er_src_type: Optional[str] = None,
        er_lp_period: Optional[int] = None,
        er_hp_period: Optional[int] = None,
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        smooth_er: bool = True,
        er_alma_period: int = 3,
        er_alma_offset: float = 0.85,
        er_alma_sigma: float = 6,
        self_adaptive: bool = False,
        
        # チョピネス固有パラメータ
        chop_detector_type: Optional[str] = None,
        chop_cycle_part: Optional[float] = None,
        chop_max_cycle: Optional[int] = None,
        chop_min_cycle: Optional[int] = None,
        chop_max_output: Optional[int] = None, 
        chop_min_output: Optional[int] = None,
        chop_src_type: Optional[str] = None,
        chop_lp_period: Optional[int] = None,
        chop_hp_period: Optional[int] = None,
        smooth_chop: bool = True,
        chop_alma_period: int = 3,
        chop_alma_offset: float = 0.85,
        chop_alma_sigma: float = 6,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.45
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 共通のドミナントサイクル検出器タイプ
            cycle_part: 共通のサイクル部分
            max_cycle: 共通の最大サイクル期間
            min_cycle: 共通の最小サイクル期間
            max_output: 共通の最大出力値
            min_output: 共通の最小出力値
            src_type: 共通の価格ソース
            lp_period: 共通のローパスフィルター期間
            hp_period: 共通のハイパスフィルター期間
            
            er_detector_type: CER固有のドミナントサイクル検出器タイプ
            er_cycle_part: CER固有のサイクル部分
            er_max_cycle: CER固有の最大サイクル期間
            er_min_cycle: CER固有の最小サイクル期間
            er_max_output: CER固有の最大出力値
            er_min_output: CER固有の最小出力値
            er_src_type: CER固有の価格ソース
            er_lp_period: CER固有のローパスフィルター期間
            er_hp_period: CER固有のハイパスフィルター期間
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            smooth_er: 効率比にALMAスムージングを適用するかどうか
            er_alma_period: ALMAスムージングの期間
            er_alma_offset: ALMAスムージングのオフセット
            er_alma_sigma: ALMAスムージングのシグマ
            self_adaptive: セルフアダプティブモードを有効にするかどうか
            
            chop_detector_type: チョピネス固有のドミナントサイクル検出器タイプ
            chop_cycle_part: チョピネス固有のサイクル部分
            chop_max_cycle: チョピネス固有の最大サイクル期間 
            chop_min_cycle: チョピネス固有の最小サイクル期間
            chop_max_output: チョピネス固有の最大出力値
            chop_min_output: チョピネス固有の最小出力値
            chop_src_type: チョピネス固有の価格ソース
            chop_lp_period: チョピネス固有のローパスフィルター期間
            chop_hp_period: チョピネス固有のハイパスフィルター期間
            smooth_chop: チョピネス値にALMAスムージングを適用するかどうか
            chop_alma_period: ALMAスムージングの期間
            chop_alma_offset: ALMAスムージングのオフセット
            chop_alma_sigma: ALMAスムージングのシグマ
            
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
        """
        # パラメータの設定
        params = {
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'src_type': src_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'er_detector_type': er_detector_type,
            'er_cycle_part': er_cycle_part,
            'er_max_cycle': er_max_cycle,
            'er_min_cycle': er_min_cycle,
            'er_max_output': er_max_output,
            'er_min_output': er_min_output,
            'er_src_type': er_src_type,
            'er_lp_period': er_lp_period,
            'er_hp_period': er_hp_period,
            'use_kalman_filter': use_kalman_filter,
            'kalman_measurement_noise': kalman_measurement_noise,
            'kalman_process_noise': kalman_process_noise,
            'kalman_n_states': kalman_n_states,
            'smooth_er': smooth_er,
            'er_alma_period': er_alma_period,
            'er_alma_offset': er_alma_offset,
            'er_alma_sigma': er_alma_sigma,
            'self_adaptive': self_adaptive,
            'chop_detector_type': chop_detector_type,
            'chop_cycle_part': chop_cycle_part,
            'chop_max_cycle': chop_max_cycle,
            'chop_min_cycle': chop_min_cycle,
            'chop_max_output': chop_max_output,
            'chop_min_output': chop_min_output,
            'chop_src_type': chop_src_type,
            'chop_lp_period': chop_lp_period,
            'chop_hp_period': chop_hp_period,
            'smooth_chop': smooth_chop,
            'chop_alma_period': chop_alma_period,
            'chop_alma_offset': chop_alma_offset,
            'chop_alma_sigma': chop_alma_sigma,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        super().__init__(
            f"CycleTrendIndexFilter(det={detector_type}, er_smooth={'Y' if smooth_er else 'N'}, chop_smooth={'Y' if smooth_chop else 'N'})",
            params
        )
        
        # サイクルトレンドインデックスインジケーターの初期化
        self._filter = CycleTrendIndex(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period,
            er_detector_type=er_detector_type,
            er_cycle_part=er_cycle_part,
            er_max_cycle=er_max_cycle,
            er_min_cycle=er_min_cycle,
            er_max_output=er_max_output,
            er_min_output=er_min_output,
            er_src_type=er_src_type,
            er_lp_period=er_lp_period,
            er_hp_period=er_hp_period,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_process_noise=kalman_process_noise,
            kalman_n_states=kalman_n_states,
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma,
            self_adaptive=self_adaptive,
            chop_detector_type=chop_detector_type,
            chop_cycle_part=chop_cycle_part,
            chop_max_cycle=chop_max_cycle,
            chop_min_cycle=chop_min_cycle,
            chop_max_output=chop_max_output,
            chop_min_output=chop_min_output,
            chop_src_type=chop_src_type,
            chop_lp_period=chop_lp_period,
            chop_hp_period=chop_hp_period,
            smooth_chop=smooth_chop,
            chop_alma_period=chop_alma_period,
            chop_alma_offset=chop_alma_offset,
            chop_alma_sigma=chop_alma_sigma,
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
            elif data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # サイクルトレンドインデックスの計算
            filter_result = self._filter.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if filter_result is None:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            # サイクルトレンドインデックス値と動的しきい値の取得
            filter_values = filter_result.values
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
            print(f"CycleTrendIndexFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        サイクルトレンドインデックス値を取得する
        
        Returns:
            サイクルトレンドインデックス値の配列
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
    
    def get_er_values(self) -> np.ndarray:
        """
        サイクル効率比（CER）値を取得する
        
        Returns:
            サイクル効率比の配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.er_values
        return np.array([])
    
    def get_chop_values(self) -> np.ndarray:
        """
        サイクルチョピネス値を取得する
        
        Returns:
            サイクルチョピネスの配列
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.chop_values
        return np.array([])
    
    def get_trend_state(self) -> np.ndarray:
        """
        トレンド状態を取得する
        
        Returns:
            トレンド状態の配列 (1=トレンド、0=レンジ、NaN=不明)
        """
        if hasattr(self._filter, '_result') and self._filter._result is not None:
            return self._filter._result.trend_state
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
