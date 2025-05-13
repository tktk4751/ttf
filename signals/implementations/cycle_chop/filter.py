#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.cycle_chop import CycleChoppiness


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    chop_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        chop_values: サイクルチョピネス値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: 高チョピネス/レンジ相場, -1: 低チョピネス/トレンド相場)
    """
    length = len(chop_values)
    signals = np.ones(length) * -1  # デフォルトは低チョピネス/トレンド相場 (-1)
    
    for i in prange(length):
        if np.isnan(chop_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif chop_values[i] >= threshold_values[i]:
            signals[i] = 1  # 高チョピネス/レンジ相場
    
    return signals


class CycleChoppinessFilterSignal(BaseSignal, IFilterSignal):
    """
    サイクルチョピネスを使用したフィルターシグナル
    
    特徴:
    - ドミナントサイクルを使用して動的なウィンドウサイズでのチョピネスインデックスを計算
    - サイクル効率比（CER）に基づく動的しきい値でより正確な市場状態の検出が可能
    - トレンド相場とレンジ相場を高精度に識別
    
    動作:
    - チョピネス値が動的しきい値以上：レンジ相場 (1)
    - チョピネス値が動的しきい値未満：トレンド相場 (-1)
    
    使用方法:
    - トレンド系/レンジ系ストラテジーの自動切り替え
    - エントリー条件の最適化
    - リスク管理の調整
    """
    
    def __init__(
        self,
        detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        smooth_chop: bool = True,
        chop_alma_period: int = 3,
        chop_alma_offset: float = 0.85,
        chop_alma_sigma: float = 6,
        max_threshold: float = 0.6,  # 動的しきい値の最大値
        min_threshold: float = 0.4,   # 動的しきい値の最小値
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
        self_adaptive: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
            lp_period: ドミナントサイクル用ローパスフィルター期間
            hp_period: ドミナントサイクル用ハイパスフィルター期間
            cycle_part: ドミナントサイクル計算用サイクル部分
            max_cycle: ドミナントサイクル最大期間
            min_cycle: ドミナントサイクル最小期間
            max_output: ドミナントサイクル最大出力値
            min_output: ドミナントサイクル最小出力値
            src_type: 価格ソース ('close', 'hlc3', etc.)
            smooth_chop: チョピネス値にALMAスムージングを適用するかどうか
            chop_alma_period: ALMAスムージングの期間
            chop_alma_offset: ALMAスムージングのオフセット
            chop_alma_sigma: ALMAスムージングのシグマ
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
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
        """
        # パラメータの設定
        params = {
            'detector_type': detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'src_type': src_type,
            'smooth_chop': smooth_chop,
            'chop_alma_period': chop_alma_period,
            'chop_alma_offset': chop_alma_offset,
            'chop_alma_sigma': chop_alma_sigma,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
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
            'self_adaptive': self_adaptive
        }
        
        super().__init__(
            f"CycleChopFilter(det={detector_type}, part={cycle_part}, smooth={'Y' if smooth_chop else 'N'})",
            params
        )
        
        # サイクルチョピネスインジケーターの初期化
        self._filter = CycleChoppiness(
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            smooth_chop=smooth_chop,
            chop_alma_period=chop_alma_period,
            chop_alma_offset=chop_alma_offset,
            chop_alma_sigma=chop_alma_sigma,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
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
            self_adaptive=self_adaptive
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
            シグナルの配列 (1: 高チョピネス/レンジ相場, -1: 低チョピネス/トレンド相場)
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
            
            # サイクルチョピネスの計算
            self._filter.calculate(data)
            result = self._filter.get_result()
            
            # 計算が失敗した場合はNaNシグナルを返す
            if result is None:
                self._signals = np.full(len(data), np.nan)
                return self._signals
                
            # サイクルチョピネス値と動的しきい値の取得
            chop_values = result.values
            threshold_values = result.dynamic_threshold
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(chop_values, threshold_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"CycleChoppinessFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(len(data), np.nan)
    
    def get_filter_values(self) -> np.ndarray:
        """
        サイクルチョピネス値を取得する
        
        Returns:
            サイクルチョピネス値の配列
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
        if self._filter.get_result() is not None:
            return self._filter.get_result().dynamic_threshold
        return np.array([])
    
    def get_chop_state(self) -> np.ndarray:
        """
        チョピネス状態を取得する
        
        Returns:
            チョピネス状態の配列 (1=高チョピネス、0=低チョピネス、NaN=不明)
        """
        if self._filter.get_result() is not None:
            return self._filter.get_result().chop_state
        return np.array([])
    
    def get_cycle_periods(self) -> np.ndarray:
        """
        計算に使用された動的サイクル期間を取得する
        
        Returns:
            サイクル期間の配列
        """
        if self._filter.get_result() is not None:
            return self._filter.get_result().cycle_periods
        return np.array([])
    
    def get_er_values(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            サイクル効率比の値の配列
        """
        if self._filter.get_result() is not None:
            return self._filter.get_result().er_values
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