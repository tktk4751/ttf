#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import hashlib
from numba import njit, prange, float64, int64, boolean

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.z_long_reversal_index import ZLongReversalIndex
from indicators.z_short_reversal_index import ZShortReversalIndex


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    long_reversal_values: np.ndarray,
    short_reversal_values: np.ndarray,
    long_threshold_values: np.ndarray,
    short_threshold_values: np.ndarray
) -> np.ndarray:
    """
    Numbaによる高速なシグナル生成

    Args:
        long_reversal_values: Zロングリバーサルインデックスの値
        short_reversal_values: Zショートリバーサルインデックスの値
        long_threshold_values: ロングリバーサルの動的しきい値
        short_threshold_values: ショートリバーサルの動的しきい値

    Returns:
        シグナル配列 (1: ロング, -1: ショート, 0: ニュートラル)
    """
    length = len(long_reversal_values)
    signals = np.zeros(length, dtype=np.int64)

    for i in prange(length):
        if np.isnan(long_reversal_values[i]) or np.isnan(short_reversal_values[i]):
            signals[i] = 0
        # ロングリバーサルが動的しきい値よりも高い場合、ショートシグナル
        elif long_reversal_values[i] > long_threshold_values[i]:
            signals[i] = -1
        # ショートリバーサルが動的しきい値よりも低い場合、ロングシグナル
        elif short_reversal_values[i] < short_threshold_values[i]:
            signals[i] = 1
        else:
            signals[i] = 0

    return signals


class ZReversalFilterSignal(BaseSignal, IFilterSignal):
    """
    Zリバーサルフィルターシグナル

    Zロングリバーサルインデックスが動的しきい値を上回った場合にロングフィルター、
    Zショートリバーサルインデックスが動的しきい値を下回った場合にショートフィルターを生成します。
    
    リバーサル（反転）シグナルを効率比に基づいて動的に調整し、トレンドの強さに応じて
    フィルタリングの厳しさを自動調整します。
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるしきい値
    - トレンドが強い時はリバーサルシグナルが発生しやすくなる
    - レンジ相場時はリバーサルシグナルが発生しにくくなる
    - ZADXとZRSXを組み合わせた高精度な反転検出
    """

    def __init__(
        self,
        # Zロングリバーサルインデックスのパラメータ
        long_max_rms_window: int = 13,
        long_min_rms_window: int = 5,
        long_max_threshold: float = 0.9,
        long_min_threshold: float = 0.75,
        
        # Zショートリバーサルインデックスのパラメータ
        short_max_rms_window: int = 13,
        short_min_rms_window: int = 5,
        short_max_threshold: float = 0.25,
        short_min_threshold: float = 0.1,
        
        # 共通パラメータ
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        
        # 組み合わせパラメータ
        zadx_weight: float = 0.4,
        zrsx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        
        # ZADX用パラメータ
        zadx_max_dc_cycle_part: float = 0.5,
        zadx_max_dc_max_cycle: int = 34,
        zadx_max_dc_min_cycle: int = 5,
        zadx_max_dc_max_output: int = 21,
        zadx_max_dc_min_output: int = 8,
        zadx_min_dc_cycle_part: float = 0.25,
        zadx_min_dc_max_cycle: int = 21,
        zadx_min_dc_min_cycle: int = 3,
        zadx_min_dc_max_output: int = 13,
        zadx_min_dc_min_output: int = 3,
        zadx_er_period: int = 21,
        
        # ZRSX用パラメータ
        zrsx_max_dc_cycle_part: float = 0.5,
        zrsx_max_dc_max_cycle: int = 55,
        zrsx_max_dc_min_cycle: int = 5,
        zrsx_max_dc_max_output: int = 21,
        zrsx_max_dc_min_output: int = 10,
        zrsx_min_dc_cycle_part: float = 0.25,
        zrsx_min_dc_max_cycle: int = 34,
        zrsx_min_dc_min_cycle: int = 3,
        zrsx_min_dc_max_output: int = 10,
        zrsx_min_dc_min_output: int = 5,
        zrsx_er_period: int = 10,
        
        smoother_type: str = 'alma'  # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            long_max_rms_window: ロングリバーサル用RMSの最大ウィンドウサイズ
            long_min_rms_window: ロングリバーサル用RMSの最小ウィンドウサイズ
            long_max_threshold: ロングリバーサル用しきい値の最大値
            long_min_threshold: ロングリバーサル用しきい値の最小値
            
            short_max_rms_window: ショートリバーサル用RMSの最大ウィンドウサイズ
            short_min_rms_window: ショートリバーサル用RMSの最小ウィンドウサイズ
            short_max_threshold: ショートリバーサル用しきい値の最大値
            short_min_threshold: ショートリバーサル用しきい値の最小値
            
            cycle_detector_type: ドミナントサイクル検出アルゴリズム
            lp_period: 効率比計算用ローパスフィルター期間
            hp_period: 効率比計算用ハイパスフィルター期間
            cycle_part: ドミナントサイクルの一部として使用する割合
            
            zadx_weight: ZADX重み付け
            zrsx_weight: ZRSX重み付け
            combination_method: 組み合わせ方法("sigmoid", "rms", "simple")
            
            zadx_max_dc_cycle_part: ZADX用最大ドミナントサイクル一部として使用する割合
            zadx_max_dc_max_cycle: ZADX用最大ドミナントサイクル最大値
            zadx_max_dc_min_cycle: ZADX用最大ドミナントサイクル最小値
            zadx_max_dc_max_output: ZADX用最大出力期間の最大値
            zadx_max_dc_min_output: ZADX用最大出力期間の最小値
            zadx_min_dc_cycle_part: ZADX用最小ドミナントサイクル一部として使用する割合
            zadx_min_dc_max_cycle: ZADX用最小ドミナントサイクル最大値
            zadx_min_dc_min_cycle: ZADX用最小ドミナントサイクル最小値
            zadx_min_dc_max_output: ZADX用最小出力期間の最大値
            zadx_min_dc_min_output: ZADX用最小出力期間の最小値
            zadx_er_period: ZADX用効率比計算期間
            
            zrsx_max_dc_cycle_part: ZRSX用最大ドミナントサイクル一部として使用する割合
            zrsx_max_dc_max_cycle: ZRSX用最大ドミナントサイクル最大値
            zrsx_max_dc_min_cycle: ZRSX用最大ドミナントサイクル最小値
            zrsx_max_dc_max_output: ZRSX用最大出力期間の最大値
            zrsx_max_dc_min_output: ZRSX用最大出力期間の最小値
            zrsx_min_dc_cycle_part: ZRSX用最小ドミナントサイクル一部として使用する割合
            zrsx_min_dc_max_cycle: ZRSX用最小ドミナントサイクル最大値
            zrsx_min_dc_min_cycle: ZRSX用最小ドミナントサイクル最小値
            zrsx_min_dc_max_output: ZRSX用最小出力期間の最大値
            zrsx_min_dc_min_output: ZRSX用最小出力期間の最小値
            zrsx_er_period: ZRSX用効率比計算期間
            
            smoother_type: スムーサータイプ('alma'または'hyper')
        """
        super().__init__("ZReversalFilter")
        
        # パラメータの保存
        # ロングリバーサル用パラメータ
        self.long_max_rms_window = long_max_rms_window
        self.long_min_rms_window = long_min_rms_window
        self.long_max_threshold = long_max_threshold
        self.long_min_threshold = long_min_threshold
        
        # ショートリバーサル用パラメータ
        self.short_max_rms_window = short_max_rms_window
        self.short_min_rms_window = short_min_rms_window
        self.short_max_threshold = short_max_threshold
        self.short_min_threshold = short_min_threshold
        
        # 共通パラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        self.zadx_weight = zadx_weight
        self.zrsx_weight = zrsx_weight
        self.combination_method = combination_method
        
        # ZADX用パラメータ
        self.zadx_max_dc_cycle_part = zadx_max_dc_cycle_part
        self.zadx_max_dc_max_cycle = zadx_max_dc_max_cycle
        self.zadx_max_dc_min_cycle = zadx_max_dc_min_cycle
        self.zadx_max_dc_max_output = zadx_max_dc_max_output
        self.zadx_max_dc_min_output = zadx_max_dc_min_output
        self.zadx_min_dc_cycle_part = zadx_min_dc_cycle_part
        self.zadx_min_dc_max_cycle = zadx_min_dc_max_cycle
        self.zadx_min_dc_min_cycle = zadx_min_dc_min_cycle
        self.zadx_min_dc_max_output = zadx_min_dc_max_output
        self.zadx_min_dc_min_output = zadx_min_dc_min_output
        self.zadx_er_period = zadx_er_period
        
        # ZRSX用パラメータ
        self.zrsx_max_dc_cycle_part = zrsx_max_dc_cycle_part
        self.zrsx_max_dc_max_cycle = zrsx_max_dc_max_cycle
        self.zrsx_max_dc_min_cycle = zrsx_max_dc_min_cycle
        self.zrsx_max_dc_max_output = zrsx_max_dc_max_output
        self.zrsx_max_dc_min_output = zrsx_max_dc_min_output
        self.zrsx_min_dc_cycle_part = zrsx_min_dc_cycle_part
        self.zrsx_min_dc_max_cycle = zrsx_min_dc_max_cycle
        self.zrsx_min_dc_min_cycle = zrsx_min_dc_min_cycle
        self.zrsx_min_dc_max_output = zrsx_min_dc_max_output
        self.zrsx_min_dc_min_output = zrsx_min_dc_min_output
        self.zrsx_er_period = zrsx_er_period
        
        self.smoother_type = smoother_type
        
        # インジケーターの初期化
        self.z_long_reversal = ZLongReversalIndex(
            max_rms_window=long_max_rms_window,
            min_rms_window=long_min_rms_window,
            max_threshold=long_max_threshold,
            min_threshold=long_min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            zadx_weight=zadx_weight,
            zrsx_weight=zrsx_weight,
            combination_method=combination_method,
            zadx_max_dc_cycle_part=zadx_max_dc_cycle_part,
            zadx_max_dc_max_cycle=zadx_max_dc_max_cycle,
            zadx_max_dc_min_cycle=zadx_max_dc_min_cycle,
            zadx_max_dc_max_output=zadx_max_dc_max_output,
            zadx_max_dc_min_output=zadx_max_dc_min_output,
            zadx_min_dc_cycle_part=zadx_min_dc_cycle_part,
            zadx_min_dc_max_cycle=zadx_min_dc_max_cycle,
            zadx_min_dc_min_cycle=zadx_min_dc_min_cycle,
            zadx_min_dc_max_output=zadx_min_dc_max_output,
            zadx_min_dc_min_output=zadx_min_dc_min_output,
            zadx_er_period=zadx_er_period,
            zrsx_max_dc_cycle_part=zrsx_max_dc_cycle_part,
            zrsx_max_dc_max_cycle=zrsx_max_dc_max_cycle,
            zrsx_max_dc_min_cycle=zrsx_max_dc_min_cycle,
            zrsx_max_dc_max_output=zrsx_max_dc_max_output,
            zrsx_max_dc_min_output=zrsx_max_dc_min_output,
            zrsx_min_dc_cycle_part=zrsx_min_dc_cycle_part,
            zrsx_min_dc_max_cycle=zrsx_min_dc_max_cycle,
            zrsx_min_dc_min_cycle=zrsx_min_dc_min_cycle,
            zrsx_min_dc_max_output=zrsx_min_dc_max_output,
            zrsx_min_dc_min_output=zrsx_min_dc_min_output,
            zrsx_er_period=zrsx_er_period,
            smoother_type=smoother_type
        )
        
        self.z_short_reversal = ZShortReversalIndex(
            max_rms_window=short_max_rms_window,
            min_rms_window=short_min_rms_window,
            max_threshold=short_max_threshold,
            min_threshold=short_min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            zadx_weight=zadx_weight,
            zrsx_weight=zrsx_weight,
            combination_method=combination_method,
            zadx_max_dc_cycle_part=zadx_max_dc_cycle_part,
            zadx_max_dc_max_cycle=zadx_max_dc_max_cycle,
            zadx_max_dc_min_cycle=zadx_max_dc_min_cycle,
            zadx_max_dc_max_output=zadx_max_dc_max_output,
            zadx_max_dc_min_output=zadx_max_dc_min_output,
            zadx_min_dc_cycle_part=zadx_min_dc_cycle_part,
            zadx_min_dc_max_cycle=zadx_min_dc_max_cycle,
            zadx_min_dc_min_cycle=zadx_min_dc_min_cycle,
            zadx_min_dc_max_output=zadx_min_dc_max_output,
            zadx_min_dc_min_output=zadx_min_dc_min_output,
            zadx_er_period=zadx_er_period,
            zrsx_max_dc_cycle_part=zrsx_max_dc_cycle_part,
            zrsx_max_dc_max_cycle=zrsx_max_dc_max_cycle,
            zrsx_max_dc_min_cycle=zrsx_max_dc_min_cycle,
            zrsx_max_dc_max_output=zrsx_max_dc_max_output,
            zrsx_max_dc_min_output=zrsx_max_dc_min_output,
            zrsx_min_dc_cycle_part=zrsx_min_dc_cycle_part,
            zrsx_min_dc_max_cycle=zrsx_min_dc_max_cycle,
            zrsx_min_dc_min_cycle=zrsx_min_dc_min_cycle,
            zrsx_min_dc_max_output=zrsx_min_dc_max_output,
            zrsx_min_dc_min_output=zrsx_min_dc_min_output,
            zrsx_er_period=zrsx_er_period,
            smoother_type=smoother_type
        )
        
        # 結果を保存する属性
        self._long_reversal_values = None
        self._short_reversal_values = None
        self._long_threshold_values = None
        self._short_threshold_values = None
        self._signals = None
        
        # キャッシュ用の属性
        self._data_hash = None
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算する

        Args:
            data: データ

        Returns:
            str: ハッシュ値
        """
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合はnumpy配列に変換
            data_array = data.values
        else:
            data_array = data
            
        # データの形状とパラメータを含めたハッシュの生成
        param_str = (
            f"{self.long_max_rms_window}_{self.long_min_rms_window}_{self.long_max_threshold}_{self.long_min_threshold}_"
            f"{self.short_max_rms_window}_{self.short_min_rms_window}_{self.short_max_threshold}_{self.short_min_threshold}_"
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.zadx_weight}_{self.zrsx_weight}_{self.combination_method}"
        )
        data_shape_str = f"{data_array.shape}_{data_array.dtype}"
        hash_str = f"{param_str}_{data_shape_str}"
        
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zリバーサルフィルターシグナルを生成

        Args:
            data: 価格データ

        Returns:
            np.ndarray: シグナル配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        try:
            # データのハッシュ値を計算
            data_hash = self._get_data_hash(data)
            
            # 同じデータでキャッシュが存在する場合、キャッシュを返す
            if self._data_hash == data_hash and self._signals is not None:
                return self._signals
            
            # ハッシュを更新
            self._data_hash = data_hash
            
            # Zロング/ショートリバーサルインデックスの計算
            long_result = self.z_long_reversal.calculate(data)
            short_result = self.z_short_reversal.calculate(data)
            
            # リバーサル値としきい値の取得
            long_reversal_values = long_result.values
            short_reversal_values = short_result.values
            long_threshold_values = long_result.dynamic_threshold
            short_threshold_values = short_result.dynamic_threshold
            
            # シグナルの生成
            signals = generate_signals_numba(
                long_reversal_values,
                short_reversal_values,
                long_threshold_values,
                short_threshold_values
            )
            
            # 結果を保存
            self._long_reversal_values = long_reversal_values
            self._short_reversal_values = short_reversal_values
            self._long_threshold_values = long_threshold_values
            self._short_threshold_values = short_threshold_values
            self._signals = signals
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"Zリバーサルフィルターシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はゼロシグナルを返す
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(data.shape[0])
    
    def get_long_reversal_values(self) -> np.ndarray:
        """Zロングリバーサルインデックスの値を取得"""
        return self._long_reversal_values if self._long_reversal_values is not None else np.array([])
    
    def get_short_reversal_values(self) -> np.ndarray:
        """Zショートリバーサルインデックスの値を取得"""
        return self._short_reversal_values if self._short_reversal_values is not None else np.array([])
    
    def get_long_threshold_values(self) -> np.ndarray:
        """Zロングリバーサルインデックスのしきい値を取得"""
        return self._long_threshold_values if self._long_threshold_values is not None else np.array([])
    
    def get_short_threshold_values(self) -> np.ndarray:
        """Zショートリバーサルインデックスのしきい値を取得"""
        return self._short_threshold_values if self._short_threshold_values is not None else np.array([])
        
    def get_signals(self) -> np.ndarray:
        """シグナル配列を取得"""
        return self._signals if self._signals is not None else np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """効率比を取得"""
        try:
            return self.z_long_reversal.get_efficiency_ratio()
        except:
            return np.array([])
    
    def reset(self) -> None:
        """状態をリセット"""
        self.z_long_reversal.reset()
        self.z_short_reversal.reset()
        self._long_reversal_values = None
        self._short_reversal_values = None
        self._long_threshold_values = None
        self._short_threshold_values = None
        self._signals = None
        self._data_hash = None 