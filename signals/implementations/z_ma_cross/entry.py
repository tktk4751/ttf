#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_ma import ZMA
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


@njit(fastmath=True, parallel=True)
def generate_cross_signals_numba(
    fast_ma: np.ndarray,
    slow_ma: np.ndarray
) -> np.ndarray:
    """
    Numbaによる高速なクロスシグナル生成

    Args:
        fast_ma: 短期ZMAの配列
        slow_ma: 長期ZMAの配列

    Returns:
        シグナル配列 (1: ゴールデンクロス, -1: デッドクロス, 0: ニュートラル)
    """
    length = len(fast_ma)
    signals = np.zeros(length, dtype=np.int64)
    
    # 最初の部分はシグナルなし（十分なデータがないため）
    min_idx = 1
    
    for i in prange(min_idx, length):
        # 有効なデータがあるか確認
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(fast_ma[i-1]) or np.isnan(slow_ma[i-1]):
            signals[i] = 0
            continue
        
        # ゴールデンクロス: 短期ZMAが長期ZMAを下から上に抜ける
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            signals[i] = 1
        # デッドクロス: 短期ZMAが長期ZMAを上から下に抜ける
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            signals[i] = -1
        # それ以外はシグナルなし
        else:
            signals[i] = 0
    
    return signals


class ZMACrossEntrySignal(BaseSignal, IEntrySignal):
    """
    ZMAのゴールデンクロス/デッドクロスによるエントリーシグナル
    
    特徴:
    - 効率比（CER）に基づいて動的に調整される2つのZMAを使用
    - 短期ZMAが長期ZMAを下から上に抜けた場合: ゴールデンクロス = ロングエントリー (1)
    - 短期ZMAが長期ZMAを上から下に抜けた場合: デッドクロス = ショートエントリー (-1)
    - Numbaによる最適化で高速処理
    - トレンドの強さに応じて適応的なクロス判定
    
    パラメータ:
    - 短期ZMAと長期ZMAのパラメータ（最大/最小期間用ドミナントサイクル設定）
    - サイクル効率比（CER）の計算パラメータ
    """
    
    def __init__(
        self,
        # ドミナントサイクル・効率比（CER）の基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # 短期ZMA用パラメータ
        fast_max_dc_cycle_part: float = 0.25,
        fast_max_dc_max_cycle: int = 55,
        fast_max_dc_min_cycle: int = 5,
        fast_max_dc_max_output: int = 34,
        fast_max_dc_min_output: int = 13,
        
        fast_min_dc_cycle_part: float = 0.25,
        fast_min_dc_max_cycle: int = 21,
        fast_min_dc_min_cycle: int = 5,
        fast_min_dc_max_output: int = 8,
        fast_min_dc_min_output: int = 3,
        
        fast_max_slow_period: int = 21,
        fast_min_slow_period: int = 8,
        fast_max_fast_period: int = 5,
        fast_min_fast_period: int = 2,
        fast_hyper_smooth_period: int = 0,
        
        # 長期ZMA用パラメータ
        slow_max_dc_cycle_part: float = 0.5,
        slow_max_dc_max_cycle: int = 144,
        slow_max_dc_min_cycle: int = 13,
        slow_max_dc_max_output: int = 89,
        slow_max_dc_min_output: int = 21,
        
        slow_min_dc_cycle_part: float = 0.25,
        slow_min_dc_max_cycle: int = 55,
        slow_min_dc_min_cycle: int = 5,
        slow_min_dc_max_output: int = 13,
        slow_min_dc_min_output: int = 5,
        
        slow_max_slow_period: int = 34,
        slow_min_slow_period: int = 13,
        slow_max_fast_period: int = 8,
        slow_min_fast_period: int = 3,
        slow_hyper_smooth_period: int = 0,
        
        # ソースタイプ
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: ドミナントサイクル検出器のタイプ
            lp_period: ローパスフィルターの期間
            hp_period: ハイパスフィルターの期間
            cycle_part: サイクル部分の割合
            
            fast_max_dc_*: 短期ZMAの最大期間用ドミナントサイクルパラメータ
            fast_min_dc_*: 短期ZMAの最小期間用ドミナントサイクルパラメータ
            fast_max_slow_period: 短期ZMAの最大スロー期間
            fast_min_slow_period: 短期ZMAの最小スロー期間
            fast_max_fast_period: 短期ZMAの最大ファスト期間
            fast_min_fast_period: 短期ZMAの最小ファスト期間
            fast_hyper_smooth_period: 短期ZMAのハイパースムーザー期間
            
            slow_max_dc_*: 長期ZMAの最大期間用ドミナントサイクルパラメータ
            slow_min_dc_*: 長期ZMAの最小期間用ドミナントサイクルパラメータ
            slow_max_slow_period: 長期ZMAの最大スロー期間
            slow_min_slow_period: 長期ZMAの最小スロー期間
            slow_max_fast_period: 長期ZMAの最大ファスト期間
            slow_min_fast_period: 長期ZMAの最小ファスト期間
            slow_hyper_smooth_period: 長期ZMAのハイパースムーザー期間
            
            src_type: 価格計算の元となる価格タイプ
        """
        super().__init__(f"ZMACross({fast_max_dc_max_output}-{slow_max_dc_max_output})")
        
        # パラメータの保存
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        # 短期ZMAパラメータ
        self.fast_max_dc_cycle_part = fast_max_dc_cycle_part
        self.fast_max_dc_max_cycle = fast_max_dc_max_cycle
        self.fast_max_dc_min_cycle = fast_max_dc_min_cycle
        self.fast_max_dc_max_output = fast_max_dc_max_output
        self.fast_max_dc_min_output = fast_max_dc_min_output
        
        self.fast_min_dc_cycle_part = fast_min_dc_cycle_part
        self.fast_min_dc_max_cycle = fast_min_dc_max_cycle
        self.fast_min_dc_min_cycle = fast_min_dc_min_cycle
        self.fast_min_dc_max_output = fast_min_dc_max_output
        self.fast_min_dc_min_output = fast_min_dc_min_output
        
        self.fast_max_slow_period = fast_max_slow_period
        self.fast_min_slow_period = fast_min_slow_period
        self.fast_max_fast_period = fast_max_fast_period
        self.fast_min_fast_period = fast_min_fast_period
        self.fast_hyper_smooth_period = fast_hyper_smooth_period
        
        # 長期ZMAパラメータ
        self.slow_max_dc_cycle_part = slow_max_dc_cycle_part
        self.slow_max_dc_max_cycle = slow_max_dc_max_cycle
        self.slow_max_dc_min_cycle = slow_max_dc_min_cycle
        self.slow_max_dc_max_output = slow_max_dc_max_output
        self.slow_max_dc_min_output = slow_max_dc_min_output
        
        self.slow_min_dc_cycle_part = slow_min_dc_cycle_part
        self.slow_min_dc_max_cycle = slow_min_dc_max_cycle
        self.slow_min_dc_min_cycle = slow_min_dc_min_cycle
        self.slow_min_dc_max_output = slow_min_dc_max_output
        self.slow_min_dc_min_output = slow_min_dc_min_output
        
        self.slow_max_slow_period = slow_max_slow_period
        self.slow_min_slow_period = slow_min_slow_period
        self.slow_max_fast_period = slow_max_fast_period
        self.slow_min_fast_period = slow_min_fast_period
        self.slow_hyper_smooth_period = slow_hyper_smooth_period
        
        self.src_type = src_type
        
        # インジケーターの初期化
        # 短期ZMA
        self.fast_zma = ZMA(
            max_dc_cycle_part=fast_max_dc_cycle_part,
            max_dc_max_cycle=fast_max_dc_max_cycle,
            max_dc_min_cycle=fast_max_dc_min_cycle,
            max_dc_max_output=fast_max_dc_max_output,
            max_dc_min_output=fast_max_dc_min_output,
            
            min_dc_cycle_part=fast_min_dc_cycle_part,
            min_dc_max_cycle=fast_min_dc_max_cycle,
            min_dc_min_cycle=fast_min_dc_min_cycle,
            min_dc_max_output=fast_min_dc_max_output,
            min_dc_min_output=fast_min_dc_min_output,
            
            max_slow_period=fast_max_slow_period,
            min_slow_period=fast_min_slow_period,
            max_fast_period=fast_max_fast_period,
            min_fast_period=fast_min_fast_period,
            hyper_smooth_period=fast_hyper_smooth_period,
            src_type=src_type
        )
        
        # 長期ZMA
        self.slow_zma = ZMA(
            max_dc_cycle_part=slow_max_dc_cycle_part,
            max_dc_max_cycle=slow_max_dc_max_cycle,
            max_dc_min_cycle=slow_max_dc_min_cycle,
            max_dc_max_output=slow_max_dc_max_output,
            max_dc_min_output=slow_max_dc_min_output,
            
            min_dc_cycle_part=slow_min_dc_cycle_part,
            min_dc_max_cycle=slow_min_dc_max_cycle,
            min_dc_min_cycle=slow_min_dc_min_cycle,
            min_dc_max_output=slow_min_dc_max_output,
            min_dc_min_output=slow_min_dc_min_output,
            
            max_slow_period=slow_max_slow_period,
            min_slow_period=slow_min_slow_period,
            max_fast_period=slow_max_fast_period,
            min_fast_period=slow_min_fast_period,
            hyper_smooth_period=slow_hyper_smooth_period,
            src_type=src_type
        )
        
        # サイクル効率比（CER）の初期化
        self.cer = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type
        )
        
        # 結果を保存する属性
        self._signals = None
        self._fast_zma_values = None
        self._slow_zma_values = None
        self._cer_values = None
        
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
        import hashlib
        
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合はnumpy配列に変換
            data_array = data.values
        else:
            data_array = data
            
        # データの形状とパラメータを含めたハッシュの生成
        param_str = (
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.fast_max_dc_cycle_part}_{self.fast_max_dc_max_cycle}_{self.fast_max_dc_min_cycle}_"
            f"{self.fast_max_dc_max_output}_{self.fast_max_dc_min_output}_"
            f"{self.fast_min_dc_cycle_part}_{self.fast_min_dc_max_cycle}_{self.fast_min_dc_min_cycle}_"
            f"{self.fast_min_dc_max_output}_{self.fast_min_dc_min_output}_"
            f"{self.fast_max_slow_period}_{self.fast_min_slow_period}_{self.fast_max_fast_period}_"
            f"{self.fast_min_fast_period}_{self.fast_hyper_smooth_period}_"
            f"{self.slow_max_dc_cycle_part}_{self.slow_max_dc_max_cycle}_{self.slow_max_dc_min_cycle}_"
            f"{self.slow_max_dc_max_output}_{self.slow_max_dc_min_output}_"
            f"{self.slow_min_dc_cycle_part}_{self.slow_min_dc_max_cycle}_{self.slow_min_dc_min_cycle}_"
            f"{self.slow_min_dc_max_output}_{self.slow_min_dc_min_output}_"
            f"{self.slow_max_slow_period}_{self.slow_min_slow_period}_{self.slow_max_fast_period}_"
            f"{self.slow_min_fast_period}_{self.slow_hyper_smooth_period}_{self.src_type}"
        )
        data_shape_str = f"{data_array.shape}_{data_array.dtype}"
        hash_str = f"{param_str}_{data_shape_str}"
        
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ZMAクロスシグナルを生成

        Args:
            data: 価格データ

        Returns:
            np.ndarray: シグナル配列 (1: ゴールデンクロス, -1: デッドクロス, 0: ニュートラル)
        """
        try:
            # データのハッシュ値を計算
            data_hash = self._get_data_hash(data)
            
            # 同じデータでキャッシュが存在する場合、キャッシュを返す
            if self._data_hash == data_hash and self._signals is not None:
                return self._signals
            
            # ハッシュを更新
            self._data_hash = data_hash
            
            # 効率比（CER）の計算
            cer_values = self.cer.calculate(data)
            
            # 短期ZMAと長期ZMAの計算
            fast_zma_values = self.fast_zma.calculate(data, external_er=cer_values)
            slow_zma_values = self.slow_zma.calculate(data, external_er=cer_values)
            
            # クロスシグナルの計算（Numba高速化版）
            signals = generate_cross_signals_numba(fast_zma_values, slow_zma_values)
            
            # 結果を保存
            self._signals = signals
            self._fast_zma_values = fast_zma_values
            self._slow_zma_values = slow_zma_values
            self._cer_values = cer_values
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZMAクロスシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はゼロシグナルを返す
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(data.shape[0])
    
    def get_zma_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZMA値を取得
        
        Returns:
            tuple: (短期ZMA, 長期ZMA)のタプル
        """
        if self._fast_zma_values is None or self._slow_zma_values is None:
            return np.array([]), np.array([])
        return self._fast_zma_values, self._slow_zma_values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比を取得
        
        Returns:
            np.ndarray: 効率比（CER）の値
        """
        return self._cer_values if self._cer_values is not None else np.array([])
    
    def get_fast_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        短期ZMAの動的な期間を取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        return self.fast_zma.get_dynamic_periods()
    
    def get_slow_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        長期ZMAの動的な期間を取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        return self.slow_zma.get_dynamic_periods()
    
    def get_signals(self) -> np.ndarray:
        """
        シグナル配列を取得
        
        Returns:
            np.ndarray: シグナル配列
        """
        return self._signals if self._signals is not None else np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        self.fast_zma.reset()
        self.slow_zma.reset()
        self.cer.reset()
        self._signals = None
        self._fast_zma_values = None
        self._slow_zma_values = None
        self._cer_values = None
        self._data_hash = None 