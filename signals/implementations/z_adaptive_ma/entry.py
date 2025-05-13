#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_adaptive_ma import ZAdaptiveMA
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio

class ZAdaptiveMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    ZAdaptiveMACrossoverを使用したエントリーシグナル
    - 短期ZMA > 長期ZMA: ロングエントリー (1)
    - 短期ZMA < 長期ZMA: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 短期ZMAパラメータ
        short_fast_period: int = 2,
        short_slow_period: int = 30,
        # 長期ZMAパラメータ
        long_fast_period: int = 2,
        long_slow_period: int = 60,
        # 共通パラメータ
        src_type: str = 'hlc3',
        # 短期サイクルERパラメータ
        short_detector_type: str = 'dudi',
        short_lp_period: int = 5,
        short_hp_period: int = 89,
        short_cycle_part: float = 0.5,
        short_cycle_max: int = 89,
        short_cycle_min: int = 5,
        short_max_output: int = 55,
        short_min_output: int = 5,
        short_use_kalman_filter: bool = False,
        short_kalman_measurement_noise: float = 1.0,
        short_kalman_process_noise: float = 0.01,
        short_kalman_n_states: int = 5,
        short_smooth_er: bool = True,
        short_er_alma_period: int = 5,
        short_er_alma_offset: float = 0.85,
        short_er_alma_sigma: float = 6,
        short_self_adaptive: bool = False,
        # 長期サイクルERパラメータ
        long_detector_type: str = 'dudi',
        long_lp_period: int = 5,
        long_hp_period: int = 144,
        long_cycle_part: float = 0.5,
        long_cycle_max: int = 233,
        long_cycle_min: int = 5,
        long_max_output: int = 233,
        long_min_output: int = 89,
        long_use_kalman_filter: bool = False,
        long_kalman_measurement_noise: float = 1.0,
        long_kalman_process_noise: float = 0.01,
        long_kalman_n_states: int = 5,
        long_smooth_er: bool = True,
        long_er_alma_period: int = 5,
        long_er_alma_offset: float = 0.85,
        long_er_alma_sigma: float = 6,
        long_self_adaptive: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            # 短期ZMAパラメータ
            short_fast_period: 短期ZMAの速い移動平均期間
            short_slow_period: 短期ZMAの遅い移動平均期間
            # 長期ZMAパラメータ
            long_fast_period: 長期ZMAの速い移動平均期間
            long_slow_period: 長期ZMAの遅い移動平均期間
            # 共通パラメータ
            src_type: 価格ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            
            # 短期サイクルERパラメータ
            short_detector_type: 短期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            short_lp_period: 短期サイクル用ローパスフィルター期間
            short_hp_period: 短期サイクル用ハイパスフィルター期間
            short_cycle_part: 短期サイクル部分比率
            short_cycle_max: 短期サイクルの最大期間
            short_cycle_min: 短期サイクルの最小期間
            short_max_output: 短期サイクルの最大出力値
            short_min_output: 短期サイクルの最小出力値
            short_use_kalman_filter: 短期サイクルでカルマンフィルターを使用するか
            short_kalman_measurement_noise: 短期カルマンフィルターの測定ノイズ
            short_kalman_process_noise: 短期カルマンフィルターのプロセスノイズ
            short_kalman_n_states: 短期カルマンフィルターの状態数
            short_smooth_er: 短期効率比をスムージングするかどうか
            short_er_alma_period: 短期効率比スムージング用ALMAの期間
            short_er_alma_offset: 短期効率比スムージング用ALMAのオフセット
            short_er_alma_sigma: 短期効率比スムージング用ALMAのシグマ
            short_self_adaptive: 短期効率比をセルフアダプティブにするかどうか
            
            # 長期サイクルERパラメータ
            long_detector_type: 長期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            long_lp_period: 長期サイクル用ローパスフィルター期間
            long_hp_period: 長期サイクル用ハイパスフィルター期間
            long_cycle_part: 長期サイクル部分比率
            long_cycle_max: 長期サイクルの最大期間
            long_cycle_min: 長期サイクルの最小期間
            long_max_output: 長期サイクルの最大出力値
            long_min_output: 長期サイクルの最小出力値
            long_use_kalman_filter: 長期サイクルでカルマンフィルターを使用するか
            long_kalman_measurement_noise: 長期カルマンフィルターの測定ノイズ
            long_kalman_process_noise: 長期カルマンフィルターのプロセスノイズ
            long_kalman_n_states: 長期カルマンフィルターの状態数
            long_smooth_er: 長期効率比をスムージングするかどうか
            long_er_alma_period: 長期効率比スムージング用ALMAの期間
            long_er_alma_offset: 長期効率比スムージング用ALMAのオフセット
            long_er_alma_sigma: 長期効率比スムージング用ALMAのシグマ
            long_self_adaptive: 長期効率比をセルフアダプティブにするかどうか
        """
        params = {
            # 短期ZMAパラメータ
            'short_fast_period': short_fast_period,
            'short_slow_period': short_slow_period,
            # 長期ZMAパラメータ
            'long_fast_period': long_fast_period,
            'long_slow_period': long_slow_period,
            # 共通パラメータ
            'src_type': src_type,
            # 短期サイクルERパラメータ
            'short_detector_type': short_detector_type,
            'short_lp_period': short_lp_period,
            'short_hp_period': short_hp_period,
            'short_cycle_part': short_cycle_part,
            'short_cycle_max': short_cycle_max,
            'short_cycle_min': short_cycle_min,
            'short_max_output': short_max_output,
            'short_min_output': short_min_output,
            'short_use_kalman_filter': short_use_kalman_filter,
            'short_kalman_measurement_noise': short_kalman_measurement_noise,
            'short_kalman_process_noise': short_kalman_process_noise,
            'short_kalman_n_states': short_kalman_n_states,
            'short_smooth_er': short_smooth_er,
            'short_er_alma_period': short_er_alma_period,
            'short_er_alma_offset': short_er_alma_offset,
            'short_er_alma_sigma': short_er_alma_sigma,
            'short_self_adaptive': short_self_adaptive,
            # 長期サイクルERパラメータ
            'long_detector_type': long_detector_type,
            'long_lp_period': long_lp_period,
            'long_hp_period': long_hp_period,
            'long_cycle_part': long_cycle_part,
            'long_cycle_max': long_cycle_max,
            'long_cycle_min': long_cycle_min,
            'long_max_output': long_max_output,
            'long_min_output': long_min_output,
            'long_use_kalman_filter': long_use_kalman_filter,
            'long_kalman_measurement_noise': long_kalman_measurement_noise,
            'long_kalman_process_noise': long_kalman_process_noise,
            'long_kalman_n_states': long_kalman_n_states,
            'long_smooth_er': long_smooth_er,
            'long_er_alma_period': long_er_alma_period,
            'long_er_alma_offset': long_er_alma_offset,
            'long_er_alma_sigma': long_er_alma_sigma,
            'long_self_adaptive': long_self_adaptive
        }
        super().__init__(f"ZMACrossover({short_slow_period}, {long_slow_period})", params)
        
        # 短期サイクル効率比の初期化
        self._short_cycle_er = CycleEfficiencyRatio(
            detector_type=short_detector_type,
            lp_period=short_lp_period,
            hp_period=short_hp_period,
            cycle_part=short_cycle_part,
            max_cycle=short_cycle_max,
            min_cycle=short_cycle_min,
            max_output=short_max_output,
            min_output=short_min_output,
            src_type=src_type,
            use_kalman_filter=short_use_kalman_filter,
            kalman_measurement_noise=short_kalman_measurement_noise,
            kalman_process_noise=short_kalman_process_noise,
            kalman_n_states=short_kalman_n_states,
            smooth_er=short_smooth_er,
            er_alma_period=short_er_alma_period,
            er_alma_offset=short_er_alma_offset,
            er_alma_sigma=short_er_alma_sigma,
            self_adaptive=short_self_adaptive
        )
        
        # 長期サイクル効率比の初期化
        self._long_cycle_er = CycleEfficiencyRatio(
            detector_type=long_detector_type,
            lp_period=long_lp_period,
            hp_period=long_hp_period,
            cycle_part=long_cycle_part,
            max_cycle=long_cycle_max,
            min_cycle=long_cycle_min,
            max_output=long_max_output,
            min_output=long_min_output,
            src_type=src_type,
            use_kalman_filter=long_use_kalman_filter,
            kalman_measurement_noise=long_kalman_measurement_noise,
            kalman_process_noise=long_kalman_process_noise,
            kalman_n_states=long_kalman_n_states,
            smooth_er=long_smooth_er,
            er_alma_period=long_er_alma_period,
            er_alma_offset=long_er_alma_offset,
            er_alma_sigma=long_er_alma_sigma,
            self_adaptive=long_self_adaptive
        )
        
        # ZAdaptiveMAインジケーターの初期化
        self._short_zma = ZAdaptiveMA(
            fast_period=short_fast_period,
            slow_period=short_slow_period,
            src_type=src_type
        )
        
        self._long_zma = ZAdaptiveMA(
            fast_period=long_fast_period,
            slow_period=long_slow_period,
            src_type=src_type
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # サイクル効率比の計算
        short_er = self._short_cycle_er.calculate(data)
        long_er = self._long_cycle_er.calculate(data)
        
        # ZAdaptiveMAの計算
        short_zma = self._short_zma.calculate(data, short_er)
        long_zma = self._long_zma.calculate(data, long_er)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # クロスオーバーの検出
        # 前日のクロス状態と当日のクロス状態を比較
        prev_short = np.roll(short_zma, 1)
        prev_long = np.roll(long_zma, 1)
        
        # ゴールデンクロス（短期が長期を上抜け）
        golden_cross = (prev_short <= prev_long) & (short_zma > long_zma)
        
        # デッドクロス（短期が長期を下抜け）
        dead_cross = (prev_short >= prev_long) & (short_zma < long_zma)
        
        # シグナルの設定
        signals = np.where(golden_cross, 1, signals)  # ロングエントリー
        signals = np.where(dead_cross, -1, signals)   # ショートエントリー
        
        # 最初の要素はクロスの判定ができないのでシグナルなし
        signals[0] = 0
        
        return signals
        
    def get_short_ma(self) -> np.ndarray:
        """
        短期ZMAの値を取得
        
        Returns:
            np.ndarray: 短期ZMAの値
        """
        if hasattr(self._short_zma, 'get_values'):
            return self._short_zma.get_values()
        return np.array([])
        
    def get_long_ma(self) -> np.ndarray:
        """
        長期ZMAの値を取得
        
        Returns:
            np.ndarray: 長期ZMAの値
        """
        if hasattr(self._long_zma, 'get_values'):
            return self._long_zma.get_values()
        return np.array([]) 