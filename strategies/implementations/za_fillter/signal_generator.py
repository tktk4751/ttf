#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_adaptive_channel.breakout_entry import ZAdaptiveChannelBreakoutEntrySignal
from signals.implementations.cycle_trend_index.fillter import CycleTrendIndexFilterSignal


@njit(fastmath=True, parallel=True)
def generate_combined_signals_numba(
    z_adaptive_signals: np.ndarray,
    cycle_trend_signals: np.ndarray
) -> np.ndarray:
    """
    Zアダプティブチャネルシグナルとサイクルトレンドインデックスシグナルを組み合わせた複合シグナルを生成する
    
    Args:
        z_adaptive_signals: Zアダプティブチャネルシグナルの配列
        cycle_trend_signals: サイクルトレンドインデックスシグナルの配列
    
    Returns:
        複合シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: エントリーなし)
    """
    length = len(z_adaptive_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        if np.isnan(z_adaptive_signals[i]) or np.isnan(cycle_trend_signals[i]):
            signals[i] = 0
        elif z_adaptive_signals[i] == 1 and cycle_trend_signals[i] == 1:
            signals[i] = 1  # ロングエントリー
        elif z_adaptive_signals[i] == -1 and cycle_trend_signals[i] == 1:
            signals[i] = -1  # ショートエントリー
    
    return signals


class ZACTISignalGenerator(BaseSignalGenerator):
    """
    Zアダプティブチャネルとサイクルトレンドインデックスフィルターを組み合わせたシグナル生成クラス
    
    エントリー条件:
    - ロング: Zアダプティブチャネルとサイクルトレンドインデックスのシグナルがともに1（トレンド相場で買い）
    - ショート: Zアダプティブチャネルのシグナルが-1でサイクルトレンドインデックスのシグナルが1（トレンド相場で売り）
    
    エグジット条件:
    - ロング: Zアダプティブチャネルのシグナルが-1
    - ショート: Zアダプティブチャネルのシグナルが1
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        src_type: str = 'hlc3',
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # CERパラメータ
        detector_type: str = 'phac_e',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.5,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 55,              # CER用ハイパスフィルター期間
        max_cycle: int = 55,              # CER用最大サイクル期間
        min_cycle: int = 5,               # CER用最小サイクル期間
        max_output: int = 34,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        use_kalman_filter: bool = False,   # CER用カルマンフィルター使用有無
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 30,            # 遅い移動平均の期間（固定値）
        
        # サイクルトレンドインデックスフィルターのパラメータ
        cti_detector_type: str = 'phac_e',
        cti_cycle_part: float = 0.5,
        cti_max_cycle: int = 144,
        cti_min_cycle: int = 5,
        cti_max_output: int = 55,
        cti_min_output: int = 5,
        cti_src_type: str = 'hlc3',
        cti_lp_period: int = 5,
        cti_hp_period: int = 144,
        smooth_er: bool = True,
        er_alma_period: int = 3,
        er_alma_offset: float = 0.85,
        er_alma_sigma: float = 6,
        smooth_chop: bool = True,
        chop_alma_period: int = 3,
        chop_alma_offset: float = 0.85,
        chop_alma_sigma: float = 6,
        max_threshold: float = 0.75,
        min_threshold: float = 0.45
    ):
        """初期化"""
        super().__init__("ZACTISignalGenerator")
        
        # パラメータの設定
        self._params = {
            # 基本パラメータ
            'band_lookback': band_lookback,
            'src_type': src_type,
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # CERパラメータ
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'use_kalman_filter': use_kalman_filter,
            
            # ZAdaptiveMA用パラメータ
            'fast_period': fast_period,
            'slow_period': slow_period,
            
            # サイクルトレンドインデックスフィルターのパラメータ
            'cti_detector_type': cti_detector_type,
            'cti_cycle_part': cti_cycle_part,
            'cti_max_cycle': cti_max_cycle,
            'cti_min_cycle': cti_min_cycle,
            'cti_max_output': cti_max_output,
            'cti_min_output': cti_min_output,
            'cti_src_type': cti_src_type,
            'cti_lp_period': cti_lp_period,
            'cti_hp_period': cti_hp_period,
            'smooth_er': smooth_er,
            'er_alma_period': er_alma_period, 
            'er_alma_offset': er_alma_offset,
            'er_alma_sigma': er_alma_sigma,
            'smooth_chop': smooth_chop,
            'chop_alma_period': chop_alma_period,
            'chop_alma_offset': chop_alma_offset,
            'chop_alma_sigma': chop_alma_sigma,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # ZAdaptiveChannelブレイクアウトシグナルの初期化
        self.z_adaptive_channel_signal = ZAdaptiveChannelBreakoutEntrySignal(
            # 基本パラメータ
            band_lookback=band_lookback,
            z_adaptive_channel_params={
                'src_type': src_type,
                
                # 動的乗数の範囲パラメータ
                'max_max_multiplier': max_max_multiplier,
                'min_max_multiplier': min_max_multiplier,
                'max_min_multiplier': max_min_multiplier,
                'min_min_multiplier': min_min_multiplier,
                
                # CERパラメータ
                'detector_type': detector_type,
                'cycle_part': cycle_part,
                'lp_period': lp_period,
                'hp_period': hp_period,
                'max_cycle': max_cycle,
                'min_cycle': min_cycle,
                'max_output': max_output,
                'min_output': min_output,
                'use_kalman_filter': use_kalman_filter,
                
                # ZAdaptiveMA用パラメータ
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        )
        
        # サイクルトレンドインデックスフィルターの初期化
        self.cycle_trend_filter = CycleTrendIndexFilterSignal(
            detector_type=cti_detector_type,
            cycle_part=cti_cycle_part,
            max_cycle=cti_max_cycle,
            min_cycle=cti_min_cycle,
            max_output=cti_max_output,
            min_output=cti_min_output,
            src_type=cti_src_type,
            lp_period=cti_lp_period,
            hp_period=cti_hp_period,
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma,
            smooth_chop=smooth_chop,
            chop_alma_period=chop_alma_period,
            chop_alma_offset=chop_alma_offset,
            chop_alma_sigma=chop_alma_sigma,
            max_threshold=max_threshold,
            min_threshold=min_threshold
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_adaptive_channel_signals = None
        self._cycle_trend_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # ZAdaptiveChannelシグナルの計算
                    z_adaptive_channel_signals = self.z_adaptive_channel_signal.generate(df)
                    
                    # サイクルトレンドインデックスフィルターの計算
                    cycle_trend_signals = self.cycle_trend_filter.generate(df)
                    
                    # 2つのシグナルを組み合わせた複合シグナルの生成
                    combined_signals = generate_combined_signals_numba(
                        z_adaptive_channel_signals, cycle_trend_signals
                    )
                    
                    # キャッシュに保存
                    self._signals = combined_signals
                    self._z_adaptive_channel_signals = z_adaptive_channel_signals
                    self._cycle_trend_signals = cycle_trend_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_adaptive_channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._cycle_trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_adaptive_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._cycle_trend_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._z_adaptive_channel_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._z_adaptive_channel_signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZAdaptiveChannelのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_adaptive_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_adaptive_channel_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_trend_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルトレンドインデックスフィルター値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクルトレンドインデックスフィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cycle_trend_filter.get_filter_values()
        except Exception as e:
            self.logger.error(f"サイクルトレンドインデックスフィルター値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_trend_threshold(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルトレンドインデックスの動的しきい値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的しきい値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cycle_trend_filter.get_threshold_values()
        except Exception as e:
            self.logger.error(f"動的しきい値取得中にエラー: {str(e)}")
            return np.array([])
