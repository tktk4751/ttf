#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_adaptive_channel.breakout_entry import ZAdaptiveChannelBreakoutEntrySignal


class ZASimpleSignalGenerator(BaseSignalGenerator):
    """
    ZAdaptiveChannelのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: ZAdaptiveChannelのブレイクアウトで買いシグナル
    - ショート: ZAdaptiveChannelのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: ZAdaptiveChannelの売りシグナル
    - ショート: ZAdaptiveChannelの買いシグナル
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
        
        # 乗数計算方法選択
        multiplier_method: str = 'simple_adjustment',  # 'adaptive', 'simple', 'simple_adjustment'
        
        # トリガーソース選択
        multiplier_source: str = 'cer',  # 'cer', 'x_trend', 'z_trend'
        ma_source: str = 'x_trend',      # ZAdaptiveMAに渡すソース（'cer', 'x_trend'）
        
        # X-Trend Index調整の有効化
        use_x_trend_adjustment: bool = True,
        
        # 乗数平滑化オプション
        multiplier_smoothing_method: str = 'none',  # 'none', 'alma', 'hma', 'hyper', 'ema'
        multiplier_smoothing_period: int = 4,        # 平滑化期間
        alma_offset: float = 0.85,                   # ALMA用オフセット
        alma_sigma: float = 6,                       # ALMA用シグマ
        
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
        
        # Xトレンドインデックスパラメータ
        x_detector_type: str = 'dudi_e',
        x_cycle_part: float = 0.7,
        x_max_cycle: int = 120,
        x_min_cycle: int = 5,
        x_max_output: int = 55,
        x_min_output: int = 8,
        x_smoother_type: str = 'alma',
        
        # 固定しきい値のパラメータ（XTrendIndex用）
        fixed_threshold: float = 0.65,
        
        # ROC Persistenceパラメータ
        roc_detector_type: str = 'hody_e',
        roc_max_persistence_periods: int = 89,
        roc_smooth_persistence: bool = False,
        roc_persistence_smooth_period: int = 3,
        roc_smooth_roc: bool = True,
        roc_alma_period: int = 5,
        roc_alma_offset: float = 0.85,
        roc_alma_sigma: float = 6,
        roc_signal_threshold: float = 0.0,
        
        # Cycle RSXパラメータ
        cycle_rsx_detector_type: str = 'dudi_e',
        cycle_rsx_lp_period: int = 5,
        cycle_rsx_hp_period: int = 89,
        cycle_rsx_cycle_part: float = 0.4,
        cycle_rsx_max_cycle: int = 55,
        cycle_rsx_min_cycle: int = 5,
        cycle_rsx_max_output: int = 34,
        cycle_rsx_min_output: int = 3,
        cycle_rsx_src_type: str = 'hlc3'
    ):
        """初期化"""
        super().__init__("ZASimpleSignalGenerator")
        
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
            
            # 乗数計算方法選択
            'multiplier_method': multiplier_method,
            
            # トリガーソース選択
            'multiplier_source': multiplier_source,
            'ma_source': ma_source,
            
            # X-Trend Index調整の有効化
            'use_x_trend_adjustment': use_x_trend_adjustment,
            
            # 乗数平滑化オプション
            'multiplier_smoothing_method': multiplier_smoothing_method,
            'multiplier_smoothing_period': multiplier_smoothing_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            
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
            
            # Xトレンドインデックスパラメータ
            'x_detector_type': x_detector_type,
            'x_cycle_part': x_cycle_part,
            'x_max_cycle': x_max_cycle,
            'x_min_cycle': x_min_cycle,
            'x_max_output': x_max_output,
            'x_min_output': x_min_output,
            'x_smoother_type': x_smoother_type,
            
            # 固定しきい値のパラメータ（XTrendIndex用）
            'fixed_threshold': fixed_threshold,
            
            # ROC Persistenceパラメータ
            'roc_detector_type': roc_detector_type,
            'roc_max_persistence_periods': roc_max_persistence_periods,
            'roc_smooth_persistence': roc_smooth_persistence,
            'roc_persistence_smooth_period': roc_persistence_smooth_period,
            'roc_smooth_roc': roc_smooth_roc,
            'roc_alma_period': roc_alma_period,
            'roc_alma_offset': roc_alma_offset,
            'roc_alma_sigma': roc_alma_sigma,
            'roc_signal_threshold': roc_signal_threshold,
            
            # Cycle RSXパラメータ
            'cycle_rsx_detector_type': cycle_rsx_detector_type,
            'cycle_rsx_lp_period': cycle_rsx_lp_period,
            'cycle_rsx_hp_period': cycle_rsx_hp_period,
            'cycle_rsx_cycle_part': cycle_rsx_cycle_part,
            'cycle_rsx_max_cycle': cycle_rsx_max_cycle,
            'cycle_rsx_min_cycle': cycle_rsx_min_cycle,
            'cycle_rsx_max_output': cycle_rsx_max_output,
            'cycle_rsx_min_output': cycle_rsx_min_output,
            'cycle_rsx_src_type': cycle_rsx_src_type
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
                
                # 乗数計算方法選択
                'multiplier_method': multiplier_method,
                
                # トリガーソース選択
                'multiplier_source': multiplier_source,
                'ma_source': ma_source,
                
                # X-Trend Index調整の有効化
                'use_x_trend_adjustment': use_x_trend_adjustment,
                
                # 乗数平滑化オプション
                'multiplier_smoothing_method': multiplier_smoothing_method,
                'multiplier_smoothing_period': multiplier_smoothing_period,
                'alma_offset': alma_offset,
                'alma_sigma': alma_sigma,
                
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
                
                # Xトレンドインデックスパラメータ
                'x_detector_type': x_detector_type,
                'x_cycle_part': x_cycle_part,
                'x_max_cycle': x_max_cycle,
                'x_min_cycle': x_min_cycle,
                'x_max_output': x_max_output,
                'x_min_output': x_min_output,
                'x_smoother_type': x_smoother_type,
                
                # 固定しきい値のパラメータ（XTrendIndex用）
                'fixed_threshold': fixed_threshold,
                
                # ROC Persistenceパラメータ
                'roc_detector_type': roc_detector_type,
                'roc_max_persistence_periods': roc_max_persistence_periods,
                'roc_smooth_persistence': roc_smooth_persistence,
                'roc_persistence_smooth_period': roc_persistence_smooth_period,
                'roc_smooth_roc': roc_smooth_roc,
                'roc_alma_period': roc_alma_period,
                'roc_alma_offset': roc_alma_offset,
                'roc_alma_sigma': roc_alma_sigma,
                'roc_signal_threshold': roc_signal_threshold,
                
                # Cycle RSXパラメータ
                'cycle_rsx_detector_type': cycle_rsx_detector_type,
                'cycle_rsx_lp_period': cycle_rsx_lp_period,
                'cycle_rsx_hp_period': cycle_rsx_hp_period,
                'cycle_rsx_cycle_part': cycle_rsx_cycle_part,
                'cycle_rsx_max_cycle': cycle_rsx_max_cycle,
                'cycle_rsx_min_cycle': cycle_rsx_min_cycle,
                'cycle_rsx_max_output': cycle_rsx_max_output,
                'cycle_rsx_min_output': cycle_rsx_min_output,
                'cycle_rsx_src_type': cycle_rsx_src_type
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_adaptive_channel_signals = None
    
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
                
                # ZAdaptiveChannelシグナルの計算
                try:
                    z_adaptive_channel_signals = self.z_adaptive_channel_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = z_adaptive_channel_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._z_adaptive_channel_signals = z_adaptive_channel_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_adaptive_channel_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_adaptive_channel_signals = np.zeros(len(data), dtype=np.int8)
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
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_adaptive_channel_signal.get_dynamic_multiplier()
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZATRの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_adaptive_channel_signal.get_z_atr()
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([]) 