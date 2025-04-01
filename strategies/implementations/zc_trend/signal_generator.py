#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_channel.breakout_entry import ZChannelBreakoutEntrySignal
from signals.implementations.z_trend_index.filter import ZTrendIndexSignal


@njit(fastmath=True, parallel=True)
def combine_signals_numba(
    z_channel_signals: np.ndarray,
    trend_index_signals: np.ndarray
) -> np.ndarray:
    """
    ZチャネルシグナルとZトレンドインデックスシグナルを組み合わせる（高速化版）
    
    Args:
        z_channel_signals: Zチャネルシグナルの配列
        trend_index_signals: Zトレンドインデックスシグナルの配列
    
    Returns:
        組み合わせたシグナルの配列
    """
    length = len(z_channel_signals)
    combined_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        # 両方のシグナルが有効な値を持つ場合のみ処理
        if not np.isnan(z_channel_signals[i]) and not np.isnan(trend_index_signals[i]):
            # トレンド相場（trend_index_signals=1）の場合のみエントリー
            if trend_index_signals[i] == 1:
                combined_signals[i] = z_channel_signals[i]
            # レンジ相場では新規エントリーなし
    
    return combined_signals


class ZCTrendSignalGenerator(BaseSignalGenerator):
    """
    Zチャネル&トレンドインデックスのシグナル生成クラス
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナル(1) かつ Zトレンドインデックスがトレンド相場(1)
    - ショート: Zチャネルの売りシグナル(-1) かつ Zトレンドインデックスがトレンド相場(1)
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル(-1)
    - ショート: Zチャネルの買いシグナル(1)
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # Zトレンドインデックスのパラメータ
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
        max_stddev_period: int = 21,
        min_stddev_period: int = 14,
        max_lookback_period: int = 14,
        min_lookback_period: int = 7,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """初期化"""
        super().__init__("ZCTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # Zチャネルのパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # Zトレンドインデックスのパラメータ
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
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # Zチャネルブレイクアウトシグナルの初期化
        self.z_channel_signal = ZChannelBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            lookback=band_lookback
        )
        
        # Zトレンドインデックスシグナルの初期化
        self.z_trend_index_signal = ZTrendIndexSignal(
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
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            max_threshold=max_threshold,
            min_threshold=min_threshold
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_channel_signals = None
        self._trend_index_signals = None
    
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
                    # Zチャネルシグナルの計算
                    z_channel_signals = self.z_channel_signal.generate(df)
                    
                    # Zトレンドインデックスシグナルの計算
                    trend_index_signals = self.z_trend_index_signal.generate(df)
                    
                    # シグナルの組み合わせ（Numba最適化版）
                    combined_signals = combine_signals_numba(z_channel_signals, trend_index_signals)
                    
                    # 結果とキャッシュを設定
                    self._signals = combined_signals
                    self._z_channel_signals = z_channel_signals
                    self._trend_index_signals = trend_index_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._trend_index_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._trend_index_signals = np.zeros(len(data), dtype=np.int8)
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
        
        # キャッシュされたZチャネルシグナルを使用してエグジット判定
        # トレンドフィルターはエグジットには影響しない
        if position == 1:  # ロングポジション
            return bool(self._z_channel_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._z_channel_signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_z_channel_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zチャネルシグナルの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Zチャネルシグナルの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._z_channel_signals
        except Exception as e:
            self.logger.error(f"Zチャネルシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドインデックスシグナルの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Zトレンドインデックスシグナルの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._trend_index_signals
        except Exception as e:
            self.logger.error(f"Zトレンドインデックスシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドインデックスの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Zトレンドインデックスの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_index_signal.get_filter_values()
        except Exception as e:
            self.logger.error(f"Zトレンドインデックス値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
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
                
            return self.z_channel_signal.get_cycle_efficiency_ratio()
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
                
            return self.z_channel_signal.get_dynamic_multiplier()
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
                
            return self.z_channel_signal.get_z_atr()
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドインデックスの動的しきい値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的しきい値の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_trend_index_signal.get_threshold_values()
        except Exception as e:
            self.logger.error(f"しきい値取得中にエラー: {str(e)}")
            return np.array([]) 