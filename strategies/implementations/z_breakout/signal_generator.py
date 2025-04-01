#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_channel.breakout_entry import ZChannelBreakoutEntrySignal
from signals.implementations.z_hurst_exponent.filter import ZHurstExponentSignal


@njit(fastmath=True, parallel=True)
def generate_combined_signals(z_channel_signals: np.ndarray, z_hurst_signals: np.ndarray) -> np.ndarray:
    """
    ZChannelとZHurstExponentの組み合わせシグナルを生成する（高速化版）
    
    Args:
        z_channel_signals: Zチャネルシグナルの配列
        z_hurst_signals: Zハーストエクスポネントシグナルの配列
    
    Returns:
        組み合わせシグナルの配列
    """
    length = len(z_channel_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        # 無効な値をチェック
        if np.isnan(z_channel_signals[i]) or np.isnan(z_hurst_signals[i]):
            signals[i] = 0
            continue
            
        # ZChannelBreakoutSignalが1かつZHurstExponentSignalが1のときにロング
        if z_channel_signals[i] == 1 and z_hurst_signals[i] == 1:
            signals[i] = 1
        # ZChannelBreakoutSignalが-1かつZHurstExponentSignalが1のときにショート
        elif z_channel_signals[i] == -1 and z_hurst_signals[i] == 1:
            signals[i] = -1
    
    return signals


class ZBreakoutSignalGenerator(BaseSignalGenerator):
    """
    Zブレイクアウトシグナル生成クラス - トレンド相場でのみエントリーするバージョン
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナルかつZハーストフィルターがトレンド相場
    - ショート: Zチャネルの売りシグナルかつZハーストフィルターがトレンド相場
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル
    - ショート: Zチャネルの買いシグナル
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 7.5,    # 最大乗数の最大値
        min_max_multiplier: float = 4.5,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 144,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 89,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        zma_max_slow_period: int = 34,          # ZMA: 遅い移動平均の最大期間
        zma_min_slow_period: int = 9,           # ZMA: 遅い移動平均の最小期間
        zma_max_fast_period: int = 8,           # ZMA: 速い移動平均の最大期間
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.5,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 55,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 55,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.25,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3,        # ZATR: 最小期間用ドミナントサイクル計算用
        
        # Zハーストエクスポネントフィルターのパラメータ
        # 分析ウィンドウパラメータ
        max_window_dc_cycle_part: float = 0.75,
        max_window_dc_max_cycle: int = 144,
        max_window_dc_min_cycle: int = 8,
        max_window_dc_max_output: int = 200,
        max_window_dc_min_output: int = 50,
        
        min_window_dc_cycle_part: float = 0.5,
        min_window_dc_max_cycle: int = 55,
        min_window_dc_min_cycle: int = 5,
        min_window_dc_max_output: int = 50,
        min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,  # 最大ラグはウィンドウの何%か
        min_lag_ratio: float = 0.1,  # 最小ラグはウィンドウの何%か
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.7,
        min_threshold: float = 0.55
    ):
        """初期化"""
        super().__init__("ZBreakoutSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # Zチャネルのパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_max_slow_period': zma_max_slow_period,
            'zma_min_slow_period': zma_min_slow_period,
            'zma_max_fast_period': zma_max_fast_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # ZATR用パラメータ
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            
            # Zハーストエクスポネントフィルターのパラメータ
            'max_window_dc_cycle_part': max_window_dc_cycle_part,
            'max_window_dc_max_cycle': max_window_dc_max_cycle,
            'max_window_dc_min_cycle': max_window_dc_min_cycle,
            'max_window_dc_max_output': max_window_dc_max_output,
            'max_window_dc_min_output': max_window_dc_min_output,
            'min_window_dc_cycle_part': min_window_dc_cycle_part,
            'min_window_dc_max_cycle': min_window_dc_max_cycle,
            'min_window_dc_min_cycle': min_window_dc_min_cycle,
            'min_window_dc_max_output': min_window_dc_max_output,
            'min_window_dc_min_output': min_window_dc_min_output,
            'max_lag_ratio': max_lag_ratio,
            'min_lag_ratio': min_lag_ratio,
            'max_step': max_step,
            'min_step': min_step,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # Zチャネルブレイクアウトシグナルの初期化
        self.z_channel_signal = ZChannelBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            lookback=band_lookback,
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            zma_max_slow_period=zma_max_slow_period,
            zma_min_slow_period=zma_min_slow_period,
            zma_max_fast_period=zma_max_fast_period,
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # ZATR用パラメータ
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output
        )
        
        # Zハーストエクスポネントフィルターの初期化
        self.z_hurst_filter = ZHurstExponentSignal(
            # 分析ウィンドウパラメータ
            max_window_dc_cycle_part=max_window_dc_cycle_part,
            max_window_dc_max_cycle=max_window_dc_max_cycle,
            max_window_dc_min_cycle=max_window_dc_min_cycle,
            max_window_dc_max_output=max_window_dc_max_output,
            max_window_dc_min_output=max_window_dc_min_output,
            
            min_window_dc_cycle_part=min_window_dc_cycle_part,
            min_window_dc_max_cycle=min_window_dc_max_cycle,
            min_window_dc_min_cycle=min_window_dc_min_cycle,
            min_window_dc_max_output=min_window_dc_max_output,
            min_window_dc_min_output=min_window_dc_min_output,
            
            # ラグパラメータ
            max_lag_ratio=max_lag_ratio,
            min_lag_ratio=min_lag_ratio,
            
            # ステップパラメータ
            max_step=max_step,
            min_step=min_step,
            
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            # 動的しきい値のパラメータ
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            
            src_type=src_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._z_channel_signals = None
        self._z_hurst_signals = None
    
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
                
                # Zチャネルシグナルの計算
                try:
                    z_channel_signals = self.z_channel_signal.generate(df)
                    
                    # Zハーストエクスポネントフィルターの計算
                    z_hurst_signals = self.z_hurst_filter.generate(df)
                    
                    # 組み合わせシグナルの生成
                    self._signals = generate_combined_signals(z_channel_signals, z_hurst_signals)
                    
                    # エグジット用のシグナルのキャッシュ
                    self._z_channel_signals = z_channel_signals
                    self._z_hurst_signals = z_hurst_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._z_channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._z_hurst_signals = np.ones(current_len, dtype=np.int8)  # デフォルトはトレンド相場
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._z_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._z_hurst_signals = np.ones(len(data), dtype=np.int8)  # デフォルトはトレンド相場
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
    
    def get_hurst_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zハーストエクスポネント値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Zハーストエクスポネント値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_hurst_filter.get_filter_values()
        except Exception as e:
            self.logger.error(f"ハースト値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的しきい値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的しきい値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.z_hurst_filter.get_threshold_values()
        except Exception as e:
            self.logger.error(f"しきい値取得中にエラー: {str(e)}")
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