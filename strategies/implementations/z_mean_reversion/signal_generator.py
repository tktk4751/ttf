#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_rsx.trigger import ZRSXTriggerSignal
from signals.implementations.z_hurst_exponent.filter import ZHurstExponentSignal
from signals.implementations.z_donchian.entry import ZDonchianBreakoutEntrySignal


@njit(fastmath=True, parallel=True)
def generate_entry_signals_numba(
    rsx_signals: np.ndarray,
    hurst_signals: np.ndarray,
) -> np.ndarray:
    """
    エントリーシグナルを生成する（高速化版）
    
    Args:
        rsx_signals: ZRSXトリガーシグナルの配列
        hurst_signals: Zハーストシグナルの配列
    
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: ニュートラル)
    """
    length = len(rsx_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        if np.isnan(rsx_signals[i]) or np.isnan(hurst_signals[i]):
            continue
        
        # ロングエントリー: RSXが1（買い）かつハーストが-1（レンジ相場）
        if rsx_signals[i] == 1 and hurst_signals[i] == -1:
            signals[i] = 1
        
        # ショートエントリー: RSXが-1（売り）かつハーストが-1（レンジ相場）
        elif rsx_signals[i] == -1 and hurst_signals[i] == -1:
            signals[i] = -1
    
    return signals


class ZMeanReversionSignalGenerator(BaseSignalGenerator):
    """
    Z平均回帰シグナル生成クラス
    
    エントリー条件:
    - ロング: ZRSXトリガーが買いシグナル(1)かつZハーストが平均回帰モード(-1)
    - ショート: ZRSXトリガーが売りシグナル(-1)かつZハーストが平均回帰モード(-1)
    
    エグジット条件:
    - ロング: Zドンチャンが売りシグナル(-1)
    - ショート: Zドンチャンが買いシグナル(1)
    """
    
    def __init__(
        self,
        # ZRSXトリガーシグナルのパラメータ
        # サイクル効率比(ER)のパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        er_period: int = 10,
        
        # 最大ドミナントサイクル計算パラメータ
        rsx_max_dc_cycle_part: float = 0.5,
        rsx_max_dc_max_cycle: int = 55,
        rsx_max_dc_min_cycle: int = 5,
        rsx_max_dc_max_output: int = 34,
        rsx_max_dc_min_output: int = 14,
        
        # 最小ドミナントサイクル計算パラメータ
        rsx_min_dc_cycle_part: float = 0.25,
        rsx_min_dc_max_cycle: int = 34,
        rsx_min_dc_min_cycle: int = 3,
        rsx_min_dc_max_output: int = 13,
        rsx_min_dc_min_output: int = 3,
        
        # 買われすぎ/売られすぎレベルパラメータ
        min_high_level: float = 75.0,
        max_high_level: float = 85.0,
        min_low_level: float = 25.0,
        max_low_level: float = 15.0,
        
        # Zハーストエクスポネントシグナルのパラメータ
        # 分析ウィンドウパラメータ
        hurst_max_window_dc_cycle_part: float = 0.75,
        hurst_max_window_dc_max_cycle: int = 144,
        hurst_max_window_dc_min_cycle: int = 8,
        hurst_max_window_dc_max_output: int = 200,
        hurst_max_window_dc_min_output: int = 50,
        
        hurst_min_window_dc_cycle_part: float = 0.5,
        hurst_min_window_dc_max_cycle: int = 55,
        hurst_min_window_dc_min_cycle: int = 5,
        hurst_min_window_dc_max_output: int = 50,
        hurst_min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,
        min_lag_ratio: float = 0.1,
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.7,
        min_threshold: float = 0.55,
        
        # Zドンチャンブレイクアウトのパラメータ
        # 最大期間用パラメータ
        donchian_max_dc_cycle_part: float = 0.5,
        donchian_max_dc_max_cycle: int = 144,
        donchian_max_dc_min_cycle: int = 13,
        donchian_max_dc_max_output: int = 89,
        donchian_max_dc_min_output: int = 21,
        
        # 最小期間用パラメータ
        donchian_min_dc_cycle_part: float = 0.25,
        donchian_min_dc_max_cycle: int = 55,
        donchian_min_dc_min_cycle: int = 5,
        donchian_min_dc_max_output: int = 21,
        donchian_min_dc_min_output: int = 8,
        
        # ブレイクアウトパラメータ
        lookback: int = 1,
        
        # ソースタイプ
        src_type: str = 'hlc3'
    ):
        """初期化"""
        super().__init__("ZMeanReversionSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # ZRSXトリガーパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'er_period': er_period,
            
            'rsx_max_dc_cycle_part': rsx_max_dc_cycle_part,
            'rsx_max_dc_max_cycle': rsx_max_dc_max_cycle,
            'rsx_max_dc_min_cycle': rsx_max_dc_min_cycle,
            'rsx_max_dc_max_output': rsx_max_dc_max_output,
            'rsx_max_dc_min_output': rsx_max_dc_min_output,
            
            'rsx_min_dc_cycle_part': rsx_min_dc_cycle_part,
            'rsx_min_dc_max_cycle': rsx_min_dc_max_cycle,
            'rsx_min_dc_min_cycle': rsx_min_dc_min_cycle,
            'rsx_min_dc_max_output': rsx_min_dc_max_output,
            'rsx_min_dc_min_output': rsx_min_dc_min_output,
            
            'min_high_level': min_high_level,
            'max_high_level': max_high_level,
            'min_low_level': min_low_level,
            'max_low_level': max_low_level,
            
            # Zハーストパラメータ
            'hurst_max_window_dc_cycle_part': hurst_max_window_dc_cycle_part,
            'hurst_max_window_dc_max_cycle': hurst_max_window_dc_max_cycle,
            'hurst_max_window_dc_min_cycle': hurst_max_window_dc_min_cycle,
            'hurst_max_window_dc_max_output': hurst_max_window_dc_max_output,
            'hurst_max_window_dc_min_output': hurst_max_window_dc_min_output,
            
            'hurst_min_window_dc_cycle_part': hurst_min_window_dc_cycle_part,
            'hurst_min_window_dc_max_cycle': hurst_min_window_dc_max_cycle,
            'hurst_min_window_dc_min_cycle': hurst_min_window_dc_min_cycle,
            'hurst_min_window_dc_max_output': hurst_min_window_dc_max_output,
            'hurst_min_window_dc_min_output': hurst_min_window_dc_min_output,
            
            'max_lag_ratio': max_lag_ratio,
            'min_lag_ratio': min_lag_ratio,
            
            'max_step': max_step,
            'min_step': min_step,
            
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            
            # Zドンチャンパラメータ
            'donchian_max_dc_cycle_part': donchian_max_dc_cycle_part,
            'donchian_max_dc_max_cycle': donchian_max_dc_max_cycle,
            'donchian_max_dc_min_cycle': donchian_max_dc_min_cycle,
            'donchian_max_dc_max_output': donchian_max_dc_max_output,
            'donchian_max_dc_min_output': donchian_max_dc_min_output,
            
            'donchian_min_dc_cycle_part': donchian_min_dc_cycle_part,
            'donchian_min_dc_max_cycle': donchian_min_dc_max_cycle,
            'donchian_min_dc_min_cycle': donchian_min_dc_min_cycle,
            'donchian_min_dc_max_output': donchian_min_dc_max_output,
            'donchian_min_dc_min_output': donchian_min_dc_min_output,
            
            'lookback': lookback,
            'src_type': src_type
        }
        
        # ZRSXトリガーシグナルの初期化
        self.z_rsx_signal = ZRSXTriggerSignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            er_period=er_period,
            
            max_dc_cycle_part=rsx_max_dc_cycle_part,
            max_dc_max_cycle=rsx_max_dc_max_cycle,
            max_dc_min_cycle=rsx_max_dc_min_cycle,
            max_dc_max_output=rsx_max_dc_max_output,
            max_dc_min_output=rsx_max_dc_min_output,
            
            min_dc_cycle_part=rsx_min_dc_cycle_part,
            min_dc_max_cycle=rsx_min_dc_max_cycle,
            min_dc_min_cycle=rsx_min_dc_min_cycle,
            min_dc_max_output=rsx_min_dc_max_output,
            min_dc_min_output=rsx_min_dc_min_output,
            
            min_high_level=min_high_level,
            max_high_level=max_high_level,
            min_low_level=min_low_level,
            max_low_level=max_low_level
        )
        
        # Zハーストエクスポーネントシグナルの初期化
        self.z_hurst_signal = ZHurstExponentSignal(
            max_window_dc_cycle_part=hurst_max_window_dc_cycle_part,
            max_window_dc_max_cycle=hurst_max_window_dc_max_cycle,
            max_window_dc_min_cycle=hurst_max_window_dc_min_cycle,
            max_window_dc_max_output=hurst_max_window_dc_max_output,
            max_window_dc_min_output=hurst_max_window_dc_min_output,
            
            min_window_dc_cycle_part=hurst_min_window_dc_cycle_part,
            min_window_dc_max_cycle=hurst_min_window_dc_max_cycle,
            min_window_dc_min_cycle=hurst_min_window_dc_min_cycle,
            min_window_dc_max_output=hurst_min_window_dc_max_output,
            min_window_dc_min_output=hurst_min_window_dc_min_output,
            
            max_lag_ratio=max_lag_ratio,
            min_lag_ratio=min_lag_ratio,
            
            max_step=max_step,
            min_step=min_step,
            
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            
            src_type=src_type
        )
        
        # Zドンチャンブレイクアウトシグナルの初期化
        self.z_donchian_signal = ZDonchianBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            max_dc_cycle_part=donchian_max_dc_cycle_part,
            max_dc_max_cycle=donchian_max_dc_max_cycle,
            max_dc_min_cycle=donchian_max_dc_min_cycle,
            max_dc_max_output=donchian_max_dc_max_output,
            max_dc_min_output=donchian_max_dc_min_output,
            
            min_dc_cycle_part=donchian_min_dc_cycle_part,
            min_dc_max_cycle=donchian_min_dc_max_cycle,
            min_dc_min_cycle=donchian_min_dc_min_cycle,
            min_dc_max_output=donchian_min_dc_max_output,
            min_dc_min_output=donchian_min_dc_min_output,
            
            lookback=lookback,
            src_type=src_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._rsx_signals = None
        self._hurst_signals = None
        self._donchian_signals = None
    
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
                    # 各シグナルの計算
                    self._rsx_signals = self.z_rsx_signal.generate(df)
                    self._hurst_signals = self.z_hurst_signal.generate(df)
                    self._donchian_signals = self.z_donchian_signal.generate(df)
                    
                    # エントリーシグナルの計算（Numba高速化）
                    self._signals = generate_entry_signals_numba(
                        self._rsx_signals,
                        self._hurst_signals
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._rsx_signals = np.zeros(current_len, dtype=np.int8)
                    self._hurst_signals = np.zeros(current_len, dtype=np.int8)
                    self._donchian_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._rsx_signals = np.zeros(len(data), dtype=np.int8)
                self._hurst_signals = np.zeros(len(data), dtype=np.int8)
                self._donchian_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._donchian_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._donchian_signals[index] == 1)
        return False
    
    def get_rsx_values(self) -> np.ndarray:
        """ZRSXの値を取得"""
        return self.z_rsx_signal.get_rsx_values() if hasattr(self.z_rsx_signal, 'get_rsx_values') else np.array([])
    
    def get_hurst_values(self) -> np.ndarray:
        """Zハースト指数値を取得"""
        return self.z_hurst_signal.get_filter_values() if hasattr(self.z_hurst_signal, 'get_filter_values') else np.array([])
    
    def get_donchian_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zドンチャンバンド値を取得"""
        return self.z_donchian_signal.get_band_values() if hasattr(self.z_donchian_signal, 'get_band_values') else (np.array([]), np.array([]), np.array([]))
    
    def get_rsx_levels(self) -> Tuple[np.ndarray, np.ndarray]:
        """ZRSX適応的レベルを取得"""
        return self.z_rsx_signal.get_levels() if hasattr(self.z_rsx_signal, 'get_levels') else (np.array([]), np.array([]))
    
    def get_hurst_threshold(self) -> np.ndarray:
        """Zハースト動的しきい値を取得"""
        return self.z_hurst_signal.get_threshold_values() if hasattr(self.z_hurst_signal, 'get_threshold_values') else np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """サイクル効率比を取得"""
        return self.z_rsx_signal.get_efficiency_ratio() if hasattr(self.z_rsx_signal, 'get_efficiency_ratio') else np.array([]) 