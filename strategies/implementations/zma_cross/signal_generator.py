#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_ma_cross.entry import ZMACrossEntrySignal


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    z_ma_cross_signals: np.ndarray
) -> np.ndarray:
    """
    ZMA交差シグナルを処理する高速実装
    
    Args:
        z_ma_cross_signals: ZMA交差シグナルの配列
    
    Returns:
        処理したシグナルの配列
    """
    length = len(z_ma_cross_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        # 有効なデータがあるか確認
        if not np.isnan(z_ma_cross_signals[i]):
            if z_ma_cross_signals[i] == 1:
                # ゴールデンクロス（ロングエントリー）
                signals[i] = 1
            elif z_ma_cross_signals[i] == -1:
                # デッドクロス（ショートエントリー）
                signals[i] = -1
            else:
                # ニュートラル
                signals[i] = 0
    
    return signals


class ZMACrossSignalGenerator(BaseSignalGenerator):
    """
    ZMA交差のシグナル生成クラス
    
    エントリー条件:
    - ロング: ZMA交差のゴールデンクロス(1)
    - ショート: ZMA交差のデッドクロス(-1)
    
    エグジット条件:
    - ロング: ZMA交差のデッドクロス(-1)
    - ショート: ZMA交差のゴールデンクロス(1)
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
        """初期化"""
        super().__init__("ZMACrossSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # ドミナントサイクル・効率比（CER）の基本パラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            
            # 短期ZMA用パラメータ
            'fast_max_dc_cycle_part': fast_max_dc_cycle_part,
            'fast_max_dc_max_cycle': fast_max_dc_max_cycle,
            'fast_max_dc_min_cycle': fast_max_dc_min_cycle,
            'fast_max_dc_max_output': fast_max_dc_max_output,
            'fast_max_dc_min_output': fast_max_dc_min_output,
            
            'fast_min_dc_cycle_part': fast_min_dc_cycle_part,
            'fast_min_dc_max_cycle': fast_min_dc_max_cycle,
            'fast_min_dc_min_cycle': fast_min_dc_min_cycle,
            'fast_min_dc_max_output': fast_min_dc_max_output,
            'fast_min_dc_min_output': fast_min_dc_min_output,
            
            'fast_max_slow_period': fast_max_slow_period,
            'fast_min_slow_period': fast_min_slow_period,
            'fast_max_fast_period': fast_max_fast_period,
            'fast_min_fast_period': fast_min_fast_period,
            'fast_hyper_smooth_period': fast_hyper_smooth_period,
            
            # 長期ZMA用パラメータ
            'slow_max_dc_cycle_part': slow_max_dc_cycle_part,
            'slow_max_dc_max_cycle': slow_max_dc_max_cycle,
            'slow_max_dc_min_cycle': slow_max_dc_min_cycle,
            'slow_max_dc_max_output': slow_max_dc_max_output,
            'slow_max_dc_min_output': slow_max_dc_min_output,
            
            'slow_min_dc_cycle_part': slow_min_dc_cycle_part,
            'slow_min_dc_max_cycle': slow_min_dc_max_cycle,
            'slow_min_dc_min_cycle': slow_min_dc_min_cycle,
            'slow_min_dc_max_output': slow_min_dc_max_output,
            'slow_min_dc_min_output': slow_min_dc_min_output,
            
            'slow_max_slow_period': slow_max_slow_period,
            'slow_min_slow_period': slow_min_slow_period,
            'slow_max_fast_period': slow_max_fast_period,
            'slow_min_fast_period': slow_min_fast_period,
            'slow_hyper_smooth_period': slow_hyper_smooth_period,
            
            # ソースタイプ
            'src_type': src_type
        }
        
        # ZMA交差シグナルの初期化
        self.zma_cross_signal = ZMACrossEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            fast_max_dc_cycle_part=fast_max_dc_cycle_part,
            fast_max_dc_max_cycle=fast_max_dc_max_cycle,
            fast_max_dc_min_cycle=fast_max_dc_min_cycle,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            
            fast_min_dc_cycle_part=fast_min_dc_cycle_part,
            fast_min_dc_max_cycle=fast_min_dc_max_cycle,
            fast_min_dc_min_cycle=fast_min_dc_min_cycle,
            fast_min_dc_max_output=fast_min_dc_max_output,
            fast_min_dc_min_output=fast_min_dc_min_output,
            
            fast_max_slow_period=fast_max_slow_period,
            fast_min_slow_period=fast_min_slow_period,
            fast_max_fast_period=fast_max_fast_period,
            fast_min_fast_period=fast_min_fast_period,
            fast_hyper_smooth_period=fast_hyper_smooth_period,
            
            slow_max_dc_cycle_part=slow_max_dc_cycle_part,
            slow_max_dc_max_cycle=slow_max_dc_max_cycle,
            slow_max_dc_min_cycle=slow_max_dc_min_cycle,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            
            slow_min_dc_cycle_part=slow_min_dc_cycle_part,
            slow_min_dc_max_cycle=slow_min_dc_max_cycle,
            slow_min_dc_min_cycle=slow_min_dc_min_cycle,
            slow_min_dc_max_output=slow_min_dc_max_output,
            slow_min_dc_min_output=slow_min_dc_min_output,
            
            slow_max_slow_period=slow_max_slow_period,
            slow_min_slow_period=slow_min_slow_period,
            slow_max_fast_period=slow_max_fast_period,
            slow_min_fast_period=slow_min_fast_period,
            slow_hyper_smooth_period=slow_hyper_smooth_period,
            
            src_type=src_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._zma_cross_signals = None
    
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
                    # ZMA交差シグナルの計算
                    zma_cross_signals = self.zma_cross_signal.generate(df)
                    
                    # デバッグログの追加
                    self.logger.debug(f"ZMA交差シグナル: {zma_cross_signals[-10:]} (最後の10個)")
                    
                    # シグナルの処理（Numba最適化版）
                    signals = generate_signals_numba(zma_cross_signals)
                    
                    # デバッグログの追加
                    self.logger.debug(f"処理後のシグナル: {signals[-10:]} (最後の10個)")
                    
                    # 結果とキャッシュを設定
                    self._signals = signals
                    self._zma_cross_signals = zma_cross_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._zma_cross_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._zma_cross_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        # 最後の値をログに出力
        last_idx = len(self._signals) - 1
        if last_idx >= 0:
            self.logger.debug(f"最後のエントリーシグナル: {self._signals[last_idx]}")
        
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたZMA交差シグナルを使用してエグジット判定
        if position == 1:  # ロングポジション
            return bool(self._zma_cross_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._zma_cross_signals[index] == 1)
        return False
    
    def get_zma_cross_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZMA交差シグナルの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZMA交差シグナルの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._zma_cross_signals
        except Exception as e:
            self.logger.error(f"ZMA交差シグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_zma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (短期ZMA, 長期ZMA)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zma_cross_signal.get_zma_values()
        except Exception as e:
            self.logger.error(f"ZMA値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比（CER）の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zma_cross_signal.get_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fast_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        短期ZMAの動的な期間を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zma_cross_signal.get_fast_dynamic_periods()
        except Exception as e:
            self.logger.error(f"短期ZMA動的期間取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_slow_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        長期ZMAの動的な期間を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zma_cross_signal.get_slow_dynamic_periods()
        except Exception as e:
            self.logger.error(f"長期ZMA動的期間取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty 