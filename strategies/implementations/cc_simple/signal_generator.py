#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
# 使用するシグナルをインポート
from signals.implementations.cc_channel.breakout_entry import CCChannelBreakoutEntrySignal


class CCSimpleSignalGenerator(BaseSignalGenerator): # クラス名を変更
    """
    CCチャネルのシグナル生成クラス（シンプル版）
    
    エントリー条件:
    - ロング: CCチャネルのブレイクアウトで買いシグナル
    - ショート: CCチャネルのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: CCチャネルの売りシグナル
    - ショート: CCチャネルの買いシグナル
    """
    
    def __init__( # 引数を band_lookback のみに変更 -> 全パラメータを受け取るように変更
        self,
        band_lookback: int = 1,
        # CCChannelBreakoutEntrySignal に渡すパラメータを追加
        multiplier_method: str = 'adaptive', # multiplier_method を追加
        new_method_er_source: str = 'cver', # new_method_er_source を追加
        cc_max_max_multiplier: float = 8.0,
        cc_min_max_multiplier: float = 3.0,
        cc_max_min_multiplier: float = 1.5,
        cc_min_min_multiplier: float = 0.5,
        cma_detector_type: str = 'hody_e', 
        cma_cycle_part: float = 0.618, 
        cma_lp_period: int = 5, 
        cma_hp_period: int = 89, 
        cma_max_cycle: int = 55, 
        cma_min_cycle: int = 5, 
        cma_max_output: int = 34, 
        cma_min_output: int = 8, 
        cma_fast_period: int = 2, 
        cma_slow_period: int = 30, 
        cma_src_type: str = 'hlc3',
        catr_detector_type: str = 'hody', 
        catr_cycle_part: float = 0.5, 
        catr_lp_period: int = 5, 
        catr_hp_period: int = 55, 
        catr_max_cycle: int = 55, 
        catr_min_cycle: int = 5, 
        catr_max_output: int = 34, 
        catr_min_output: int = 5, 
        catr_smoother_type: str = 'alma',
        cver_detector_type: str = 'hody', 
        cver_lp_period: int = 5, 
        cver_hp_period: int = 144, 
        cver_cycle_part: float = 0.5, 
        cver_max_cycle: int = 144, 
        cver_min_cycle: int = 5, 
        cver_max_output: int = 89, 
        cver_min_output: int = 5, 
        cver_src_type: str = 'hlc3',
        cer_detector_type: str = 'hody', 
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,
        cer_max_cycle: int = 144,
        cer_min_cycle: int = 5,
        cer_max_output: int = 89,
        cer_min_output: int = 5,
        cer_src_type: str = 'hlc3'
    ):
        """初期化"""
        super().__init__("CCSimpleSignalGenerator") # クラス名を更新
        
        # パラメータの設定 (全パラメータを保存)
        self._params = {
            'band_lookback': band_lookback,
            'multiplier_method': multiplier_method, # multiplier_method を追加
            'new_method_er_source': new_method_er_source, # new_method_er_source を追加
            'cc_max_max_multiplier': cc_max_max_multiplier,
            'cc_min_max_multiplier': cc_min_max_multiplier,
            'cc_max_min_multiplier': cc_max_min_multiplier,
            'cc_min_min_multiplier': cc_min_min_multiplier,
            'cma_detector_type': cma_detector_type, 
            'cma_cycle_part': cma_cycle_part, 
            'cma_lp_period': cma_lp_period, 
            'cma_hp_period': cma_hp_period, 
            'cma_max_cycle': cma_max_cycle, 
            'cma_min_cycle': cma_min_cycle, 
            'cma_max_output': cma_max_output, 
            'cma_min_output': cma_min_output, 
            'cma_fast_period': cma_fast_period, 
            'cma_slow_period': cma_slow_period, 
            'cma_src_type': cma_src_type,
            'catr_detector_type': catr_detector_type, 
            'catr_cycle_part': catr_cycle_part, 
            'catr_lp_period': catr_lp_period, 
            'catr_hp_period': catr_hp_period, 
            'catr_max_cycle': catr_max_cycle, 
            'catr_min_cycle': catr_min_cycle, 
            'catr_max_output': catr_max_output, 
            'catr_min_output': catr_min_output, 
            'catr_smoother_type': catr_smoother_type,
            'cver_detector_type': cver_detector_type, 
            'cver_lp_period': cver_lp_period, 
            'cver_hp_period': cver_hp_period, 
            'cver_cycle_part': cver_cycle_part, 
            'cver_max_cycle': cver_max_cycle, 
            'cver_min_cycle': cver_min_cycle, 
            'cver_max_output': cver_max_output, 
            'cver_min_output': cver_min_output, 
            'cver_src_type': cver_src_type,
            'cer_detector_type': cer_detector_type, 
            'cer_lp_period': cer_lp_period,
            'cer_hp_period': cer_hp_period,
            'cer_cycle_part': cer_cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'cer_src_type': cer_src_type
        }
        
        # CCチャネルブレイクアウトシグナルの初期化 (全パラメータを渡す)
        self.cc_channel_signal = CCChannelBreakoutEntrySignal(
            band_lookback=band_lookback,
            multiplier_method=multiplier_method, # multiplier_method を渡す
            new_method_er_source=new_method_er_source, # new_method_er_source を渡す
            cc_max_max_multiplier=cc_max_max_multiplier,
            cc_min_max_multiplier=cc_min_max_multiplier,
            cc_max_min_multiplier=cc_max_min_multiplier,
            cc_min_min_multiplier=cc_min_min_multiplier,
            cma_detector_type=cma_detector_type, 
            cma_cycle_part=cma_cycle_part, 
            cma_lp_period=cma_lp_period, 
            cma_hp_period=cma_hp_period, 
            cma_max_cycle=cma_max_cycle, 
            cma_min_cycle=cma_min_cycle, 
            cma_max_output=cma_max_output, 
            cma_min_output=cma_min_output, 
            cma_fast_period=cma_fast_period, 
            cma_slow_period=cma_slow_period, 
            cma_src_type=cma_src_type,
            catr_detector_type=catr_detector_type, 
            catr_cycle_part=catr_cycle_part, 
            catr_lp_period=catr_lp_period, 
            catr_hp_period=catr_hp_period, 
            catr_max_cycle=catr_max_cycle, 
            catr_min_cycle=catr_min_cycle, 
            catr_max_output=catr_max_output, 
            catr_min_output=catr_min_output, 
            catr_smoother_type=catr_smoother_type,
            cver_detector_type=cver_detector_type, 
            cver_lp_period=cver_lp_period, 
            cver_hp_period=cver_hp_period, 
            cver_cycle_part=cver_cycle_part, 
            cver_max_cycle=cver_max_cycle, 
            cver_min_cycle=cver_min_cycle, 
            cver_max_output=cver_max_output, 
            cver_min_output=cver_min_output, 
            cver_src_type=cver_src_type,
            cer_detector_type=cer_detector_type, 
            cer_lp_period=cer_lp_period,
            cer_hp_period=cer_hp_period,
            cer_cycle_part=cer_cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            cer_src_type=cer_src_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._cc_channel_signals = None # 変数名を変更 (z_channel -> cc_channel)
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            # 内部の CCChannelBreakoutEntrySignal がキャッシュを持つため、
            # ここでのキャッシュは限定的かもしれないが、念のため保持
            if self._signals is None or current_len != self._data_len:
                # データ検証は cc_channel_signal 側で行われる
                
                # CCチャネルシグナルの計算
                try:
                    cc_channel_signals = self.cc_channel_signal.generate(data)
                    
                    # 単純なブレイクアウトシグナルを使用
                    self._signals = cc_channel_signals
                    
                    # エグジット用のシグナルをキャッシュ
                    self._cc_channel_signals = cc_channel_signals # 変数名を変更
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._cc_channel_signals = np.zeros(current_len, dtype=np.int8) # 変数名を変更
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._cc_channel_signals = np.zeros(len(data), dtype=np.int8) # 変数名を変更
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
        
        # インデックス範囲チェックを追加
        if index < 0 or index >= len(self._cc_channel_signals):
             self.logger.warning(f"get_exit_signals: 無効なインデックス {index} (データ長: {len(self._cc_channel_signals)})。False を返します。")
             return False

        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._cc_channel_signals[index] == -1) # 変数名を変更
        elif position == -1:  # ショートポジション
            return bool(self._cc_channel_signals[index] == 1) # 変数名を変更
        return False
    
    # --- ゲッターメソッド (CCChannelBreakoutEntrySignal のゲッターをラップ) --- 
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CCチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線(CMA), 上限バンド, 下限バンド)のタプル
        """
        try:
            # calculate_signals を通さずに直接シグナルのメソッドを呼ぶ
            # シグナル側でキャッシュが効くはず
            return self.cc_channel_signal.get_band_values(data)
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_cycle_volatility_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルボラティリティ効率比 (CVER) の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CVERの値
        """
        try:
            return self.cc_channel_signal.get_cycle_volatility_er(data)
        except Exception as e:
            self.logger.error(f"CVER取得中にエラー: {str(e)}")
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
            return self.cc_channel_signal.get_dynamic_multiplier(data)
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_catr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATRの値
        """
        try:
            return self.cc_channel_signal.get_catr(data)
        except Exception as e:
            self.logger.error(f"CATR取得中にエラー: {str(e)}")
            return np.array([]) 