#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_v_channel.breakout_entry import ZVChannelBreakoutEntrySignal


class ZVBreakoutSignalGenerator(BaseSignalGenerator):
    """
    ZVチャネルブレイクアウトのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: ZVチャネルの上限ブレイクアウトで買いシグナル(1)
    - ショート: ZVチャネルの下限ブレイクアウトで売りシグナル(-1)
    
    エグジット条件:
    - ロング: ZVチャネルの売りシグナル(-1)
    - ショート: ZVチャネルの買いシグナル(1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # ボリンジャーバンドパラメータ
        bb_max_multiplier: float = 2.5,
        bb_min_multiplier: float = 1.0,
        
        # ZBBの標準偏差計算用パラメータ
        bb_max_cycle_part: float = 0.5,    # 標準偏差最大期間用サイクル部分
        bb_max_max_cycle: int = 144,       # 標準偏差最大期間用最大サイクル
        bb_max_min_cycle: int = 10,        # 標準偏差最大期間用最小サイクル
        bb_max_max_output: int = 89,       # 標準偏差最大期間用最大出力値
        bb_max_min_output: int = 13,       # 標準偏差最大期間用最小出力値
        bb_min_cycle_part: float = 0.25,   # 標準偏差最小期間用サイクル部分
        bb_min_max_cycle: int = 55,        # 標準偏差最小期間用最大サイクル
        bb_min_min_cycle: int = 5,         # 標準偏差最小期間用最小サイクル
        bb_min_max_output: int = 21,       # 標準偏差最小期間用最大出力値
        bb_min_min_output: int = 5,        # 標準偏差最小期間用最小出力値
        
        # Zチャネルパラメータ
        kc_max_multiplier: float = 3.0,
        kc_min_multiplier: float = 1.5,
        kc_smoother_type: str = 'alma',
        
        # ZChannel ZMA用パラメータ
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
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間（0=平滑化しない）
        
        # ZChannel ZATR用パラメータ
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
        
        # 共通パラメータ
        src_type: str = 'hlc3',
        band_lookback: int = 1
    ):
        """初期化"""
        super().__init__("ZVBreakoutSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            
            'bb_max_multiplier': bb_max_multiplier,
            'bb_min_multiplier': bb_min_multiplier,
            
            'bb_max_cycle_part': bb_max_cycle_part,
            'bb_max_max_cycle': bb_max_max_cycle,
            'bb_max_min_cycle': bb_max_min_cycle,
            'bb_max_max_output': bb_max_max_output,
            'bb_max_min_output': bb_max_min_output,
            'bb_min_cycle_part': bb_min_cycle_part,
            'bb_min_max_cycle': bb_min_max_cycle,
            'bb_min_min_cycle': bb_min_min_cycle,
            'bb_min_max_output': bb_min_max_output,
            'bb_min_min_output': bb_min_min_output,
            
            'kc_max_multiplier': kc_max_multiplier,
            'kc_min_multiplier': kc_min_multiplier,
            'kc_smoother_type': kc_smoother_type,
            
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
            
            'src_type': src_type,
            'band_lookback': band_lookback
        }
        
        # ZVチャネルブレイクアウトシグナルの初期化
        self.zv_channel_signal = ZVChannelBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            bb_max_multiplier=bb_max_multiplier,
            bb_min_multiplier=bb_min_multiplier,
            
            bb_max_cycle_part=bb_max_cycle_part,
            bb_max_max_cycle=bb_max_max_cycle,
            bb_max_min_cycle=bb_max_min_cycle,
            bb_max_max_output=bb_max_max_output,
            bb_max_min_output=bb_max_min_output,
            bb_min_cycle_part=bb_min_cycle_part,
            bb_min_max_cycle=bb_min_max_cycle,
            bb_min_min_cycle=bb_min_min_cycle,
            bb_min_max_output=bb_min_max_output,
            bb_min_min_output=bb_min_min_output,
            
            kc_max_multiplier=kc_max_multiplier,
            kc_min_multiplier=kc_min_multiplier,
            kc_smoother_type=kc_smoother_type,
            
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
            
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            
            src_type=src_type,
            lookback=band_lookback
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
    
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
                
                # ZVチャネルシグナルの計算
                try:
                    self._signals = self.zv_channel_signal.generate(df)
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZVチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zv_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_bb_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部ボリンジャーバンドの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zv_channel_signal.get_bb_band_values()
        except Exception as e:
            self.logger.error(f"BBバンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_kc_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部ケルトナーチャネル（Zチャネル）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.zv_channel_signal.get_kc_band_values()
        except Exception as e:
            self.logger.error(f"KCバンド値取得中にエラー: {str(e)}")
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
                
            return self.zv_channel_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([]) 