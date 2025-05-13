#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from .breakout_entry import CZChannelBreakoutEntrySignal


class CZSimpleSignalGenerator(BaseSignal):
    """
    CZチャネルのシグナル生成クラス（両方向・高速化版）- トレンドフィルターなし
    
    エントリー条件:
    - ロング: CZチャネルのブレイクアウトで買いシグナル
    - ショート: CZチャネルのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: CZチャネルの売りシグナル
    - ショート: CZチャネルの買いシグナル
    """
    
    def __init__(
        self,
        # CZチャネルのパラメータ
        detector_type: str = 'hody',
        cer_detector_type: str = None,  # CER用の検出器タイプ（デフォルトではdetector_typeと同じ）
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
        zma_max_dc_lp_period: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用LPピリオド
        zma_max_dc_hp_period: int = 55,         # ZMA: 最大期間用ドミナントサイクル計算用HPピリオド
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_lp_period: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用LPピリオド
        zma_min_dc_hp_period: int = 34,         # ZMA: 最小期間用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        zma_slow_max_dc_lp_period: int = 5,      # ZMA: Slow最大用ドミナントサイクル計算用LPピリオド
        zma_slow_max_dc_hp_period: int = 55,     # ZMA: Slow最大用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        zma_slow_min_dc_lp_period: int = 5,      # ZMA: Slow最小用ドミナントサイクル計算用LPピリオド
        zma_slow_min_dc_hp_period: int = 34,     # ZMA: Slow最小用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,
        zma_fast_max_dc_lp_period: int = 5,      # ZMA: Fast最大用ドミナントサイクル計算用LPピリオド
        zma_fast_max_dc_hp_period: int = 21,     # ZMA: Fast最大用ドミナントサイクル計算用HPピリオド
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """初期化"""
        super().__init__("CZSimpleSignalGenerator")
        
        # CER検出器タイプの初期化（None の場合は detector_type を使用）
        if cer_detector_type is None:
            cer_detector_type = detector_type
        
        # パラメータの設定
        self._params = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
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
            'zma_max_dc_lp_period': zma_max_dc_lp_period,
            'zma_max_dc_hp_period': zma_max_dc_hp_period,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_min_dc_lp_period': zma_min_dc_lp_period,
            'zma_min_dc_hp_period': zma_min_dc_hp_period,
            'zma_slow_max_dc_cycle_part': zma_slow_max_dc_cycle_part,
            'zma_slow_max_dc_max_cycle': zma_slow_max_dc_max_cycle,
            'zma_slow_max_dc_min_cycle': zma_slow_max_dc_min_cycle,
            'zma_slow_max_dc_max_output': zma_slow_max_dc_max_output,
            'zma_slow_max_dc_min_output': zma_slow_max_dc_min_output,
            'zma_slow_max_dc_lp_period': zma_slow_max_dc_lp_period,
            'zma_slow_max_dc_hp_period': zma_slow_max_dc_hp_period,
            'zma_slow_min_dc_cycle_part': zma_slow_min_dc_cycle_part,
            'zma_slow_min_dc_max_cycle': zma_slow_min_dc_max_cycle,
            'zma_slow_min_dc_min_cycle': zma_slow_min_dc_min_cycle,
            'zma_slow_min_dc_max_output': zma_slow_min_dc_max_output,
            'zma_slow_min_dc_min_output': zma_slow_min_dc_min_output,
            'zma_slow_min_dc_lp_period': zma_slow_min_dc_lp_period,
            'zma_slow_min_dc_hp_period': zma_slow_min_dc_hp_period,
            'zma_fast_max_dc_cycle_part': zma_fast_max_dc_cycle_part,
            'zma_fast_max_dc_max_cycle': zma_fast_max_dc_max_cycle,
            'zma_fast_max_dc_min_cycle': zma_fast_max_dc_min_cycle,
            'zma_fast_max_dc_max_output': zma_fast_max_dc_max_output,
            'zma_fast_max_dc_min_output': zma_fast_max_dc_min_output,
            'zma_fast_max_dc_lp_period': zma_fast_max_dc_lp_period,
            'zma_fast_max_dc_hp_period': zma_fast_max_dc_hp_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # CATR用パラメータ
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type
        }
        
        # CZチャネルブレイクアウトシグナルの初期化
        self.cz_channel_signal = CZChannelBreakoutEntrySignal(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # ZMA基本パラメータ
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            zma_max_dc_lp_period=zma_max_dc_lp_period,
            zma_max_dc_hp_period=zma_max_dc_hp_period,
            
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            zma_min_dc_lp_period=zma_min_dc_lp_period,
            zma_min_dc_hp_period=zma_min_dc_hp_period,
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part=zma_slow_max_dc_cycle_part,
            zma_slow_max_dc_max_cycle=zma_slow_max_dc_max_cycle,
            zma_slow_max_dc_min_cycle=zma_slow_max_dc_min_cycle,
            zma_slow_max_dc_max_output=zma_slow_max_dc_max_output,
            zma_slow_max_dc_min_output=zma_slow_max_dc_min_output,
            zma_slow_max_dc_lp_period=zma_slow_max_dc_lp_period,
            zma_slow_max_dc_hp_period=zma_slow_max_dc_hp_period,
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part=zma_slow_min_dc_cycle_part,
            zma_slow_min_dc_max_cycle=zma_slow_min_dc_max_cycle,
            zma_slow_min_dc_min_cycle=zma_slow_min_dc_min_cycle,
            zma_slow_min_dc_max_output=zma_slow_min_dc_max_output,
            zma_slow_min_dc_min_output=zma_slow_min_dc_min_output,
            zma_slow_min_dc_lp_period=zma_slow_min_dc_lp_period,
            zma_slow_min_dc_hp_period=zma_slow_min_dc_hp_period,
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part=zma_fast_max_dc_cycle_part,
            zma_fast_max_dc_max_cycle=zma_fast_max_dc_max_cycle,
            zma_fast_max_dc_min_cycle=zma_fast_max_dc_min_cycle,
            zma_fast_max_dc_max_output=zma_fast_max_dc_max_output,
            zma_fast_max_dc_min_output=zma_fast_max_dc_min_output,
            zma_fast_max_dc_lp_period=zma_fast_max_dc_lp_period,
            zma_fast_max_dc_hp_period=zma_fast_max_dc_hp_period,
            
            # ZMA追加パラメータ
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # CATR基本パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._cz_channel_signals = None
    
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
                
                # CZチャネルシグナルの計算
                try:
                    cz_channel_signals = self.cz_channel_signal.generate(df)
                    
                    # CZトレンドフィルターなしの単純なシグナル
                    self._signals = cz_channel_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._cz_channel_signals = cz_channel_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._cz_channel_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._cz_channel_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._cz_channel_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._cz_channel_signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CZチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cz_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
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
                
            return self.cz_channel_signal.get_cycle_efficiency_ratio()
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
                
            return self.cz_channel_signal.get_dynamic_multiplier()
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATRの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cz_channel_signal.get_c_atr()
        except Exception as e:
            self.logger.error(f"CATR取得中にエラー: {str(e)}")
            return np.array([])
            
    def reset(self) -> None:
        """
        シグナルジェネレーターの状態をリセットする
        """
        super().reset()
        self._data_len = 0
        self._signals = None
        self._cz_channel_signals = None
        self.cz_channel_signal.reset() 