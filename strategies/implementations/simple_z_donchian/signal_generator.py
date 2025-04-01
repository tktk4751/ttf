#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List
from numba import njit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_donchian.entry import ZDonchianBreakoutEntrySignal

@njit
def calculate_entry_signals(
    donchian_breakout_signals: np.ndarray,
) -> np.ndarray:
    """
    エントリーシグナルを計算する

    Args:
        donchian_breakout_signals: ドンチャンブレイクアウトシグナル配列

    Returns:
        np.ndarray: エントリーシグナル配列（1: ロング、-1: ショート、0: ニュートラル）
    """
    signal_length = len(donchian_breakout_signals)
    entry_signals = np.zeros(signal_length, dtype=np.int8)
    
    for i in range(signal_length):
        # ドンチャンブレイクアウトシグナルに基づいてエントリーシグナルを設定
        entry_signals[i] = donchian_breakout_signals[i]
    
    return entry_signals


class SimpleZDonchianSignalGenerator(BaseSignalGenerator):
    """
    シンプルなZドンチャンシグナルジェネレーター
    ZTrendFilterを使用せず、Zドンチャンブレイクアウトシグナルのみを使用します。
    
    特徴:
    - サイクル効率比(CER)に基づく動的なパラメータ最適化
    - 動的なAVZモード
    - 動的なATR乗数
    """
    
    def __init__(
        self,
        # Zドンチャンチャネルのパラメータ
        donchian_length: int = 55,
        # ドミナントサイクル検出器用パラメータ
        cycle_detector_type: str = 'dudi_dc',
        cycle_part: float = 0.5,
        # CERと長期サイクル検出器用パラメータ
        max_cycle: int = 233,
        min_cycle: int = 13,
        max_output: int = 144,
        min_output: int = 21,
        # 短期サイクル検出器用パラメータ
        short_max_cycle: int = 55,
        short_min_cycle: int = 5,
        short_max_output: int = 34,
        short_min_output: int = 5,
        # AVZオプション
        enable_dynamic_avz: bool = True,
        avz_length: int = 5,
        short_length: int = 3,
        # ATR乗数オプション
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        # 動的ATR乗数用パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.3,    # 最小乗数の最小値
        # その他のパラメータ
        price_mode: str = 'close',
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            donchian_length: ドンチャンチャネルの長さ
            cycle_detector_type: サイクル検出器の種類
            cycle_part: サイクル部分
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
            max_output: 最大出力
            min_output: 最小出力
            short_max_cycle: 短期最大サイクル
            short_min_cycle: 短期最小サイクル
            short_max_output: 短期最大出力
            short_min_output: 短期最小出力
            enable_dynamic_avz: 動的AVZを有効にするかどうか
            avz_length: AVZ長さ
            short_length: 短期長さ
            max_multiplier: 最大乗数
            min_multiplier: 最小乗数
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            price_mode: 価格モード
            src_type: ソースタイプ
        """
        super().__init__("SimpleZDonchianSignalGenerator")

        # ローパスフィルター期間としてAVZ長さを使用
        lp_period = avz_length

        # ハイパスフィルター期間として最大サイクルを使用
        hp_period = max_cycle
        
        # 最大期間パラメータ
        max_dc_cycle_part = cycle_part
        max_dc_max_cycle = max_cycle
        max_dc_min_cycle = min_cycle
        max_dc_max_output = max_output
        max_dc_min_output = min_output

        # 最小期間パラメータ
        min_dc_cycle_part = cycle_part / 2  # 通常は最大の半分
        min_dc_max_cycle = short_max_cycle
        min_dc_min_cycle = short_min_cycle
        min_dc_max_output = short_max_output
        min_dc_min_output = short_min_output

        # Zドンチャンブレイクアウトシグナルの初期化
        self.z_donchian_signal = ZDonchianBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=max_dc_max_output,
            max_dc_min_output=max_dc_min_output,
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=min_dc_max_output,
            min_dc_min_output=min_dc_min_output,
            lookback=1,  # デフォルト値
            src_type=src_type
        )
        
        # 計算済みのシグナルを保存
        self._donchian_result = None
        self._breakout_signals = None
        self._entry_signals = None
        self._data_hash = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        シグナルを計算
        
        Args:
            data: 価格データ
        """
        # データハッシュの計算（オプショナル - キャッシング用）
        data_hash = self._get_data_hash(data) if hasattr(self, '_get_data_hash') else None
        
        if data_hash is not None and data_hash == self._data_hash and self._entry_signals is not None:
            return
            
        if data_hash is not None:
            self._data_hash = data_hash
        
        # Zドンチャンブレイクアウトシグナル計算
        self._breakout_signals = self.z_donchian_signal.generate(data)
        
        # エントリーシグナル計算
        self._entry_signals = calculate_entry_signals(
            self._breakout_signals
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        import hashlib
        
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        return str(data_hash)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        if self._entry_signals is None:
            self.calculate(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if self._entry_signals is None:
            self.calculate(data)
            
        # 反対方向のシグナルがエグジット条件
        if position == 1 and self._entry_signals[index] == -1:
            return True
        elif position == -1 and self._entry_signals[index] == 1:
            return True
        
        return False
    
    def get_upper_band(self) -> np.ndarray:
        """上側バンドを取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        return self.z_donchian_signal.get_band_values()[0]
    
    def get_lower_band(self) -> np.ndarray:
        """下側バンドを取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        return self.z_donchian_signal.get_band_values()[1]
    
    def get_mid_band(self) -> np.ndarray:
        """中間バンドを取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        return self.z_donchian_signal.get_band_values()[2]
    
    def get_cer_values(self) -> np.ndarray:
        """サイクル効率比（CER）値を取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        return self.z_donchian_signal.get_efficiency_ratio()
    
    def get_dynamic_max_multiplier(self) -> np.ndarray:
        """動的最大乗数を取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        
        # ZDonchianBreakoutEntrySignalにはget_dynamic_max_multiplierメソッドがないため、
        # CER値を使って自前で計算
        cer_values = self.get_cer_values()
        
        # 動的な最大乗数を計算（トレンドが弱いほど大きい値）
        max_max_multiplier = 8.0  # 最大乗数の最大値
        min_max_multiplier = 3.0  # 最大乗数の最小値
        
        # CERが低い（トレンドが弱い）ほど最大乗数は大きく、
        # CERが高い（トレンドが強い）ほど最大乗数は小さくなる
        max_multipliers = max_max_multiplier - np.abs(cer_values) * (max_max_multiplier - min_max_multiplier)
        return max_multipliers
    
    def get_dynamic_min_multiplier(self) -> np.ndarray:
        """動的最小乗数を取得"""
        if self._breakout_signals is None:
            raise ValueError("先にcalculateメソッドを呼び出してください")
        
        # ZDonchianBreakoutEntrySignalにはget_dynamic_min_multiplierメソッドがないため、
        # CER値を使って自前で計算
        cer_values = self.get_cer_values()
        
        # 動的な最小乗数を計算（トレンドが強いほど大きい値）
        max_min_multiplier = 1.5  # 最小乗数の最大値
        min_min_multiplier = 0.3  # 最小乗数の最小値
        
        # CERが低い（トレンドが弱い）ほど最小乗数は小さく、
        # CERが高い（トレンドが強い）ほど最小乗数は大きくなる
        min_multipliers = max_min_multiplier - np.abs(cer_values) * (max_min_multiplier - min_min_multiplier)
        return min_multipliers 