#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.divergence.z_macd_divergence import ZMACDDivergenceSignal
from signals.implementations.z_bb.breakout_entry import ZBBBreakoutEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_entry_signals(z_macd_divergence_signals: np.ndarray) -> np.ndarray:
    """
    エントリーシグナルを計算する（高速化版）
    
    Args:
        z_macd_divergence_signals: ZMACDダイバージェンスシグナルの配列 (1:ロング, -1:ショート, 0:ニュートラル)
    
    Returns:
        np.ndarray: エントリーシグナル (1:ロング, -1:ショート, 0:ニュートラル)
    """
    # ZMACDダイバージェンスシグナルをそのまま使用
    return z_macd_divergence_signals.copy()


@njit(fastmath=True)
def check_exit_signal(position: int, index: int, z_macd_divergence_signals: np.ndarray, z_bb_signals: np.ndarray) -> bool:
    """
    エグジットシグナルをチェック（高速化版）
    
    Args:
        position: 現在のポジション (1:ロング, -1:ショート)
        index: チェックするインデックス
        z_macd_divergence_signals: ZMACDダイバージェンスシグナル
        z_bb_signals: ZBBブレイクアウトシグナル
    
    Returns:
        bool: エグジットすべきかどうか
    """
    # インデックスが範囲外の場合はエグジットなし
    if index < 0 or index >= len(z_macd_divergence_signals):
        return False
    
    # ロングポジションのエグジット条件: ZBBBreakoutEntrySignalが-1
    if position == 1 and z_bb_signals[index] == -1:
        return True
    
    # ショートポジションのエグジット条件: ZBBBreakoutEntrySignalが1
    if position == -1 and z_bb_signals[index] == 1:
        return True
    
    return False


class ZMACDBreakoutSignalGenerator(BaseSignalGenerator):
    """
    ZMACDブレイクアウトシグナルジェネレーター
    
    特徴:
    - ZMACDダイバージェンスシグナルをエントリーに使用
    - ZBBブレイクアウトシグナルをエグジットに使用
    - サイクル効率比（CER）に基づく動的なパラメータ調整
    
    エントリー条件:
    - ロング: ZMACDダイバージェンスシグナルが1のとき
    - ショート: ZMACDダイバージェンスシグナルが-1のとき
    
    エグジット条件:
    - ロング決済: ZBBブレイクアウトシグナルが-1のとき
    - ショート決済: ZBBブレイクアウトシグナルが1のとき
    """
    
    def __init__(
        self,
        # 共通パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        src_type: str = 'hlc3',
        
        # ZMACDダイバージェンス用パラメータ
        er_period: int = 21,
        # 短期線用パラメータ
        fast_max_dc_max_output: int = 21,
        fast_max_dc_min_output: int = 5,
        # 長期線用パラメータ
        slow_max_dc_max_output: int = 55,
        slow_max_dc_min_output: int = 13,
        # シグナル線用パラメータ
        signal_max_dc_max_output: int = 21,
        signal_max_dc_min_output: int = 5,
        max_slow_period: int = 34,
        min_slow_period: int = 13,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        lookback: int = 30,
        
        # ZBB用パラメータ
        bb_max_multiplier: float = 2.5,
        bb_min_multiplier: float = 1.0,
        
        # ZBB標準偏差最大期間用パラメータ
        bb_max_cycle_part: float = 0.5,
        bb_max_max_cycle: int = 144,
        bb_max_min_cycle: int = 10,
        bb_max_max_output: int = 89,
        bb_max_min_output: int = 13,
        
        # ZBB標準偏差最小期間用パラメータ
        bb_min_cycle_part: float = 0.25,
        bb_min_max_cycle: int = 55,
        bb_min_min_cycle: int = 5,
        bb_min_max_output: int = 21,
        bb_min_min_output: int = 5,
        
        # ブレイクアウトパラメータ
        bb_lookback: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            src_type: ソースタイプ
            
            er_period: 効率比の計算期間
            fast_max_dc_max_output: 短期線用最大期間出力値
            fast_max_dc_min_output: 短期線用最大期間出力の最小値
            slow_max_dc_max_output: 長期線用最大期間出力値
            slow_max_dc_min_output: 長期線用最大期間出力の最小値
            signal_max_dc_max_output: シグナル線用最大期間出力値
            signal_max_dc_min_output: シグナル線用最大期間出力の最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
            lookback: ダイバージェンス検出のルックバック期間
            
            bb_max_multiplier: ボリンジャーバンド最大乗数
            bb_min_multiplier: ボリンジャーバンド最小乗数
            bb_max_cycle_part: BB最大期間用サイクル部分
            bb_max_max_cycle: BB最大期間用最大サイクル
            bb_max_min_cycle: BB最大期間用最小サイクル
            bb_max_max_output: BB最大期間用最大出力
            bb_max_min_output: BB最大期間用最小出力
            bb_min_cycle_part: BB最小期間用サイクル部分
            bb_min_max_cycle: BB最小期間用最大サイクル
            bb_min_min_cycle: BB最小期間用最小サイクル
            bb_min_max_output: BB最小期間用最大出力
            bb_min_min_output: BB最小期間用最小出力
            bb_lookback: ブレイクアウト用ルックバック期間
        """
        super().__init__("ZMACDBreakoutSignalGenerator")
        
        # ZMACDダイバージェンスシグナルの初期化
        self.z_macd_divergence_signal = ZMACDDivergenceSignal(
            er_period=er_period,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            signal_max_dc_max_output=signal_max_dc_max_output,
            signal_max_dc_min_output=signal_max_dc_min_output,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=lookback
        )
        
        # ZBBブレイクアウトシグナルの初期化
        self.z_bb_signal = ZBBBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=bb_max_multiplier,
            min_multiplier=bb_min_multiplier,
            max_cycle_part=bb_max_cycle_part,
            max_max_cycle=bb_max_max_cycle,
            max_min_cycle=bb_max_min_cycle,
            max_max_output=bb_max_max_output,
            max_min_output=bb_max_min_output,
            min_cycle_part=bb_min_cycle_part,
            min_max_cycle=bb_min_max_cycle,
            min_min_cycle=bb_min_min_cycle,
            min_max_output=bb_min_max_output,
            min_min_output=bb_min_min_output,
            src_type=src_type,
            lookback=bb_lookback
        )
        
        # 計算済みのシグナルを保存
        self._macd_divergence_signals = None
        self._bb_signals = None
        self._entry_signals = None
        self._data_hash = None
        
        # シグナルをキャッシュするための辞書
        self._cached_signals = {}
    
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
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        シグナルを計算
        
        Args:
            data: 価格データ
        """
        # データハッシュの計算
        data_hash = self._get_data_hash(data)
        
        if data_hash == self._data_hash and self._entry_signals is not None:
            return
            
        self._data_hash = data_hash
        
        # ZMACDダイバージェンスシグナル計算
        self._macd_divergence_signals = self.z_macd_divergence_signal.generate(data)
        
        # ZBBブレイクアウトシグナル計算
        self._bb_signals = self.z_bb_signal.generate(data)
        
        # エントリーシグナル計算
        self._entry_signals = calculate_entry_signals(self._macd_divergence_signals)
        
        # シグナルをキャッシュに保存
        self._set_cached_signal('macd_divergence', self._macd_divergence_signals)
        self._set_cached_signal('bb_signals', self._bb_signals)
        self._set_cached_signal('entry', self._entry_signals)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        if self._entry_signals is None:
            self.calculate_signals(data)
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
        if self._macd_divergence_signals is None or self._bb_signals is None:
            self.calculate_signals(data)
            
        return check_exit_signal(
            position,
            index,
            self._macd_divergence_signals,
            self._bb_signals
        )
    
    def get_z_macd_values(self) -> Dict[str, np.ndarray]:
        """
        ZMACDの値を取得
        
        Returns:
            Dict[str, np.ndarray]: ZMACDの値（macd, signal, histogram）
        """
        if hasattr(self.z_macd_divergence_signal, 'get_z_macd_values'):
            return self.z_macd_divergence_signal.get_z_macd_values()
        return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
    
    def get_bb_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ボリンジャーバンドの値を取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if hasattr(self.z_bb_signal, 'get_bands'):
            return self.z_bb_signal.get_bands()
        return (np.array([]), np.array([]), np.array([]))
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Returns:
            np.ndarray: 効率比の値
        """
        if hasattr(self.z_macd_divergence_signal, 'get_efficiency_ratio'):
            return self.z_macd_divergence_signal.get_efficiency_ratio()
        return np.array([])
    
    def get_bb_dynamic_multiplier(self) -> np.ndarray:
        """
        BBの動的乗数を取得
        
        Returns:
            np.ndarray: 動的乗数の値
        """
        if hasattr(self.z_bb_signal, 'get_dynamic_multiplier'):
            return self.z_bb_signal.get_dynamic_multiplier()
        return np.array([])
    
    def _set_cached_signal(self, key: str, signal: np.ndarray) -> None:
        """
        シグナルをキャッシュに保存
        
        Args:
            key: シグナルのキー
            signal: シグナル配列
        """
        self._cached_signals[key] = signal
    
    def _get_cached_signal(self, key: str) -> np.ndarray:
        """
        キャッシュからシグナルを取得
        
        Args:
            key: シグナルのキー
            
        Returns:
            np.ndarray: シグナル配列
        """
        return self._cached_signals.get(key, np.array([])) 