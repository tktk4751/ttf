#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.laguerre_rsi.trend_reversal_entry import LaguerreRSITrendReversalEntrySignal


@njit(fastmath=True, parallel=False)
def laguerre_rsi_trend_reversal_exit_conditions(lrsi_signals: np.ndarray, position: int, index: int) -> bool:
    """
    Laguerre RSI トレンドリバーサル エグジット条件をチェック（Numba最適化版）
    
    Args:
        lrsi_signals: Laguerre RSIシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(lrsi_signals):
        return False
    
    signal = lrsi_signals[index]
    
    # ロングポジション: シグナル=-1でエグジット
    if position == 1 and signal == -1:
        return True
    
    # ショートポジション: シグナル=1でエグジット
    if position == -1 and signal == 1:
        return True
    
    return False


class LaguerreRSITrendReversalSignalGenerator(BaseSignalGenerator):
    """
    Laguerre RSI トレンドリバーサル シグナル生成クラス
    
    特徴:
    - Laguerre RSI（ラゲール変換ベースRSI）を使用したトレンドリバーサル戦略
    - John Ehlers's Laguerre transform filterによる高感度RSI
    - パインスクリプト仕様準拠のリバーサルシグナル生成
    
    エントリー条件:
    - ロング: RSI < buy_band (0.2) - 売られすぎからの反転狙い
    - ショート: RSI > sell_band (0.8) - 買われすぎからの反転狙い
    - position_mode=True: 閾値内では前回ポジション維持
    - position_mode=False: クロスオーバーでのみシグナル発生
    - mean_reversion_mode=True: 平均回帰モード（傾向変化を考慮）
    
    エグジット条件:
    - ロング: シグナル=-1
    - ショート: シグナル=1
    """
    
    def __init__(
        self,
        # Laguerre RSIパラメータ
        gamma: float = 0.5,                      # ガンマパラメータ
        src_type: str = 'close',                 # ソースタイプ
        # シグナル閾値（リバーサル用）
        buy_band: float = 0.2,                   # 買い閾値（売られすぎ水準）
        sell_band: float = 0.8,                  # 売り閾値（買われすぎ水準）
        # ルーフィングフィルターパラメータ（オプション）
        use_roofing_filter: bool = False,        # ルーフィングフィルターを使用するか
        roofing_hp_cutoff: float = 48.0,         # ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: float = 10.0,      # ルーフィングフィルターのSuperSmootherバンドエッジ
        # シグナル設定
        position_mode: bool = True,              # ポジション維持モード(True)またはクロスオーバーモード(False)
        mean_reversion_mode: bool = False        # 平均回帰モード（傾向変化を考慮）
    ):
        """
        初期化
        
        Args:
            gamma: ガンマパラメータ（デフォルト: 0.5）
            src_type: ソースタイプ（デフォルト: 'close'）
            buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
            sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
            use_roofing_filter: ルーフィングフィルター使用（デフォルト: False）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
            mean_reversion_mode: 平均回帰モード（傾向変化を考慮、デフォルト: False）
        """
        if mean_reversion_mode:
            signal_type = "MeanReversion"
        elif position_mode:
            signal_type = "Position"
        else:
            signal_type = "Crossover"
            
        roofing_str = f"_roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})" if use_roofing_filter else ""
        
        super().__init__(f"LaguerreRSI_TrendReversal_{signal_type}(gamma={gamma}, {src_type}{roofing_str})")
        
        # パラメータの設定
        self._params = {
            'gamma': gamma,
            'src_type': src_type,
            'buy_band': buy_band,
            'sell_band': sell_band,
            'use_roofing_filter': use_roofing_filter,
            'roofing_hp_cutoff': roofing_hp_cutoff,
            'roofing_ss_band_edge': roofing_ss_band_edge,
            'position_mode': position_mode,
            'mean_reversion_mode': mean_reversion_mode
        }
        
        self.position_mode = position_mode
        self.mean_reversion_mode = mean_reversion_mode
        
        # Laguerre RSI エントリーシグナルの初期化
        self.lrsi_entry_signal = LaguerreRSITrendReversalEntrySignal(
            gamma=gamma,
            src_type=src_type,
            buy_band=buy_band,
            sell_band=sell_band,
            use_roofing_filter=use_roofing_filter,
            roofing_hp_cutoff=roofing_hp_cutoff,
            roofing_ss_band_edge=roofing_ss_band_edge,
            position_mode=position_mode,
            mean_reversion_mode=mean_reversion_mode
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._lrsi_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._long_signals is None or current_len != self._data_len:
                # DataFrameの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # Laguerre RSIシグナルの計算
                    lrsi_signals = self.lrsi_entry_signal.generate(df)
                    self._lrsi_signals = lrsi_signals
                    
                    # ロング・ショートシグナルの分離
                    self._long_signals = (lrsi_signals == 1).astype(np.int8)
                    self._short_signals = (lrsi_signals == -1).astype(np.int8)
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._long_signals = np.zeros(current_len, dtype=np.int8)
                    self._short_signals = np.zeros(current_len, dtype=np.int8)
                    self._lrsi_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._long_signals = np.zeros(len(data), dtype=np.int8)
                self._short_signals = np.zeros(len(data), dtype=np.int8)
                self._lrsi_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得
        
        Returns:
            統合されたエントリーシグナル（ロング=1、ショート=-1、なし=0）
        """
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        # ロング・ショートシグナルを統合
        combined_signals = self._long_signals - self._short_signals
        return combined_signals
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._long_signals
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        if self._short_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._short_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._lrsi_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return laguerre_rsi_trend_reversal_exit_conditions(self._lrsi_signals, position, index)
    
    def get_lrsi_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Laguerre RSIシグナル取得"""
        if self._lrsi_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._lrsi_signals
    
    def get_lrsi_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Laguerre RSI値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_lrsi_values()
        except Exception as e:
            self.logger.error(f"Laguerre RSI値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l0_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L0値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_l0_values()
        except Exception as e:
            self.logger.error(f"L0値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L1値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_l1_values()
        except Exception as e:
            self.logger.error(f"L1値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l2_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L2値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_l2_values()
        except Exception as e:
            self.logger.error(f"L2値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l3_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L3値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_l3_values()
        except Exception as e:
            self.logger.error(f"L3値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cu_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """CU値（上昇累積）を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_cu_values()
        except Exception as e:
            self.logger.error(f"CU値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """CD値（下降累積）を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.lrsi_entry_signal.get_cd_values()
        except Exception as e:
            self.logger.error(f"CD値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            Laguerre RSIの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            metrics = {
                # Laguerre RSIメトリクス
                'lrsi_values': self.get_lrsi_values(),
                'l0_values': self.get_l0_values(),
                'l1_values': self.get_l1_values(),
                'l2_values': self.get_l2_values(),
                'l3_values': self.get_l3_values(),
                'cu_values': self.get_cu_values(),
                'cd_values': self.get_cd_values(),
                # シグナルメトリクス
                'long_signals': self.get_long_signals(data),
                'short_signals': self.get_short_signals(data),
                'lrsi_signals': self.get_lrsi_signals(data),
                # パラメータ情報
                'strategy_type': 'laguerre_rsi_trend_reversal',
                'position_mode': self.position_mode,
                'mean_reversion_mode': self.mean_reversion_mode
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._lrsi_signals = None
        
        if hasattr(self.lrsi_entry_signal, 'reset'):
            self.lrsi_entry_signal.reset()