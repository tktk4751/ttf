#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.x_mamacd.entry import (
    XMAMACDCrossoverEntrySignal,
    XMAMACDZeroLineEntrySignal,
    XMAMACDTrendFollowEntrySignal
)


class XMAMACDSignalGenerator(BaseSignalGenerator):
    """
    X_MAMACDシグナル生成クラス
    
    エントリー条件:
    - クロスオーバーモード: MAMACD > Signal に転換でロング、MAMACD < Signal に転換でショート
    - ゼロラインモード: MAMACD > 0 に転換でロング、MAMACD < 0 に転換でショート  
    - トレンドフォローモード: 複合条件でトレンド継続シグナル
    
    エグジット条件:
    - ロング: MAMACD < Signal に転換 または MAMACD < 0 に転換
    - ショート: MAMACD > Signal に転換 または MAMACD > 0 に転換
    """
    
    def __init__(
        self,
        # X_MAMACDパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        signal_period: int = 9,                # シグナルライン期間
        use_adaptive_signal: bool = True,      # 適応型シグナルラインを使用するか
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        signal_mode: str = 'crossover',        # シグナルモード ('crossover', 'zero_line', 'trend_follow')
        trend_threshold: float = 0.0,          # トレンドフォロー閾値
        momentum_mode: bool = False,           # モメンタムモードを使用するか
        momentum_lookback: int = 3             # モメンタム計算の振り返り期間
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            signal_period: シグナルライン期間（デフォルト: 9）
            use_adaptive_signal: 適応型シグナルラインを使用するか（デフォルト: True）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            signal_mode: シグナルモード（デフォルト: 'crossover'）
            trend_threshold: トレンドフォロー閾値（デフォルト: 0.0）
            momentum_mode: モメンタムモードを使用するか（デフォルト: False）
            momentum_lookback: モメンタム計算の振り返り期間（デフォルト: 3）
        """
        adaptive_str = "_adaptive" if use_adaptive_signal else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_MAMACD{signal_mode.title()}SignalGenerator{adaptive_str}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'signal_period': signal_period,
            'use_adaptive_signal': use_adaptive_signal,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'signal_mode': signal_mode,
            'trend_threshold': trend_threshold,
            'momentum_mode': momentum_mode,
            'momentum_lookback': momentum_lookback
        }
        
        self.signal_mode = signal_mode
        
        # X_MAMACDエントリーシグナルの初期化
        if signal_mode == 'crossover':
            self.x_mamacd_entry_signal = XMAMACDCrossoverEntrySignal(
                fast_limit=fast_limit,
                slow_limit=slow_limit,
                src_type=src_type,
                signal_period=signal_period,
                use_adaptive_signal=use_adaptive_signal,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type,
                kalman_process_noise=kalman_process_noise,
                kalman_observation_noise=kalman_observation_noise,
                use_zero_lag=use_zero_lag
            )
        elif signal_mode == 'zero_line':
            self.x_mamacd_entry_signal = XMAMACDZeroLineEntrySignal(
                fast_limit=fast_limit,
                slow_limit=slow_limit,
                src_type=src_type,
                signal_period=signal_period,
                use_adaptive_signal=use_adaptive_signal,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type,
                kalman_process_noise=kalman_process_noise,
                kalman_observation_noise=kalman_observation_noise,
                use_zero_lag=use_zero_lag
            )
        elif signal_mode == 'trend_follow':
            self.x_mamacd_entry_signal = XMAMACDTrendFollowEntrySignal(
                fast_limit=fast_limit,
                slow_limit=slow_limit,
                src_type=src_type,
                signal_period=signal_period,
                use_adaptive_signal=use_adaptive_signal,
                trend_threshold=trend_threshold,
                momentum_mode=momentum_mode,
                momentum_lookback=momentum_lookback,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type,
                kalman_process_noise=kalman_process_noise,
                kalman_observation_noise=kalman_observation_noise,
                use_zero_lag=use_zero_lag
            )
        else:
            raise ValueError(f"無効なシグナルモード: {signal_mode}。有効なオプション: 'crossover', 'zero_line', 'trend_follow'")
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._x_mamacd_signals = None
        self._mamacd_values = None
        self._signal_values = None
        self._histogram_values = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    if 'volume' in data.columns:
                        df = data[['open', 'high', 'low', 'close', 'volume']]
                    else:
                        df = data[['open', 'high', 'low', 'close']]
                        df['volume'] = 1000.0  # デフォルト値
                else:
                    if data.shape[1] >= 5:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                    else:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                        df['volume'] = 1000.0  # デフォルト値
                
                # X_MAMACDシグナルの計算
                try:
                    x_mamacd_signals = self.x_mamacd_entry_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = x_mamacd_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._x_mamacd_signals = x_mamacd_signals
                    
                    # MAMACD値も取得してキャッシュ
                    self._mamacd_values = self.x_mamacd_entry_signal.get_mamacd_values(df)
                    self._signal_values = self.x_mamacd_entry_signal.get_signal_values(df)
                    self._histogram_values = self.x_mamacd_entry_signal.get_histogram_values(df)
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._x_mamacd_signals = np.zeros(current_len, dtype=np.int8)
                    self._mamacd_values = np.zeros(current_len, dtype=np.float64)
                    self._signal_values = np.zeros(current_len, dtype=np.float64)
                    self._histogram_values = np.zeros(current_len, dtype=np.float64)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._x_mamacd_signals = np.zeros(len(data), dtype=np.int8)
                self._mamacd_values = np.zeros(len(data), dtype=np.float64)
                self._signal_values = np.zeros(len(data), dtype=np.float64)
                self._histogram_values = np.zeros(len(data), dtype=np.float64)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # インデックスの範囲チェック
        if index < 0 or index >= len(self._x_mamacd_signals):
            return False
        
        # モードに応じたエグジット判定
        if self.signal_mode == 'crossover':
            # クロスオーバーモードでは、逆方向のクロスオーバーが発生したらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_mamacd_signals[index] == -1)  # デッドクロス
            elif position == -1:  # ショートポジション
                return bool(self._x_mamacd_signals[index] == 1)   # ゴールデンクロス
                
        elif self.signal_mode == 'zero_line':
            # ゼロラインモードでは、逆方向のゼロラインクロスが発生したらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_mamacd_signals[index] == -1)  # ゼロライン下抜け
            elif position == -1:  # ショートポジション
                return bool(self._x_mamacd_signals[index] == 1)   # ゼロライン上抜け
                
        elif self.signal_mode == 'trend_follow':
            # トレンドフォローモードでは、逆方向のトレンドシグナルが発生したらエグジット
            if position == 1:  # ロングポジション
                return bool(self._x_mamacd_signals[index] == -1)  # ショートトレンド
            elif position == -1:  # ショートポジション
                return bool(self._x_mamacd_signals[index] == 1)   # ロングトレンド
        
        return False
    
    def get_mamacd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAMACD値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: MAMACD値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._mamacd_values if self._mamacd_values is not None else np.array([])
        except Exception as e:
            self.logger.error(f"MAMACD値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_signal_line_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        シグナルライン値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: シグナルライン値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._signal_values if self._signal_values is not None else np.array([])
        except Exception as e:
            self.logger.error(f"シグナルライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_histogram_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ヒストグラム値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ヒストグラム値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._histogram_values if self._histogram_values is not None else np.array([])
        except Exception as e:
            self.logger.error(f"ヒストグラム値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_MAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_MAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mamacd_entry_signal.x_mamacd.get_mama_values() if hasattr(self.x_mamacd_entry_signal, 'x_mamacd') else np.array([])
        except Exception as e:
            self.logger.error(f"X_MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_FAMA値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mamacd_entry_signal.x_mamacd.get_fama_values() if hasattr(self.x_mamacd_entry_signal, 'x_mamacd') else np.array([])
        except Exception as e:
            self.logger.error(f"X_FAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_period_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Period値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Period値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mamacd_entry_signal.x_mamacd.get_period_values() if hasattr(self.x_mamacd_entry_signal, 'x_mamacd') else np.array([])
        except Exception as e:
            self.logger.error(f"Period値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.x_mamacd_entry_signal.x_mamacd.get_alpha_values() if hasattr(self.x_mamacd_entry_signal, 'x_mamacd') else np.array([])
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: 全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return {
                'mamacd_values': self.get_mamacd_values(),
                'signal_line_values': self.get_signal_line_values(),
                'histogram_values': self.get_histogram_values(),
                'mama_values': self.get_mama_values(),
                'fama_values': self.get_fama_values(),
                'period_values': self.get_period_values(),
                'alpha_values': self.get_alpha_values()
            }
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """
        シグナルジェネレーターの状態をリセット
        """
        super().reset()
        self._data_len = 0
        self._signals = None
        self._x_mamacd_signals = None
        self._mamacd_values = None
        self._signal_values = None
        self._histogram_values = None
        if hasattr(self.x_mamacd_entry_signal, 'reset'):
            self.x_mamacd_entry_signal.reset()