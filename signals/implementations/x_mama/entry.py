#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.x_mama import X_MAMA


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    mama_values: np.ndarray, 
    fama_values: np.ndarray
) -> np.ndarray:
    """
    X_MAMAとX_FAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        mama_values: X_MAMA値の配列
        fama_values: X_FAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(mama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # X_MAMA値とX_FAMA値が有効かチェック
        if np.isnan(mama_values[i]) or np.isnan(fama_values[i]):
            signals[i] = 0
            continue
            
        # X_MAMA > X_FAMA: ロングシグナル
        if mama_values[i] > fama_values[i]:
            signals[i] = 1
        # X_MAMA < X_FAMA: ショートシグナル
        elif mama_values[i] < fama_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_crossover_signals(
    mama_values: np.ndarray, 
    fama_values: np.ndarray
) -> np.ndarray:
    """
    X_MAMAとX_FAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        mama_values: X_MAMA値の配列
        fama_values: X_FAMA値の配列
    
    Returns:
        シグナルの配列（1: ゴールデンクロス, -1: デッドクロス, 0: シグナルなし）
    """
    length = len(mama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(mama_values[i]) or np.isnan(fama_values[i]) or 
            np.isnan(mama_values[i-1]) or np.isnan(fama_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_mama = mama_values[i-1]
        prev_fama = fama_values[i-1]
        
        # 現在の期間
        curr_mama = mama_values[i]
        curr_fama = fama_values[i]
        
        # ゴールデンクロス: 前期間でMAMA <= FAMA、現期間でMAMA > FAMA
        if prev_mama <= prev_fama and curr_mama > curr_fama:
            signals[i] = 1
        # デッドクロス: 前期間でMAMA >= FAMA、現期間でMAMA < FAMA
        elif prev_mama >= prev_fama and curr_mama < curr_fama:
            signals[i] = -1
    
    return signals


class XMAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    X_MAMA/X_FAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - X_MAMA (eXtended Mother of Adaptive Moving Average) / X_FAMA (eXtended Following Adaptive Moving Average)
    - 市場のサイクルに応じて自動的に期間を調整する適応型移動平均線
    - カルマンフィルターとゼロラグ処理統合版
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    - トレンド強度に応じて応答速度を調整
    
    シグナル条件:
    - position_mode=True: X_MAMA > X_FAMA: ロングシグナル (1), X_MAMA < X_FAMA: ショートシグナル (-1)
    - position_mode=False: ゴールデンクロス: ロングシグナル (1), デッドクロス: ショートシグナル (-1)
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = False            # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        """
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"X_MAMA{signal_type}EntrySignal(fast={fast_limit}, slow={slow_limit}, {src_type}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode
        }
        
        self.position_mode = position_mode
        
        # X_MAMAインジケーターの初期化
        self.x_mama = X_MAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # X_MAMAの計算
            x_mama_result = self.x_mama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if x_mama_result is None or len(x_mama_result.mama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # X_MAMA値とX_FAMA値の取得
            mama_values = x_mama_result.mama_values
            fama_values = x_mama_result.fama_values
            
            # シグナルの計算（位置関係またはクロスオーバー）
            if self.position_mode:
                # 位置関係シグナル
                signals = calculate_position_signals(
                    mama_values,
                    fama_values
                )
            else:
                # クロスオーバーシグナル
                signals = calculate_crossover_signals(
                    mama_values,
                    fama_values
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"XMAMACrossoverEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_MAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_MAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_mama_values()
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_FAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_fama_values()
    
    def get_period_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Period値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Period値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_period_values()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_alpha_values()
    
    def get_phase_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Phase値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Phase値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_phase_values()
    
    def get_i1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        I1値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: I1値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_i1_values()
    
    def get_q1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Q1値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Q1値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mama.get_q1_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.x_mama.reset() if hasattr(self.x_mama, 'reset') else None
        self._signals_cache = {}