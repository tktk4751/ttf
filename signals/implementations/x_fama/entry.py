#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.x_fama import X_FAMA


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    frama_values: np.ndarray, 
    fast_fama_values: np.ndarray
) -> np.ndarray:
    """
    X_FAMAとFast X_FAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        frama_values: X_FAMA値の配列
        fast_fama_values: Fast X_FAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(frama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # X_FAMA値とFast X_FAMA値が有効かチェック
        if np.isnan(frama_values[i]) or np.isnan(fast_fama_values[i]):
            signals[i] = 0
            continue
            
        # Fast X_FAMA > X_FRAMA: ロングシグナル（高速線が上位）
        if fast_fama_values[i] > frama_values[i]:
            signals[i] = 1
        # Fast X_FAMA < X_FRAMA: ショートシグナル（高速線が下位）
        elif fast_fama_values[i] < frama_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_crossover_signals(
    frama_values: np.ndarray, 
    fast_fama_values: np.ndarray
) -> np.ndarray:
    """
    X_FAMAとFast X_FAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        frama_values: X_FAMA値の配列
        fast_fama_values: Fast X_FAMA値の配列
    
    Returns:
        シグナルの配列（1: ゴールデンクロス, -1: デッドクロス, 0: シグナルなし）
    """
    length = len(frama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(frama_values[i]) or np.isnan(fast_fama_values[i]) or 
            np.isnan(frama_values[i-1]) or np.isnan(fast_fama_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_frama = frama_values[i-1]
        prev_fast_fama = fast_fama_values[i-1]
        
        # 現在の期間
        curr_frama = frama_values[i]
        curr_fast_fama = fast_fama_values[i]
        
        # ゴールデンクロス: 前期間でFast FAMA <= FRAMA、現期間でFast FAMA > FRAMA
        if prev_fast_fama <= prev_frama and curr_fast_fama > curr_frama:
            signals[i] = 1
        # デッドクロス: 前期間でFast FAMA >= FRAMA、現期間でFast FAMA < FRAMA
        elif prev_fast_fama >= prev_frama and curr_fast_fama < curr_frama:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_fractal_trend_signals(
    fractal_dimension: np.ndarray,
    alpha_values: np.ndarray,
    trend_threshold: float = 1.5,
    alpha_threshold: float = 0.5
) -> np.ndarray:
    """
    フラクタル次元とアルファ値に基づくトレンドシグナルを計算する
    
    Args:
        fractal_dimension: フラクタル次元配列
        alpha_values: アルファ値配列  
        trend_threshold: トレンド判定のフラクタル次元閾値
        alpha_threshold: アルファ値の閾値
    
    Returns:
        シグナルの配列（1: 強気, -1: 弱気, 0: ニュートラル）
    """
    length = len(fractal_dimension)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        if np.isnan(fractal_dimension[i]) or np.isnan(alpha_values[i]):
            signals[i] = 0
            continue
            
        # フラクタル次元が低く（トレンドが強い）、アルファ値が高い（応答性が高い）場合
        if fractal_dimension[i] < trend_threshold and alpha_values[i] > alpha_threshold:
            signals[i] = 1  # 強気トレンド
        # フラクタル次元が高く（ランダム性が高い）、アルファ値が低い（応答性が低い）場合
        elif fractal_dimension[i] > (2.0 - trend_threshold) and alpha_values[i] < (1.0 - alpha_threshold):
            signals[i] = -1  # 弱気（レンジ相場）
        else:
            signals[i] = 0  # ニュートラル
    
    return signals


class XFAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    X_FAMA/Fast X_FAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - X_FRAMA (eXtended Fractal Adaptive Moving Average) + Fast X_FRAMA
    - フラクタル次元に基づく適応型移動平均線
    - カルマンフィルターとゼロラグ処理統合版
    - 通常線と高速線の2本による精密なシグナル生成
    - フラクタル次元とアルファ値によるトレンド強度判定
    
    シグナル条件:
    - position_mode=True: Fast X_FRAMA > X_FRAMA: ロングシグナル (1), Fast X_FRAMA < X_FRAMA: ショートシグナル (-1)
    - position_mode=False: ゴールデンクロス: ロングシグナル (1), デッドクロス: ショートシグナル (-1)
    - fractal_mode=True: フラクタル次元とアルファ値に基づくトレンド判定を追加
    """
    
    def __init__(
        self,
        # X_FAMAパラメータ
        period: int = 16,                      # 期間（偶数である必要がある）
        src_type: str = 'hl2',                 # ソースタイプ
        fc: int = 1,                           # Fast Constant
        sc: int = 198,                         # Slow Constant
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = False,           # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        fractal_mode: bool = False,            # フラクタル次元ベースのシグナル追加
        trend_threshold: float = 1.5,          # フラクタル次元のトレンド閾値
        alpha_threshold: float = 0.5           # アルファ値の閾値
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            fractal_mode: フラクタル次元ベースのシグナル追加
            trend_threshold: フラクタル次元のトレンド閾値
            alpha_threshold: アルファ値の閾値
        """
        signal_type = "Position" if position_mode else "Crossover"
        fractal_str = "_fractal" if fractal_mode else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"X_FAMA{signal_type}EntrySignal(period={period}, {src_type}, fc={fc}, sc={sc}{fractal_str}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
        self._params = {
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
            'fractal_mode': fractal_mode,
            'trend_threshold': trend_threshold,
            'alpha_threshold': alpha_threshold
        }
        
        self.position_mode = position_mode
        self.fractal_mode = fractal_mode
        self.trend_threshold = trend_threshold
        self.alpha_threshold = alpha_threshold
        
        # X_FAMAインジケーターの初期化
        self.x_fama = X_FAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
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
                
            # X_FAMAの計算
            x_fama_result = self.x_fama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if x_fama_result is None or len(x_fama_result.frama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # X_FAMA値とFast X_FAMA値の取得
            frama_values = x_fama_result.frama_values
            fast_fama_values = x_fama_result.fast_fama_values
            fractal_dimension = x_fama_result.fractal_dimension
            alpha_values = x_fama_result.alpha_values
            
            # 基本シグナルの計算（位置関係またはクロスオーバー）
            if self.position_mode:
                # 位置関係シグナル
                signals = calculate_position_signals(
                    frama_values,
                    fast_fama_values
                )
            else:
                # クロスオーバーシグナル
                signals = calculate_crossover_signals(
                    frama_values,
                    fast_fama_values
                )
            
            # フラクタル次元ベースのシグナル追加（オプション）
            if self.fractal_mode:
                fractal_signals = calculate_fractal_trend_signals(
                    fractal_dimension,
                    alpha_values,
                    self.trend_threshold,
                    self.alpha_threshold
                )
                
                # フラクタルシグナルでフィルタリング
                # フラクタルシグナルが0でない場合のみ基本シグナルを通す
                filtered_signals = np.zeros_like(signals)
                for i in range(len(signals)):
                    if fractal_signals[i] != 0:
                        # フラクタルシグナルと基本シグナルが同方向の場合のみ採用
                        if (fractal_signals[i] > 0 and signals[i] > 0) or \
                           (fractal_signals[i] < 0 and signals[i] < 0):
                            filtered_signals[i] = signals[i]
                
                signals = filtered_signals
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"XFAMACrossoverEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_fama.get_frama_values()
    
    def get_fast_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Fast X_FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Fast X_FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_fama.get_fast_fama_values()
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元
        """
        if data is not None:
            self.generate(data)
            
        return self.x_fama.get_fractal_dimension()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファ値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルファ値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_fama.get_alpha_values()
    
    def get_filtered_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カルマンフィルター後の価格を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルタリングされた価格
        """
        if data is not None:
            self.generate(data)
            
        return self.x_fama.get_filtered_price()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.x_fama.reset() if hasattr(self.x_fama, 'reset') else None
        self._signals_cache = {}