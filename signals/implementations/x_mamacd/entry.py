#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.x_mamacd import X_MAMACD


@njit(fastmath=True, parallel=True)
def calculate_mamacd_crossover_signals(
    mamacd_values: np.ndarray, 
    signal_values: np.ndarray
) -> np.ndarray:
    """
    MAMACDとシグナルラインのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        mamacd_values: MAMACD値の配列
        signal_values: シグナルライン値の配列
    
    Returns:
        シグナルの配列（1: ゴールデンクロス, -1: デッドクロス, 0: シグナルなし）
    """
    length = len(mamacd_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(mamacd_values[i]) or np.isnan(signal_values[i]) or 
            np.isnan(mamacd_values[i-1]) or np.isnan(signal_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_mamacd = mamacd_values[i-1]
        prev_signal = signal_values[i-1]
        
        # 現在の期間
        curr_mamacd = mamacd_values[i]
        curr_signal = signal_values[i]
        
        # ゴールデンクロス: 前期間でMAMACD <= Signal、現期間でMAMACD > Signal
        if prev_mamacd <= prev_signal and curr_mamacd > curr_signal:
            signals[i] = 1
        # デッドクロス: 前期間でMAMACD >= Signal、現期間でMAMACD < Signal
        elif prev_mamacd >= prev_signal and curr_mamacd < curr_signal:
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_mamacd_zero_line_signals(
    mamacd_values: np.ndarray
) -> np.ndarray:
    """
    MAMACDのゼロラインクロスシグナルを計算する（高速化版）
    
    Args:
        mamacd_values: MAMACD値の配列
    
    Returns:
        シグナルの配列（1: ゼロライン上抜け, -1: ゼロライン下抜け, 0: シグナルなし）
    """
    length = len(mamacd_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でゼロラインクロスを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if np.isnan(mamacd_values[i]) or np.isnan(mamacd_values[i-1]):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_mamacd = mamacd_values[i-1]
        
        # 現在の期間
        curr_mamacd = mamacd_values[i]
        
        # ゼロライン上抜け: 前期間でMAMACD <= 0、現期間でMAMACD > 0
        if prev_mamacd <= 0.0 and curr_mamacd > 0.0:
            signals[i] = 1
        # ゼロライン下抜け: 前期間でMAMACD >= 0、現期間でMAMACD < 0
        elif prev_mamacd >= 0.0 and curr_mamacd < 0.0:
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_mamacd_trend_signals(
    mamacd_values: np.ndarray,
    signal_values: np.ndarray,
    histogram_values: np.ndarray,
    trend_threshold: float = 0.0
) -> np.ndarray:
    """
    MAMACDトレンドフォローシグナルを計算する（高速化版）
    
    Args:
        mamacd_values: MAMACD値の配列
        signal_values: シグナルライン値の配列
        histogram_values: ヒストグラム値の配列
        trend_threshold: トレンド判定の閾値
    
    Returns:
        シグナルの配列（1: ロングトレンド, -1: ショートトレンド, 0: シグナルなし）
    """
    length = len(mamacd_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # トレンドフォローシグナルの条件（並列処理化）
    for i in prange(length):
        # 値が有効かチェック
        if (np.isnan(mamacd_values[i]) or np.isnan(signal_values[i]) or 
            np.isnan(histogram_values[i])):
            signals[i] = 0
            continue
            
        mamacd = mamacd_values[i]
        signal = signal_values[i]
        histogram = histogram_values[i]
        
        # ロングトレンドの条件
        # 1. MAMACD > Signal （上昇トレンド）
        # 2. MAMACD > 閾値 （強いトレンド）
        # 3. Histogram > 0 （勢いが継続）
        if (mamacd > signal and 
            mamacd > trend_threshold and 
            histogram > 0.0):
            signals[i] = 1
            
        # ショートトレンドの条件
        # 1. MAMACD < Signal （下降トレンド）
        # 2. MAMACD < -閾値 （強いトレンド）
        # 3. Histogram < 0 （勢いが継続）
        elif (mamacd < signal and 
              mamacd < -trend_threshold and 
              histogram < 0.0):
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_mamacd_momentum_signals(
    mamacd_values: np.ndarray,
    signal_values: np.ndarray,
    histogram_values: np.ndarray,
    momentum_lookback: int = 3
) -> np.ndarray:
    """
    MAMACDモメンタムシグナルを計算する（高速化版）
    
    Args:
        mamacd_values: MAMACD値の配列
        signal_values: シグナルライン値の配列
        histogram_values: ヒストグラム値の配列
        momentum_lookback: モメンタム計算の振り返り期間
    
    Returns:
        シグナルの配列（1: ロングモメンタム, -1: ショートモメンタム, 0: シグナルなし）
    """
    length = len(mamacd_values)
    signals = np.zeros(length, dtype=np.int8)
    
    if momentum_lookback <= 0:
        momentum_lookback = 3
    
    # モメンタムシグナルの計算
    for i in range(momentum_lookback, length):
        # 必要な値が有効かチェック
        valid = True
        for j in range(momentum_lookback + 1):
            if (np.isnan(mamacd_values[i-j]) or np.isnan(signal_values[i-j]) or 
                np.isnan(histogram_values[i-j])):
                valid = False
                break
        
        if not valid:
            signals[i] = 0
            continue
        
        # 現在の値
        curr_mamacd = mamacd_values[i]
        curr_signal = signal_values[i]
        curr_histogram = histogram_values[i]
        
        # 過去の値
        past_mamacd = mamacd_values[i - momentum_lookback]
        past_signal = signal_values[i - momentum_lookback]
        past_histogram = histogram_values[i - momentum_lookback]
        
        # ロングモメンタムの条件
        # 1. MAMACD > Signal （現在上昇トレンド）
        # 2. MAMACD上昇中 （momentum_lookback期間前より高い）
        # 3. Histogram改善中 （momentum_lookback期間前より高い）
        if (curr_mamacd > curr_signal and 
            curr_mamacd > past_mamacd and 
            curr_histogram > past_histogram):
            signals[i] = 1
            
        # ショートモメンタムの条件
        # 1. MAMACD < Signal （現在下降トレンド）
        # 2. MAMACD下降中 （momentum_lookback期間前より低い）
        # 3. Histogram悪化中 （momentum_lookback期間前より低い）
        elif (curr_mamacd < curr_signal and 
              curr_mamacd < past_mamacd and 
              curr_histogram < past_histogram):
            signals[i] = -1
    
    return signals


class XMAMACDCrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    X_MAMACDクロスオーバーによるエントリーシグナル
    
    特徴:
    - X_MAMACD（X_MAMAベースのMACD）を使用
    - MAMACDとシグナルラインのクロスオーバーでエントリー判定
    - 市場のサイクルに応じて自動的に期間を調整する適応型MACD
    - カルマンフィルターとゼロラグ処理統合版
    
    シグナル条件:
    - ゴールデンクロス: MAMACD > Signal になった時点でロングシグナル (1)
    - デッドクロス: MAMACD < Signal になった時点でショートシグナル (-1)
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
        use_zero_lag: bool = True              # ゼロラグ処理を使用するか
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
        """
        adaptive_str = "_adaptive" if use_adaptive_signal else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"X_MAMACDCrossoverEntrySignal(fast={fast_limit}, slow={slow_limit}, {src_type}, signal={signal_period}{adaptive_str}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
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
            'use_zero_lag': use_zero_lag
        }
        
        # X_MAMACDインジケーターの初期化
        self.x_mamacd = X_MAMACD(
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
                
            # X_MAMACDの計算
            x_mamacd_result = self.x_mamacd.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if x_mamacd_result is None or len(x_mamacd_result.mamacd) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # MAMACD、Signal、Histogram値の取得
            mamacd_values = x_mamacd_result.mamacd
            signal_values = x_mamacd_result.signal
            
            # クロスオーバーシグナルの計算
            signals = calculate_mamacd_crossover_signals(
                mamacd_values,
                signal_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"XMAMACDCrossoverEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            return np.zeros(len(data), dtype=np.int8)
    
    def get_mamacd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAMACD値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: MAMACD値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mamacd.get_mamacd_values()
    
    def get_signal_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        シグナルライン値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: シグナルライン値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mamacd.get_signal_values()
    
    def get_histogram_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ヒストグラム値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ヒストグラム値
        """
        if data is not None:
            self.generate(data)
            
        return self.x_mamacd.get_histogram_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.x_mamacd.reset() if hasattr(self.x_mamacd, 'reset') else None
        self._signals_cache = {}


class XMAMACDZeroLineEntrySignal(BaseSignal, IEntrySignal):
    """
    X_MAMACDゼロラインクロスによるエントリーシグナル
    
    特徴:
    - MAMACDのゼロラインクロスでエントリー判定
    - より強いトレンド転換シグナル
    - ノイズが少ない
    
    シグナル条件:
    - ゼロライン上抜け: MAMACD > 0 になった時点でロングシグナル (1)
    - ゼロライン下抜け: MAMACD < 0 になった時点でショートシグナル (-1)
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
        use_zero_lag: bool = True              # ゼロラグ処理を使用するか
    ):
        """初期化"""
        adaptive_str = "_adaptive" if use_adaptive_signal else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"X_MAMACDZeroLineEntrySignal(fast={fast_limit}, slow={slow_limit}, {src_type}, signal={signal_period}{adaptive_str}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
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
            'use_zero_lag': use_zero_lag
        }
        
        # X_MAMACDインジケーターの初期化
        self.x_mamacd = X_MAMACD(
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
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """データハッシュを取得する"""
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
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
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # X_MAMACDの計算
            x_mamacd_result = self.x_mamacd.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if x_mamacd_result is None or len(x_mamacd_result.mamacd) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # MAMACD値の取得
            mamacd_values = x_mamacd_result.mamacd
            
            # ゼロラインクロスシグナルの計算
            signals = calculate_mamacd_zero_line_signals(mamacd_values)
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            print(f"XMAMACDZeroLineEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
        
    def reset(self) -> None:
        """シグナルの状態をリセットする"""
        super().reset()
        self.x_mamacd.reset() if hasattr(self.x_mamacd, 'reset') else None
        self._signals_cache = {}


class XMAMACDTrendFollowEntrySignal(BaseSignal, IEntrySignal):
    """
    X_MAMACDトレンドフォローエントリーシグナル
    
    特徴:
    - 複数の条件を組み合わせたトレンドフォローシグナル
    - ノイズの少ない強いトレンドのみを捕捉
    - モメンタム要素も考慮
    
    シグナル条件:
    - ロングトレンド: MAMACD > Signal & MAMACD > 閾値 & Histogram > 0
    - ショートトレンド: MAMACD < Signal & MAMACD < -閾値 & Histogram < 0
    """
    
    def __init__(
        self,
        # X_MAMACDパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        signal_period: int = 9,                # シグナルライン期間
        use_adaptive_signal: bool = True,      # 適応型シグナルラインを使用するか
        # トレンドフォロー設定
        trend_threshold: float = 0.0,          # トレンド判定の閾値
        momentum_mode: bool = False,           # モメンタムモードを使用するか
        momentum_lookback: int = 3,            # モメンタム計算の振り返り期間
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True              # ゼロラグ処理を使用するか
    ):
        """初期化"""
        adaptive_str = "_adaptive" if use_adaptive_signal else ""
        momentum_str = f"_momentum({momentum_lookback})" if momentum_mode else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"X_MAMACDTrendFollowEntrySignal(fast={fast_limit}, slow={slow_limit}, {src_type}, signal={signal_period}, threshold={trend_threshold}{adaptive_str}{momentum_str}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'signal_period': signal_period,
            'use_adaptive_signal': use_adaptive_signal,
            'trend_threshold': trend_threshold,
            'momentum_mode': momentum_mode,
            'momentum_lookback': momentum_lookback,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag
        }
        
        self.trend_threshold = trend_threshold
        self.momentum_mode = momentum_mode
        self.momentum_lookback = momentum_lookback
        
        # X_MAMACDインジケーターの初期化
        self.x_mamacd = X_MAMACD(
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
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """データハッシュを取得する"""
        if isinstance(ohlcv_data, pd.DataFrame):
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
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
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # X_MAMACDの計算
            x_mamacd_result = self.x_mamacd.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if x_mamacd_result is None or len(x_mamacd_result.mamacd) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # MAMACD、Signal、Histogram値の取得
            mamacd_values = x_mamacd_result.mamacd
            signal_values = x_mamacd_result.signal
            histogram_values = x_mamacd_result.histogram
            
            # トレンドフォローまたはモメンタムシグナルの計算
            if self.momentum_mode:
                signals = calculate_mamacd_momentum_signals(
                    mamacd_values,
                    signal_values,
                    histogram_values,
                    self.momentum_lookback
                )
            else:
                signals = calculate_mamacd_trend_signals(
                    mamacd_values,
                    signal_values,
                    histogram_values,
                    self.trend_threshold
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            print(f"XMAMACDTrendFollowEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
        
    def reset(self) -> None:
        """シグナルの状態をリセットする"""
        super().reset()
        self.x_mamacd.reset() if hasattr(self.x_mamacd, 'reset') else None
        self._signals_cache = {}