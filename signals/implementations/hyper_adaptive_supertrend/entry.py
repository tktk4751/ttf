#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_adaptive_supertrend import HyperAdaptiveSupertrend


@njit(fastmath=True, parallel=True)
def calculate_trend_signals(
    supertrend_values: np.ndarray,
    close_values: np.ndarray,
    trend_values: np.ndarray
) -> np.ndarray:
    """
    ハイパーアダプティブスーパートレンドの位置関係シグナルを計算する（高速化版）
    
    Args:
        supertrend_values: スーパートレンドライン値の配列
        close_values: 終値の配列
        trend_values: トレンド方向の配列（1=上昇、-1=下降）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(supertrend_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係とトレンド方向を組み合わせた判定（並列処理化）
    for i in prange(length):
        # 値が有効かチェック
        if (np.isnan(supertrend_values[i]) or np.isnan(close_values[i]) or 
            trend_values[i] == 0):
            signals[i] = 0
            continue
            
        # 上昇トレンド（trend = 1）かつ価格がスーパートレンドライン上: ロングシグナル
        if trend_values[i] == 1 and close_values[i] > supertrend_values[i]:
            signals[i] = 1
        # 下降トレンド（trend = -1）かつ価格がスーパートレンドライン下: ショートシグナル
        elif trend_values[i] == -1 and close_values[i] < supertrend_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_trend_change_signals(
    trend_values: np.ndarray
) -> np.ndarray:
    """
    ハイパーアダプティブスーパートレンドのトレンド変化シグナルを計算する（高速化版）
    
    Args:
        trend_values: トレンド方向の配列（1=上昇、-1=下降）
    
    Returns:
        シグナルの配列（1: トレンド転換でロング, -1: トレンド転換でショート, 0: シグナルなし）
    """
    length = len(trend_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でトレンド変化を検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if trend_values[i] == 0 or trend_values[i-1] == 0:
            signals[i] = 0
            continue
            
        # 前の期間
        prev_trend = trend_values[i-1]
        
        # 現在の期間
        curr_trend = trend_values[i]
        
        # 下降→上昇トレンド転換: ロングシグナル
        if prev_trend == -1 and curr_trend == 1:
            signals[i] = 1
        # 上昇→下降トレンド転換: ショートシグナル
        elif prev_trend == 1 and curr_trend == -1:
            signals[i] = -1
    
    return signals


class HyperAdaptiveSupertrendEntrySignal(BaseSignal, IEntrySignal):
    """
    ハイパーアダプティブスーパートレンドによるエントリーシグナル
    
    特徴:
    - 最強のスーパートレンドインジケーター（unified_smoother + unscented_kalman_filter + x_atr）
    - 統合スムーサーによる高精度ミッドライン計算
    - カルマンフィルターによるノイズ除去（オプション）
    - X_ATRによる拡張ボラティリティ測定
    - 動的期間調整対応
    
    シグナル条件:
    - trend_change_mode=False: 価格とスーパートレンドラインの位置関係 + トレンド方向
      * 上昇トレンドかつ価格 > スーパートレンドライン: ロングシグナル (1)
      * 下降トレンドかつ価格 < スーパートレンドライン: ショートシグナル (-1)
    - trend_change_mode=True: トレンド変化によるシグナル
      * 下降→上昇トレンド転換: ロングシグナル (1)
      * 上昇→下降トレンド転換: ショートシグナル (-1)
    """
    
    def __init__(
        self,
        # ハイパーアダプティブスーパートレンドのパラメータ
        atr_period: int = 14,                     # X_ATR期間
        multiplier: float = 3.0,                  # ATR乗数
        atr_method: str = 'str',                  # X_ATRの計算方法
        atr_smoother_type: str = 'sma',           # X_ATRのスムーサータイプ
        midline_smoother_type: str = 'frama',     # ミッドラインスムーサータイプ
        midline_period: int = 21,                 # ミッドライン期間
        src_type: str = 'hlc3',                   # ソースタイプ
        # カルマンフィルターパラメータ
        enable_kalman: bool = True,               # カルマンフィルター使用フラグ
        kalman_alpha: float = 1.0,                # UKFアルファパラメータ
        kalman_beta: float = 2.0,                 # UKFベータパラメータ
        kalman_kappa: float = 0.0,                # UKFカッパパラメータ
        kalman_process_noise: float = 0.01,       # UKFプロセスノイズ
        # 動的期間パラメータ
        use_dynamic_period: bool = True,          # 動的期間を使用するか
        cycle_part: float = 0.5,                  # サイクル部分の倍率
        detector_type: str = 'hody_e',            # 検出器タイプ
        max_cycle: int = 124,                     # 最大サイクル期間
        min_cycle: int = 13,                      # 最小サイクル期間
        max_output: int = 124,                    # 最大出力値
        min_output: int = 13,                     # 最小出力値
        lp_period: int = 13,                      # ローパスフィルター期間
        hp_period: int = 124,                     # ハイパスフィルター期間
        # シグナル設定
        trend_change_mode: bool = False           # トレンド変化シグナル(True)または位置関係シグナル(False)
    ):
        """
        初期化
        
        Args:
            atr_period: X_ATR期間（デフォルト: 14）
            multiplier: ATR乗数（デフォルト: 3.0）
            atr_method: X_ATRの計算方法（'atr' または 'str'、デフォルト: 'str'）
            atr_smoother_type: X_ATRのスムーサータイプ（デフォルト: 'sma'）
            midline_smoother_type: ミッドラインスムーサータイプ（デフォルト: 'frama'）
            midline_period: ミッドライン期間（デフォルト: 21）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            enable_kalman: カルマンフィルター使用フラグ（デフォルト: True）
            kalman_alpha: UKFアルファパラメータ（デフォルト: 1.0）
            kalman_beta: UKFベータパラメータ（デフォルト: 2.0）
            kalman_kappa: UKFカッパパラメータ（デフォルト: 0.0）
            kalman_process_noise: UKFプロセスノイズ（デフォルト: 0.01）
            use_dynamic_period: 動的期間を使用するか（デフォルト: True）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            detector_type: 検出器タイプ（デフォルト: 'hody_e'）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 124）
            min_output: 最小出力値（デフォルト: 13）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            trend_change_mode: トレンド変化シグナル(True)または位置関係シグナル(False)
        """
        signal_type = "TrendChange" if trend_change_mode else "Position"
        kalman_str = f"_kalman({kalman_alpha},{kalman_beta},{kalman_kappa})" if enable_kalman else ""
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        
        super().__init__(
            f"HyperAdaptiveSupertrend{signal_type}EntrySignal("
            f"atr={atr_period}×{multiplier}_{atr_method}_{atr_smoother_type}, "
            f"mid={midline_period}_{midline_smoother_type}, "
            f"{src_type}{kalman_str}{dynamic_str})"
        )
        
        # パラメータの保存
        self._params = {
            'atr_period': atr_period,
            'multiplier': multiplier,
            'atr_method': atr_method,
            'atr_smoother_type': atr_smoother_type,
            'midline_smoother_type': midline_smoother_type,
            'midline_period': midline_period,
            'src_type': src_type,
            'enable_kalman': enable_kalman,
            'kalman_alpha': kalman_alpha,
            'kalman_beta': kalman_beta,
            'kalman_kappa': kalman_kappa,
            'kalman_process_noise': kalman_process_noise,
            'use_dynamic_period': use_dynamic_period,
            'cycle_part': cycle_part,
            'detector_type': detector_type,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'trend_change_mode': trend_change_mode
        }
        
        self.trend_change_mode = trend_change_mode
        
        # ハイパーアダプティブスーパートレンドインジケーターの初期化
        self.hyper_supertrend = HyperAdaptiveSupertrend(
            atr_period=atr_period,
            multiplier=multiplier,
            atr_method=atr_method,
            atr_smoother_type=atr_smoother_type,
            midline_smoother_type=midline_smoother_type,
            midline_period=midline_period,
            src_type=src_type,
            enable_kalman=enable_kalman,
            kalman_alpha=kalman_alpha,
            kalman_beta=kalman_beta,
            kalman_kappa=kalman_kappa,
            kalman_process_noise=kalman_process_noise,
            use_dynamic_period=use_dynamic_period,
            cycle_part=cycle_part,
            detector_type=detector_type,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period
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
                
            # ハイパーアダプティブスーパートレンドの計算
            supertrend_result = self.hyper_supertrend.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if supertrend_result is None or len(supertrend_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # 必要な値の取得
            supertrend_values = supertrend_result.values
            trend_values = supertrend_result.trend
            
            # 終値の取得
            if isinstance(data, pd.DataFrame):
                close_values = data['close'].values
            else:
                # NumPy配列の場合、通常は (n, 5) の形状で [open, high, low, close, volume]
                if data.ndim == 2 and data.shape[1] >= 4:
                    close_values = data[:, 3]  # close価格
                elif data.ndim == 1:
                    close_values = data
                else:
                    raise ValueError("データ形式が不正です")
            
            # シグナルの計算（位置関係またはトレンド変化）
            if self.trend_change_mode:
                # トレンド変化シグナル
                signals = calculate_trend_change_signals(trend_values)
            else:
                # 位置関係シグナル
                signals = calculate_trend_signals(
                    supertrend_values,
                    close_values,
                    trend_values
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperAdaptiveSupertrendEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_supertrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        スーパートレンドライン値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: スーパートレンドライン値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_values()
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        上側バンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 上側バンド値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_upper_band()
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        下側バンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 下側バンド値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_lower_band()
    
    def get_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向値（1=上昇、-1=下降）
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_supertrend_direction()
    
    def get_midline_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ミッドライン値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ミッドライン値（統合スムーサー結果）
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_midline()
    
    def get_atr_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATR値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_ATR値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_supertrend.get_atr_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.hyper_supertrend.reset() if hasattr(self.hyper_supertrend, 'reset') else None
        self._signals_cache = {}