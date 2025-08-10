#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.unified_smoother import UnifiedSmoother
from .volatility.x_atr import XATR
from .cycle.ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class KeltnerChannelResult:
    """ケルトナーチャネルの計算結果"""
    midline_values: np.ndarray       # ミッドライン値（統合スムーサー）
    upper_channel: np.ndarray        # 上部チャネル
    lower_channel: np.ndarray        # 下部チャネル
    atr_values: np.ndarray          # ATR値
    dynamic_period: np.ndarray       # 動的期間
    multiplier_values: np.ndarray    # 乗数値
    bandwidth: np.ndarray           # バンド幅
    position: np.ndarray            # 価格位置（-1〜1）


@njit(fastmath=True, cache=True)
def calculate_keltner_channel_core(
    midline: np.ndarray,
    atr_values: np.ndarray,
    multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ケルトナーチャネルのコア計算（Numba最適化版）
    
    Args:
        midline: ミッドライン値
        atr_values: ATR値
        multiplier: 乗数
    
    Returns:
        Tuple[np.ndarray, ...]: 上部チャネル, 下部チャネル, バンド幅, 位置
    """
    length = len(midline)
    upper_channel = np.zeros(length, dtype=np.float64)
    lower_channel = np.zeros(length, dtype=np.float64)
    bandwidth = np.zeros(length, dtype=np.float64)
    position = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(midline[i]) and not np.isnan(atr_values[i]):
            # チャネル計算
            atr_band = atr_values[i] * multiplier
            upper_channel[i] = midline[i] + atr_band
            lower_channel[i] = midline[i] - atr_band
            
            # バンド幅計算
            bandwidth[i] = (upper_channel[i] - lower_channel[i]) / midline[i] * 100.0 if midline[i] != 0 else 0.0
            
            # 価格位置計算（-1〜1の範囲）
            if atr_band > 0:
                position[i] = 0.0  # midlineとの相対位置、価格データが必要
            else:
                position[i] = 0.0
        else:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
            bandwidth[i] = np.nan
            position[i] = np.nan
    
    return upper_channel, lower_channel, bandwidth, position


@njit(fastmath=True, cache=True)
def calculate_price_position(
    price: np.ndarray,
    midline: np.ndarray,
    upper_channel: np.ndarray,
    lower_channel: np.ndarray
) -> np.ndarray:
    """
    価格位置を計算（Numba最適化版）
    
    Args:
        price: 価格配列
        midline: ミッドライン
        upper_channel: 上部チャネル
        lower_channel: 下部チャネル
    
    Returns:
        価格位置の配列（-1〜1）
    """
    length = len(price)
    position = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if (not np.isnan(price[i]) and not np.isnan(midline[i]) and 
            not np.isnan(upper_channel[i]) and not np.isnan(lower_channel[i])):
            
            channel_width = upper_channel[i] - lower_channel[i]
            if channel_width > 0:
                # -1（下部チャネル）から1（上部チャネル）の範囲で正規化
                position[i] = 2.0 * (price[i] - lower_channel[i]) / channel_width - 1.0
                
                # 範囲を制限
                if position[i] > 1.0:
                    position[i] = 1.0
                elif position[i] < -1.0:
                    position[i] = -1.0
            else:
                position[i] = 0.0
        else:
            position[i] = np.nan
    
    return position


@njit(fastmath=True, cache=True)
def apply_kalman_filter_simple(
    price: np.ndarray,
    measurement_noise: float = 1.0,
    process_noise: float = 0.01
) -> np.ndarray:
    """
    シンプルなカルマンフィルター（Numba最適化版）
    
    Args:
        price: 価格配列
        measurement_noise: 測定ノイズ
        process_noise: プロセスノイズ
    
    Returns:
        フィルター済み価格配列
    """
    length = len(price)
    filtered = np.zeros(length, dtype=np.float64)
    
    if length == 0:
        return filtered
    
    # 初期化
    state = price[0] if not np.isnan(price[0]) else 0.0
    variance = 1.0
    
    for i in range(length):
        if not np.isnan(price[i]):
            # 予測ステップ
            predicted_variance = variance + process_noise
            
            # 更新ステップ
            kalman_gain = predicted_variance / (predicted_variance + measurement_noise)
            state = state + kalman_gain * (price[i] - state)
            variance = (1.0 - kalman_gain) * predicted_variance
            
            filtered[i] = state
        else:
            filtered[i] = state  # 前の値を保持
    
    return filtered


class XKeltnerChannel(Indicator):
    """
    ケルトナーチャネル（Keltner Channel）インジケーター
    
    統合的な処理フロー:
    1. ソース価格 → エラーズサイクル統合ファイルで動的期間計算
    2. → カルマンフィルター（オプション）
    3. → ミッドライン計算（固定期間か動的期間を選択可能）
    4. → X_ATRでATR計算
    5. → ケルトナーチャネル計算
    
    特徴:
    - 動的期間対応（EhlersUnifiedDC統合）が最優先
    - オプションのカルマンフィルター前処理
    - 適応的ミッドライン（統合スムーサー使用、期間選択可能）
    - 動的ATR計算（X_ATR使用）
    - 高精度のバンド計算とポジション分析
    """
    
    def __init__(
        self,
        period: int = 20,                           # 基本期間
        multiplier: float = 2.0,                    # ATR乗数
        src_type: str = 'hlc3',                     # ソースタイプ
        use_kalman_filter: bool = True,             # カルマンフィルター使用
        kalman_measurement_noise: float = 1.0,      # カルマン測定ノイズ
        kalman_process_noise: float = 0.01,         # カルマンプロセスノイズ
        # 動的期間パラメータ
        use_dynamic_period: bool = True,            # 動的期間使用
        dc_detector_type: str = 'hody_e',           # サイクル検出器タイプ
        dc_min_period: int = 6,                     # 最小サイクル期間
        dc_max_period: int = 50,                    # 最大サイクル期間
        # 統合スムーサーパラメータ
        smoother_type: str = 'frama',               # スムーサータイプ
        smoother_period: int = 20,                  # スムーサー期間
        use_dynamic_smoother_period: bool = True,   # スムーサーで動的期間使用
        # X_ATRパラメータ
        atr_period: int = 14,                       # ATR期間
        atr_smoothing: str = 'zlema',   # ATRスムージング
        adaptive_multiplier: bool = False           # 適応的乗数
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本期間（デフォルト: 20）
            multiplier: ATR乗数（デフォルト: 2.0）
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルター使用（デフォルト: True）
            kalman_measurement_noise: カルマン測定ノイズ（デフォルト: 1.0）
            kalman_process_noise: カルマンプロセスノイズ（デフォルト: 0.01）
            use_dynamic_period: 動的期間使用（デフォルト: True）
            dc_detector_type: サイクル検出器タイプ（デフォルト: 'hody_e'）
            dc_min_period: 最小サイクル期間（デフォルト: 6）
            dc_max_period: 最大サイクル期間（デフォルト: 50）
            smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 20）
            use_dynamic_smoother_period: スムーサーで動的期間使用（デフォルト: True）
            atr_period: ATR期間（デフォルト: 14）
            atr_smoothing: ATRスムージングタイプ（デフォルト: 'sma'）
            adaptive_multiplier: 適応的乗数（デフォルト: False）
        """
        # インジケーター名の作成
        indicator_name = (f"KeltnerChannel(period={period}, mult={multiplier}, {src_type}, "
                         f"dynamic={'Y' if use_dynamic_period else 'N'}, "
                         f"kalman={'Y' if use_kalman_filter else 'N'}, "
                         f"smoother={smoother_type}{'_dyn' if use_dynamic_smoother_period else '_fix'}, "
                         f"atr={atr_period})")
        super().__init__(indicator_name)
        
        # パラメータの保存（処理順序に従って）
        self.period = period
        self.multiplier = multiplier
        self.src_type = src_type.lower()
        # 動的期間パラメータ
        self.use_dynamic_period = use_dynamic_period
        self.dc_detector_type = dc_detector_type
        self.dc_min_period = dc_min_period
        self.dc_max_period = dc_max_period
        # カルマンフィルターパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        # スムーサーパラメータ
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.use_dynamic_smoother_period = use_dynamic_smoother_period
        # ATRパラメータ
        self.atr_period = atr_period
        self.atr_smoothing = atr_smoothing
        self.adaptive_multiplier = adaptive_multiplier
        
        # パラメータ検証
        if period <= 0:
            raise ValueError("期間は正の値である必要があります")
        if multiplier <= 0:
            raise ValueError("乗数は正の値である必要があります")
        if atr_period <= 0:
            raise ValueError("ATR期間は正の値である必要があります")
        if smoother_period <= 0:
            raise ValueError("スムーサー期間は正の値である必要があります")
        
        # PriceSourceユーティリティ
        self.price_source_extractor = PriceSource()
        
        # サブインジケーターの初期化（処理順序に従って）
        # 1. EhlersUnifiedDC（動的期間用）- 最初に初期化
        if use_dynamic_period:
            self.cycle_detector = EhlersUnifiedDC(
                detector_type=dc_detector_type,
                min_cycle=dc_min_period,
                max_cycle=dc_max_period,
                src_type=src_type,
                use_kalman_filter=False  # 二重フィルタリングを避ける
            )
        else:
            self.cycle_detector = None
        
        # 2. 統合スムーサー（期間は後で動的に設定）
        self.smoother = UnifiedSmoother(
            smoother_type=smoother_type,
            period=smoother_period,
            src_type=src_type
        )
        
        # 3. X_ATR
        self.x_atr = XATR(
            period=atr_period,
            smoother_type=atr_smoothing,
            src_type='close'  # X_ATRはcloseソースを使用
        )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _calculate_dynamic_midline(self, filtered_price: np.ndarray, dynamic_periods: np.ndarray, original_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的期間を使用してミッドラインを計算する
        
        Args:
            filtered_price: フィルター済み価格配列
            dynamic_periods: 動的期間配列
            original_data: 元のデータ（スムーサーが他のソースタイプを必要とする場合）
        
        Returns:
            ミッドライン値の配列
        """
        try:
            length = len(filtered_price)
            midline = np.zeros(length, dtype=np.float64)
            
            # 各時点で動的期間を使用してスムージング
            for i in range(length):
                current_period = int(dynamic_periods[i])
                
                # 期間の開始点を計算
                start_idx = max(0, i - current_period + 1)
                
                if i >= current_period - 1:
                    # 十分なデータがある場合
                    price_window = filtered_price[start_idx:i+1]
                    
                    # スムーサータイプに応じた計算
                    if self.smoother_type.lower() == 'sma':
                        midline[i] = np.mean(price_window)
                    elif self.smoother_type.lower() == 'ema':
                        # 指数移動平均の動的計算
                        alpha = 2.0 / (current_period + 1)
                        ema_val = price_window[0]
                        for price in price_window[1:]:
                            ema_val = alpha * price + (1 - alpha) * ema_val
                        midline[i] = ema_val
                    elif self.smoother_type.lower() == 'frama':
                        # FRAMA使用時は統合スムーサーを動的期間で再計算
                        # 一時的にスムーサーの期間を変更
                        temp_smoother = UnifiedSmoother(
                            smoother_type=self.smoother_type,
                            period=current_period,
                            src_type=self.src_type
                        )
                        
                        # 元データの必要な部分を取得
                        if isinstance(original_data, pd.DataFrame):
                            temp_data = original_data.iloc[start_idx:i+1].copy()
                            if self.src_type == 'close':
                                temp_data['close'] = price_window
                        else:
                            if original_data.ndim == 2 and original_data.shape[1] >= 4:
                                temp_data = pd.DataFrame(original_data[start_idx:i+1], columns=['open', 'high', 'low', 'close'])
                                temp_data['close'] = price_window
                            else:
                                temp_data = pd.DataFrame({'close': price_window})
                        
                        try:
                            smoother_result = temp_smoother.calculate(temp_data)
                            smoothed_values = smoother_result.values if hasattr(smoother_result, 'values') else smoother_result
                            midline[i] = smoothed_values[-1] if len(smoothed_values) > 0 else filtered_price[i]
                        except:
                            # フォールバック: SMAを使用
                            midline[i] = np.mean(price_window)
                    else:
                        # その他のスムーサータイプは統合スムーサーを使用
                        try:
                            temp_smoother = UnifiedSmoother(
                                smoother_type=self.smoother_type,
                                period=current_period,
                                src_type=self.src_type
                            )
                            
                            # 元データの必要な部分を取得
                            if isinstance(original_data, pd.DataFrame):
                                temp_data = original_data.iloc[start_idx:i+1].copy()
                                if self.src_type == 'close':
                                    temp_data['close'] = price_window
                            else:
                                if original_data.ndim == 2 and original_data.shape[1] >= 4:
                                    temp_data = pd.DataFrame(original_data[start_idx:i+1], columns=['open', 'high', 'low', 'close'])
                                    temp_data['close'] = price_window
                                else:
                                    temp_data = pd.DataFrame({'close': price_window})
                            
                            smoother_result = temp_smoother.calculate(temp_data)
                            smoothed_values = smoother_result.values if hasattr(smoother_result, 'values') else smoother_result
                            midline[i] = smoothed_values[-1] if len(smoothed_values) > 0 else filtered_price[i]
                        except:
                            # フォールバック: SMAを使用
                            midline[i] = np.mean(price_window)
                else:
                    # データが不足している場合は利用可能なデータで計算
                    if i > 0:
                        midline[i] = np.mean(filtered_price[:i+1])
                    else:
                        midline[i] = filtered_price[i]
            
            return midline
            
        except Exception as e:
            self.logger.warning(f"動的ミッドライン計算エラー: {e}。単純移動平均を使用します。")
            # フォールバック: 固定期間SMA
            from pandas import Series
            return Series(filtered_price).rolling(window=self.smoother_period, min_periods=1).mean().values
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータ情報
            params_sig = (f"{self.period}_{self.multiplier}_{self.src_type}_"
                         f"{self.use_dynamic_period}_{self.dc_detector_type}_{self.dc_min_period}_{self.dc_max_period}_"
                         f"{self.use_kalman_filter}_{self.kalman_measurement_noise}_{self.kalman_process_noise}_"
                         f"{self.smoother_type}_{self.smoother_period}_{self.use_dynamic_smoother_period}_"
                         f"{self.atr_period}_{self.atr_smoothing}_{self.adaptive_multiplier}")
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.multiplier}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KeltnerChannelResult:
        """
        ケルトナーチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + ソースタイプに必要なカラムが必要
        
        Returns:
            KeltnerChannelResult: ケルトナーチャネルの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュから取得
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return KeltnerChannelResult(
                    midline_values=cached_result.midline_values.copy(),
                    upper_channel=cached_result.upper_channel.copy(),
                    lower_channel=cached_result.lower_channel.copy(),
                    atr_values=cached_result.atr_values.copy(),
                    dynamic_period=cached_result.dynamic_period.copy(),
                    multiplier_values=cached_result.multiplier_values.copy(),
                    bandwidth=cached_result.bandwidth.copy(),
                    position=cached_result.position.copy()
                )
            
            # 新しい処理フロー:
            # 1. ソース価格の計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < max(self.period, self.atr_period, self.smoother_period):
                self.logger.warning(f"データが短すぎます（{data_length}点）。")
            
            # 2. エラーズサイクル統合ファイルで動的期間計算
            if self.use_dynamic_period and self.cycle_detector is not None:
                try:
                    dynamic_period_values = self.cycle_detector.calculate(data)
                    # NaN値を基本期間で置き換え
                    dynamic_period_values = np.where(
                        np.isnan(dynamic_period_values), 
                        self.period, 
                        dynamic_period_values
                    )
                    # 期間を整数に変換し、範囲を制限
                    dynamic_period_values = np.clip(
                        np.round(dynamic_period_values).astype(int), 
                        self.dc_min_period, 
                        self.dc_max_period
                    )
                except Exception as e:
                    self.logger.warning(f"動的期間計算エラー: {e}。固定期間を使用します。")
                    dynamic_period_values = np.full(data_length, self.period)
            else:
                dynamic_period_values = np.full(data_length, self.period)
            
            # 3. カルマンフィルター前処理（オプション）
            if self.use_kalman_filter:
                filtered_price = apply_kalman_filter_simple(
                    price_source, 
                    self.kalman_measurement_noise, 
                    self.kalman_process_noise
                )
            else:
                filtered_price = price_source.copy()
            
            # 4. ミッドライン計算（固定期間か動的期間を選択可能）
            if self.use_dynamic_smoother_period:
                # 動的期間を使用してミッドラインを計算
                midline_values = self._calculate_dynamic_midline(filtered_price, dynamic_period_values, data)
            else:
                # 固定期間でミッドライン計算
                # フィルター済み価格を一時的にDataFrameに変換してスムーサーに渡す
                if isinstance(data, pd.DataFrame):
                    # 元のDataFrameの構造を保持し、ソース価格のみ置き換え
                    temp_data = data.copy()
                    if self.src_type == 'close':
                        temp_data['close'] = filtered_price
                    elif self.src_type == 'hlc3':
                        # HLC3の場合、元のH,L,Cを保持したまま計算させる
                        pass  # スムーサーが内部でHLC3を計算
                    # その他のソースタイプにも対応可能
                else:
                    # NumPy配列の場合、DataFrameに変換
                    if data.ndim == 2 and data.shape[1] >= 4:
                        temp_data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                        if self.src_type == 'close':
                            temp_data['close'] = filtered_price
                    else:
                        # 1次元配列の場合
                        temp_data = pd.DataFrame({'close': filtered_price})
                
                smoother_result = self.smoother.calculate(temp_data)
                midline_values = smoother_result.values if hasattr(smoother_result, 'values') else smoother_result
            
            # 5. X_ATRによるATR計算
            atr_result = self.x_atr.calculate(data)
            atr_values = atr_result.values if hasattr(atr_result, 'values') else atr_result
            
            # 配列長を統一
            min_length = min(len(midline_values), len(atr_values), len(dynamic_period_values))
            midline_values = midline_values[:min_length]
            atr_values = atr_values[:min_length]
            dynamic_period_values = dynamic_period_values[:min_length]
            
            # 6. ケルトナーチャネル計算
            # 適応的乗数の計算（オプション）
            if self.adaptive_multiplier:
                # 動的期間に基づいて乗数を調整
                multiplier_values = self.multiplier * (dynamic_period_values / self.period)
                multiplier_values = np.clip(multiplier_values, self.multiplier * 0.5, self.multiplier * 2.0)
            else:
                multiplier_values = np.full(min_length, self.multiplier)
            # 各点で動的な乗数を適用
            upper_channel = np.zeros(min_length, dtype=np.float64)
            lower_channel = np.zeros(min_length, dtype=np.float64)
            bandwidth = np.zeros(min_length, dtype=np.float64)
            
            for i in range(min_length):
                if not np.isnan(midline_values[i]) and not np.isnan(atr_values[i]):
                    atr_band = atr_values[i] * multiplier_values[i]
                    upper_channel[i] = midline_values[i] + atr_band
                    lower_channel[i] = midline_values[i] - atr_band
                    
                    # バンド幅計算（パーセンテージ）
                    if midline_values[i] != 0:
                        bandwidth[i] = (upper_channel[i] - lower_channel[i]) / midline_values[i] * 100.0
                    else:
                        bandwidth[i] = 0.0
                else:
                    upper_channel[i] = np.nan
                    lower_channel[i] = np.nan
                    bandwidth[i] = np.nan
            
            # 7. 価格位置の計算
            # 元の価格（フィルターなし）を使用
            price_for_position = price_source[:min_length]
            position = calculate_price_position(
                price_for_position, midline_values, upper_channel, lower_channel
            )
            
            # 結果の作成
            result = KeltnerChannelResult(
                midline_values=midline_values.copy(),
                upper_channel=upper_channel.copy(),
                lower_channel=lower_channel.copy(),
                atr_values=atr_values.copy(),
                dynamic_period=dynamic_period_values.astype(np.float64).copy(),
                multiplier_values=multiplier_values.copy(),
                bandwidth=bandwidth.copy(),
                position=position.copy()
            )
            
            # キャッシュの更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = midline_values  # 基底クラスの要件
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"KeltnerChannel計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = KeltnerChannelResult(
                midline_values=np.array([]),
                upper_channel=np.array([]),
                lower_channel=np.array([]),
                atr_values=np.array([]),
                dynamic_period=np.array([]),
                multiplier_values=np.array([]),
                bandwidth=np.array([]),
                position=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.midline_values.copy()
    
    def get_midline_values(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得する"""
        return self.get_values()
    
    def get_upper_channel(self) -> Optional[np.ndarray]:
        """上部チャネル値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.upper_channel.copy()
    
    def get_lower_channel(self) -> Optional[np.ndarray]:
        """下部チャネル値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.lower_channel.copy()
    
    def get_atr_values(self) -> Optional[np.ndarray]:
        """ATR値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.atr_values.copy()
    
    def get_dynamic_period(self) -> Optional[np.ndarray]:
        """動的期間値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.dynamic_period.copy()
    
    def get_bandwidth(self) -> Optional[np.ndarray]:
        """バンド幅値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.bandwidth.copy()
    
    def get_position(self) -> Optional[np.ndarray]:
        """価格位置を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.position.copy()
    
    def get_channel_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """チャネルデータ（ミッドライン、上部、下部）を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return (result.midline_values.copy(), 
                result.upper_channel.copy(), 
                result.lower_channel.copy())
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        
        # サブインジケーターのリセット
        if hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        if hasattr(self.x_atr, 'reset'):
            self.x_atr.reset()
        if self.cycle_detector and hasattr(self.cycle_detector, 'reset'):
            self.cycle_detector.reset()