#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit, jit, prange, vectorize, float64, int64

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource


@dataclass
class CycleDonchianResult:
    """サイクルドンチャンの計算結果"""
    upper: np.ndarray          # 上限バンド（サイクル期間の最高値）
    lower: np.ndarray          # 下限バンド（サイクル期間の最安値）
    middle: np.ndarray         # 中央線（(上限 + 下限) / 2）
    cycle_values: np.ndarray   # サイクル検出器の値
    cycle_periods: np.ndarray  # サイクル期間
    period_usage: np.ndarray   # 期間使用率（0-1、デバッグ用）


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_cycle_period_vec(cycle_value: float, max_period: float, min_period: float) -> float:
    """
    サイクル値に基づいて期間を計算する（ベクトル化&並列版）
    
    Args:
        cycle_value: サイクル検出器の値
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        計算された期間値
    """
    if np.isnan(cycle_value):
        return (max_period + min_period) / 2.0  # デフォルト値
    
    # サイクル値を期間範囲にクランプ
    clamped_cycle = min(max(cycle_value, min_period), max_period)
    
    return clamped_cycle


@njit(float64[:](float64[:], float64, float64), fastmath=True, parallel=True, cache=True)
def calculate_cycle_period_optimized(cycle_values: np.ndarray, max_period: float, min_period: float) -> np.ndarray:
    """
    サイクル値に基づいて期間を計算する（最適化&並列版）
    
    Args:
        cycle_values: サイクル検出器の値の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        計算された期間値の配列
    """
    result = np.empty_like(cycle_values)
    default_period = (max_period + min_period) / 2.0
    
    for i in prange(len(cycle_values)):
        if np.isnan(cycle_values[i]):
            result[i] = default_period
        else:
            # サイクル値を期間範囲にクランプ
            clamped_cycle = min(max(cycle_values[i], min_period), max_period)
            result[i] = clamped_cycle
    
    return result


@njit(fastmath=True, cache=True)
def calculate_cycle_donchian_single_period(
    high: np.ndarray, 
    low: np.ndarray, 
    period: int, 
    start_idx: int
) -> Tuple[float, float, float]:
    """
    単一の期間でドンチャンチャネルを計算する（最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        period: 期間
        start_idx: 計算開始インデックス
    
    Returns:
        Tuple[float, float, float]: 上限、下限、中央線の値
    """
    if start_idx < period - 1:
        return np.nan, np.nan, np.nan
    
    # 指定期間の最高値と最安値を計算
    max_high = high[start_idx - period + 1]
    min_low = low[start_idx - period + 1]
    
    for j in range(start_idx - period + 2, start_idx + 1):
        if high[j] > max_high:
            max_high = high[j]
        if low[j] < min_low:
            min_low = low[j]
    
    middle = (max_high + min_low) / 2.0
    return max_high, min_low, middle


@njit(fastmath=True, parallel=True, cache=True)
def calculate_cycle_donchian_optimized(
    high: np.ndarray,
    low: np.ndarray,
    cycle_periods: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    サイクルドンチャンチャネルを計算する（最適化&並列版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        cycle_periods: サイクル期間の配列
    
    Returns:
        上限、下限、中央線のタプル
    """
    length = len(high)
    
    # 結果用の配列を初期化
    upper = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    middle = np.empty(length, dtype=np.float64)
    
    # 並列処理で各時点でのドンチャンチャネルを計算
    for i in prange(length):
        # NaN値チェック
        if np.isnan(cycle_periods[i]):
            upper[i] = np.nan
            lower[i] = np.nan
            middle[i] = np.nan
            continue
        
        # サイクル期間を整数に変換
        period = int(round(cycle_periods[i]))
        period = max(1, period)  # 最小期間は1
        
        # 十分なデータがない場合はNaN
        if i < period - 1:
            upper[i] = np.nan
            lower[i] = np.nan
            middle[i] = np.nan
            continue
        
        # 指定期間の最高値と最安値を計算
        max_high = high[i - period + 1]
        min_low = low[i - period + 1]
        
        for j in range(i - period + 2, i + 1):
            if high[j] > max_high:
                max_high = high[j]
            if low[j] < min_low:
                min_low = low[j]
        
        upper[i] = max_high
        lower[i] = min_low
        middle[i] = (max_high + min_low) / 2.0
    
    return upper, lower, middle


class CycleDonchian(Indicator):
    """
    サイクルドンチャンインディケーター
    
    エラーズ統合サイクル検出器に基づいてドンチャンチャネルの期間を動的に決定します。
    - サイクル検出器が検出したサイクル期間をドンチャンチャネルの期間として使用
    - 期間範囲を制限可能（min_period〜max_period）
    
    特徴:
    - EhlersUnifiedDCによるサイクル期間検出
    - 設定可能な期間範囲
    - Numbaによる高速化
    - キャッシュシステム
    - z_adaptive_donchian.pyと同様のインターフェース
    """
    
    def __init__(
        self,
        # 期間の範囲制限
        min_period: int = 20,       # 最小期間
        max_period: int = 400,      # 最大期間
        
        # エラーズ統合DC用パラメータ
        detector_type: str = 'cycle_period2',
        cycle_part: float = 0.7,
        max_cycle: int = 377,
        min_cycle: int = 13,
        max_output: int = 233,
        min_output: int = 20,
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        lp_period: int = 5,
        hp_period: int = 144,
        # 新しい検出器用のパラメータ
        alpha: float = 0.07,
        bandwidth: float = 0.6,
        center_period: float = 15.0,
        avg_length: float = 3.0,
        window: int = 50
    ):
        """
        コンストラクタ
        
        Args:
            min_period: 最小期間（制限用）
            max_period: 最大期間（制限用）
            detector_type: エラーズ統合DC検出器タイプ
            cycle_part: サイクル部分の倍率
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルター使用有無
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            lp_period: ローパスフィルター期間（拡張検出器用）
            hp_period: ハイパスフィルター期間（拡張検出器用）
            alpha: アルファパラメータ（新しい検出器用）
            bandwidth: 帯域幅（新しい検出器用）
            center_period: 中心周期（新しい検出器用）
            avg_length: 平均長（新しい検出器用）
            window: 分析ウィンドウ長（新しい検出器用）
        """
        super().__init__(f"CycleDonchian(period={min_period}-{max_period},det={detector_type},src={src_type})")
        
        # 期間パラメータの検証と保存
        if min_period >= max_period:
            raise ValueError(f"min_period ({min_period}) は max_period ({max_period}) より小さくなければなりません")
        if min_period < 1:
            raise ValueError(f"min_period ({min_period}) は1以上でなければなりません")
        
        self.min_period = min_period
        self.max_period = max_period
        self.src_type = src_type
        
        # エラーズ統合DCパラメータの保存
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.avg_length = avg_length
        self.window = window
        
        # EhlersUnifiedDCの初期化
        self.ehlers_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_process_noise=kalman_process_noise,
            kalman_n_states=kalman_n_states,
            lp_period=lp_period,
            hp_period=hp_period,
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window
        )
        
        # PriceSourceユーティリティ
        self.price_source_extractor = PriceSource()
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を生成（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    # 最初と最後の値、およびデータ長を使用した軽量ハッシュ
                    first_high = float(data.iloc[0].get('high', data.iloc[0, 1] if data.shape[1] > 1 else 0))
                    first_low = float(data.iloc[0].get('low', data.iloc[0, 2] if data.shape[1] > 2 else 0))
                    last_high = float(data.iloc[-1].get('high', data.iloc[-1, 1] if data.shape[1] > 1 else 0))
                    last_low = float(data.iloc[-1].get('low', data.iloc[-1, 2] if data.shape[1] > 2 else 0))
                    data_signature = (length, first_high, first_low, last_high, last_low)
                else:
                    data_signature = (0, 0.0, 0.0, 0.0, 0.0)
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_high = float(data[0, 1])  # high
                        first_low = float(data[0, 2])   # low
                        last_high = float(data[-1, 1])
                        last_low = float(data[-1, 2])
                    else:
                        # 1次元配列の場合はcloseとして扱う
                        first_high = first_low = float(data[0])
                        last_high = last_low = float(data[-1])
                    data_signature = (length, first_high, first_low, last_high, last_low)
                else:
                    data_signature = (0, 0.0, 0.0, 0.0, 0.0)
            
            # パラメータの軽量シグネチャ
            params_signature = (
                self.min_period,
                self.max_period,
                self.detector_type,
                self.cycle_part,
                self.src_type
            )
            
            # 高速ハッシュ生成
            return f"{hash(data_signature)}_{hash(params_signature)}"
            
        except Exception:
            # フォールバック: 最小限のハッシュ
            return f"{id(data)}_{self.min_period}_{self.max_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクルドンチャンチャネルを計算（高速化版）
        
        Args:
            data: DataFrame または numpy 配列（OHLCデータが必要）
        
        Returns:
            np.ndarray: 中央線（middle）の値
        """
        try:
            # データハッシュを計算して、キャッシュが有効かどうかを確認
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash].middle
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'high' not in data.columns or 'low' not in data.columns:
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
            else:
                if data.ndim < 2 or data.shape[1] < 3:
                    raise ValueError("NumPy配列はOHLC形式（少なくとも3列）でなければなりません")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
            
            # データ長の検証
            data_length = len(high)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_array = np.array([])
                empty_result = CycleDonchianResult(
                    upper=empty_array, lower=empty_array, middle=empty_array,
                    cycle_values=empty_array, cycle_periods=empty_array, period_usage=empty_array
                )
                self._result_cache[data_hash] = empty_result
                return empty_array
            
            if data_length < self.min_period:
                self.logger.warning(f"データ長（{data_length}）が最小期間（{self.min_period}）より短いです。")
            
            # 1. エラーズ統合DCでサイクル値を計算
            cycle_values = self.ehlers_dc.calculate(data)
            
            # 2. サイクル値から期間を計算（制限範囲適用）
            cycle_periods = calculate_cycle_period_optimized(
                cycle_values, 
                float(self.max_period), 
                float(self.min_period)
            )
            
            # 期間使用率の計算（デバッグ用）
            period_range = self.max_period - self.min_period
            if period_range > 0:
                period_usage = (cycle_periods - self.min_period) / period_range
            else:
                period_usage = np.ones_like(cycle_periods) * 0.5
            
            # 3. サイクルドンチャンチャネルの計算（最適化版）
            upper, lower, middle = calculate_cycle_donchian_optimized(
                high, low, cycle_periods
            )
            
            # 結果をキャッシュ
            result = CycleDonchianResult(
                upper=upper,
                lower=lower,
                middle=middle,
                cycle_values=cycle_values,
                cycle_periods=cycle_periods,
                period_usage=period_usage
            )
            
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return middle  # 元のインターフェイスと互換性を保つため中央線を返す
            
        except Exception as e:
            self.logger.error(f"サイクルドンチャン計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ドンチャンチャネルバンドを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 上限、下限、中央線のタプル
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([]), np.array([]), np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.upper, result.lower, result.middle
        except Exception as e:
            self.logger.error(f"バンド取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> CycleDonchianResult:
        """
        詳細な計算結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            CycleDonchianResult: 詳細な計算結果
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                # 空の結果を返す
                empty_array = np.array([])
                return CycleDonchianResult(
                    upper=empty_array,
                    lower=empty_array,
                    middle=empty_array,
                    cycle_values=empty_array,
                    cycle_periods=empty_array,
                    period_usage=empty_array
                )
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result
        except Exception as e:
            self.logger.error(f"詳細結果取得中にエラー: {str(e)}")
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return CycleDonchianResult(
                upper=empty_array,
                lower=empty_array,
                middle=empty_array,
                cycle_values=empty_array,
                cycle_periods=empty_array,
                period_usage=empty_array
            )
    
    def get_cycle_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル期間を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル期間の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.cycle_periods
        except Exception as e:
            self.logger.error(f"サイクル期間取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル検出器の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル検出器の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.cycle_values
        except Exception as e:
            self.logger.error(f"サイクル値取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result_cache.clear()
        self._cache_keys.clear()
        if hasattr(self.ehlers_dc, 'reset'):
            self.ehlers_dc.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"CycleDonchian(period={self.min_period}-{self.max_period}, det={self.detector_type}, src={self.src_type})" 