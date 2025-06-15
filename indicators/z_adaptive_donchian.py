#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit, jit, prange, vectorize, float64, int64

from .indicator import Indicator
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .price_source import PriceSource


@dataclass
class ZAdaptiveDonchianResult:
    """Zアダプティブドンチャンの計算結果"""
    upper: np.ndarray          # 上限バンド（動的期間の最高値）
    lower: np.ndarray          # 下限バンド（動的期間の最安値）
    middle: np.ndarray         # 中央線（(上限 + 下限) / 2）
    er_values: np.ndarray      # サイクル効率比の値
    dynamic_periods: np.ndarray # 動的に調整された期間
    period_usage: np.ndarray   # 期間使用率（0-1、デバッグ用）


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_period_vec(er_value: float, max_period: float, min_period: float) -> float:
    """
    サイクル効率比に基づいて動的期間を計算する（ベクトル化&並列版）
    
    Args:
        er_value: サイクル効率比の値（絶対値）
        max_period: 最大期間（CERが低い時に使用）
        min_period: 最小期間（CERが高い時に使用）
    
    Returns:
        動的期間値（整数として扱うが、計算は浮動小数点で行う）
    """
    # CERは負の値も取り得るため、絶対値を使用
    abs_er = abs(er_value) if not np.isnan(er_value) else 0.0
    
    # CER値を0-1の範囲にクランプ（通常CERは0-1の範囲だが、安全のため）
    clamped_er = min(max(abs_er, 0.0), 1.0)
    
    # CERが高いほど短い期間を使用する逆比例関係
    # period = max_period - er_value * (max_period - min_period)
    period = max_period - clamped_er * (max_period - min_period)
    
    return period


@njit(float64[:](float64[:], float64, float64), fastmath=True, parallel=True, cache=True)
def calculate_dynamic_period_optimized(er_values: np.ndarray, max_period: float, min_period: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的期間を計算する（最適化&並列版）
    
    Args:
        er_values: サイクル効率比の値の配列
        max_period: 最大期間（CERが低い時に使用）
        min_period: 最小期間（CERが高い時に使用）
    
    Returns:
        動的期間値の配列
    """
    result = np.empty_like(er_values)
    period_range = max_period - min_period
    
    for i in prange(len(er_values)):
        # CERは負の値も取り得るため、絶対値を使用
        abs_er = abs(er_values[i]) if not np.isnan(er_values[i]) else 0.0
        
        # CER値を0-1の範囲にクランプ
        clamped_er = min(max(abs_er, 0.0), 1.0)
        
        # CERが高いほど短い期間を使用する逆比例関係
        period = max_period - clamped_er * period_range
        
        result[i] = period
    
    return result


@njit(fastmath=True, cache=True)
def calculate_adaptive_donchian_single_period(
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
def calculate_z_adaptive_donchian_optimized(
    high: np.ndarray,
    low: np.ndarray,
    dynamic_periods: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zアダプティブドンチャンチャネルを計算する（最適化&並列版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        dynamic_periods: 動的期間の配列
    
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
        if np.isnan(dynamic_periods[i]):
            upper[i] = np.nan
            lower[i] = np.nan
            middle[i] = np.nan
            continue
        
        # 動的期間を整数に変換
        period = int(round(dynamic_periods[i]))
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


class ZAdaptiveDonchian(Indicator):
    """
    Zアダプティブドンチャンインディケーター
    
    サイクル効率比（CER）に基づいてドンチャンチャネルの期間を動的に調整します。
    - CERが高い（効率的なトレンド）→ 短い期間（20に近い）
    - CERが低い（非効率的・レンジ相場）→ 長い期間（300に近い）
    
    特徴:
    - CycleEfficiencyRatioによる動的期間調整
    - 20-300の期間範囲
    - Numbaによる高速化
    - キャッシュシステム
    - z_adaptive_channel.pyと同様のインターフェース
    """
    
    def __init__(
        self,
        # 期間の範囲
        min_period: int = 20,      # 最小期間（CERが高い時）
        max_period: int = 300,     # 最大期間（CERが低い時）
        
        # CERパラメータ
        detector_type: str = 'cycle_period2',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        smooth_er: bool = True,
        er_alma_period: int = 5,
        er_alma_offset: float = 0.85,
        er_alma_sigma: float = 6,
        self_adaptive: bool = False,
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
            min_period: 最小期間（CERが高い時に使用）
            max_period: 最大期間（CERが低い時に使用）
            detector_type: CER用ドミナントサイクル検出器タイプ
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            cycle_part: CER用サイクル部分
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: CER用カルマンフィルター使用有無
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            smooth_er: 効率比にALMAスムージングを適用するかどうか
            er_alma_period: ALMAスムージングの期間
            er_alma_offset: ALMAスムージングのオフセット
            er_alma_sigma: ALMAスムージングのシグマ
            self_adaptive: セルフアダプティブモードを有効にするかどうか
            alpha: アルファパラメータ（新しい検出器用）
            bandwidth: 帯域幅（新しい検出器用）
            center_period: 中心周期（新しい検出器用）
            avg_length: 平均長（新しい検出器用）
            window: 分析ウィンドウ長（新しい検出器用）
        """
        super().__init__(f"ZAdaptiveDonchian(period={min_period}-{max_period},det={detector_type},src={src_type})")
        
        # 期間パラメータの検証と保存
        if min_period >= max_period:
            raise ValueError(f"min_period ({min_period}) は max_period ({max_period}) より小さくなければなりません")
        if min_period < 1:
            raise ValueError(f"min_period ({min_period}) は1以上でなければなりません")
        
        self.min_period = min_period
        self.max_period = max_period
        self.src_type = src_type
        
        # CERパラメータの保存
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.smooth_er = smooth_er
        self.er_alma_period = er_alma_period
        self.er_alma_offset = er_alma_offset
        self.er_alma_sigma = er_alma_sigma
        self.self_adaptive = self_adaptive
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.avg_length = avg_length
        self.window = window
        
        # CycleEfficiencyRatioの初期化
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
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
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma,
            self_adaptive=self_adaptive,
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
        Zアダプティブドンチャンチャネルを計算（高速化版）
        
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
                empty_result = ZAdaptiveDonchianResult(
                    upper=empty_array, lower=empty_array, middle=empty_array,
                    er_values=empty_array, dynamic_periods=empty_array, period_usage=empty_array
                )
                self._result_cache[data_hash] = empty_result
                return empty_array
            
            if data_length < self.min_period:
                self.logger.warning(f"データ長（{data_length}）が最小期間（{self.min_period}）より短いです。")
            
            # 1. サイクル効率比（CER）の計算
            er_values = self.cycle_er.calculate(data)
            
            # 2. 動的期間の計算（並列化・ベクトル化版）
            dynamic_periods = calculate_dynamic_period_optimized(
                er_values, 
                float(self.max_period), 
                float(self.min_period)
            )
            
            # 期間使用率の計算（デバッグ用）
            period_range = self.max_period - self.min_period
            period_usage = (self.max_period - dynamic_periods) / period_range if period_range > 0 else np.zeros_like(dynamic_periods)
            
            # 3. Zアダプティブドンチャンチャネルの計算（最適化版）
            upper, lower, middle = calculate_z_adaptive_donchian_optimized(
                high, low, dynamic_periods
            )
            
            # 結果をキャッシュ
            result = ZAdaptiveDonchianResult(
                upper=upper,
                lower=lower,
                middle=middle,
                er_values=er_values,
                dynamic_periods=dynamic_periods,
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
            self.logger.error(f"Zアダプティブドンチャン計算中にエラー: {str(e)}")
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
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> ZAdaptiveDonchianResult:
        """
        詳細な計算結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            ZAdaptiveDonchianResult: 詳細な計算結果
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                # 空の結果を返す
                empty_array = np.array([])
                return ZAdaptiveDonchianResult(
                    upper=empty_array,
                    lower=empty_array,
                    middle=empty_array,
                    er_values=empty_array,
                    dynamic_periods=empty_array,
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
            return ZAdaptiveDonchianResult(
                upper=empty_array,
                lower=empty_array,
                middle=empty_array,
                er_values=empty_array,
                dynamic_periods=empty_array,
                period_usage=empty_array
            )
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的に調整された期間を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的期間の値
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
                
            return result.dynamic_periods
        except Exception as e:
            self.logger.error(f"動的期間取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_er_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
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
                
            return result.er_values
        except Exception as e:
            self.logger.error(f"ER値取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result_cache.clear()
        self._cache_keys.clear()
        if hasattr(self.cycle_er, 'reset'):
            self.cycle_er.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"ZAdaptiveDonchian(period={self.min_period}-{self.max_period}, det={self.detector_type}, src={self.src_type})" 