#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Literal
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, int64, boolean

from .indicator import Indicator
from .price_source import PriceSource
from .z_adaptive_channel import (
    ZAdaptiveChannel,
    calculate_dynamic_multiplier_vec,
    calculate_dynamic_max_multiplier,
    calculate_dynamic_min_multiplier
)
from .c_atr import CATR
from .z_adaptive_ma import ZAdaptiveMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZAdaptiveTrendResult:
    """Zアダプティブトレンドの計算結果"""
    middle: np.ndarray        # 中心線（ZAdaptiveMA）
    upper_band: np.ndarray    # 上限バンド（下降トレンド時のみ表示）
    lower_band: np.ndarray    # 下限バンド（上昇トレンド時のみ表示）
    trend: np.ndarray         # トレンド方向（1:上昇、-1:下降）
    er: np.ndarray            # Efficiency Ratio (CER)
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    z_atr: np.ndarray         # CATR値
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_adaptive_trend_optimized(
    z_ma: np.ndarray,
    z_atr: np.ndarray,
    dynamic_multiplier: np.ndarray,
    src: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zアダプティブトレンドを計算する（最適化&並列版）
    
    Args:
        z_ma: ZAdaptiveMAの配列（中心線）
        z_atr: CATRの配列（ボラティリティ測定・金額ベース）
        dynamic_multiplier: 動的乗数の配列
        src: 価格データのソース（トレンド判定に使用）
    
    Returns:
        中心線、上限バンド、下限バンド、トレンド方向のタプル
    """
    length = len(z_ma)
    
    # 結果用の配列を初期化
    middle = z_ma  # 直接参照で最適化（コピー不要）
    upper_band = np.empty(length, dtype=np.float64)
    lower_band = np.empty(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # まず基本的なバンドを計算
    final_upper_band = np.empty(length, dtype=np.float64)
    final_lower_band = np.empty(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(z_ma[i]) or np.isnan(z_atr[i]) or np.isnan(dynamic_multiplier[i]):
            final_upper_band[i] = np.nan
            final_lower_band[i] = np.nan
            continue
            
        # 基本的なバンド幅（SuperTrendと同様にz_maを基準点として使用）
        band_width = z_atr[i] * dynamic_multiplier[i]
        final_upper_band[i] = z_ma[i] + band_width
        final_lower_band[i] = z_ma[i] - band_width
    
    # 最初の値を設定
    if length > 0 and not np.isnan(src[0]) and not np.isnan(final_upper_band[0]):
        trend[0] = 1 if src[0] > final_upper_band[0] else -1
    
        # 最初のバンド値
        if trend[0] == 1:
            # 上昇トレンドの場合、下限バンド（サポートライン）のみを表示
            upper_band[0] = np.nan
            lower_band[0] = final_lower_band[0]
        else:
            # 下降トレンドの場合、上限バンド（レジスタンスライン）のみを表示
            upper_band[0] = final_upper_band[0]
            lower_band[0] = np.nan
    else:
        upper_band[0] = np.nan
        lower_band[0] = np.nan
    
    # バンドとトレンドの計算
    for i in range(1, length):
        if np.isnan(src[i]) or np.isnan(final_upper_band[i-1]) or np.isnan(final_lower_band[i-1]):
            trend[i] = 0  # データがない場合はトレンドなし
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            continue
            
        # トレンド判定
        if src[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif src[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定
        if trend[i] == 1:
            # 上昇トレンドの場合、下限バンド（サポートライン）のみを表示
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            # 下降トレンドの場合、上限バンド（レジスタンスライン）のみを表示
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return middle, upper_band, lower_band, trend


class ZAdaptiveTrend(Indicator):
    """
    ZAdaptiveTrend（Zアダプティブトレンド）インディケーター
    
    特徴:
    - ZAdaptiveChannelとSuperTrendの組み合わせ
    - 効率比(CER)に基づいて動的に調整されるチャネル幅
    - ZAdaptiveMAを中心線として使用
    - CATRを使用したボラティリティベースのバンド
    - トレンド判定機能：上昇/下降トレンドを自動判定
    - トレンドに基づいたバンド表示（上昇トレンドは下限バンドのみ、下降トレンドは上限バンドのみ）
    
    使用方法:
    - トレンドの方向性判定
    - 動的なサポート/レジスタンスレベルの特定
    - トレンドの方向性とボラティリティに基づくエントリー/エグジット
    - 効率比を使用したトレンド分析
    """
    
    def __init__(
        self,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        src_type: str = 'close',      # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CERパラメータ
        detector_type: str = 'dudi_e',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.4,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 100,              # CER用ハイパスフィルター期間
        max_cycle: int = 120,              # CER用最大サイクル期間
        min_cycle: int = 10,               # CER用最小サイクル期間
        max_output: int = 75,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        use_kalman_filter: bool = False,   # CER用カルマンフィルター使用有無
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 30             # 遅い移動平均の期間（固定値）
    ):
        """
        コンストラクタ
        
        Args:
            max_max_multiplier: 最大乗数の最大値（動的乗数使用時）
            min_max_multiplier: 最大乗数の最小値（動的乗数使用時）
            max_min_multiplier: 最小乗数の最大値（動的乗数使用時）
            min_min_multiplier: 最小乗数の最小値（動的乗数使用時）
            src_type: ソースタイプ（トレンド判定に使用されるプライスソース）
            
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
        """
        super().__init__(f"ZAdaptiveTrend(CER,{max_max_multiplier},{cycle_part},{src_type})")
        
        # パラメータの保存
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        self.src_type = src_type
        
        # 依存オブジェクトの初期化
        # 1. CycleEfficiencyRatio
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            src_type=src_type
        )
        
        # 2. ZAdaptiveMA
        self._z_adaptive_ma = ZAdaptiveMA(fast_period=fast_period, slow_period=slow_period)
        
        # 3. CATR
        self._c_atr = CATR()
        
        # 4. PriceSource
        self._price_source = PriceSource()
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 10  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を生成（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # DataFrameの場合はサイズと最初と最後の値のみを使用
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            # 最初と最後の10行のみ使用（大きなデータセットの場合も高速）
            if len(data) > 20:
                first_last = (
                    tuple(data.iloc[0].values) + 
                    tuple(data.iloc[-1].values) +
                    (data.shape[0],)  # データの長さも含める
                )
            else:
                # 小さなデータセットはすべて使用
                first_last = tuple(data.values.flatten()[-20:])
        else:
            shape = data.shape
            # NumPy配列も同様
            if len(data) > 20:
                if data.ndim > 1:
                    first_last = tuple(data[0]) + tuple(data[-1]) + (data.shape[0],)
                else:
                    first_last = (data[0], data[-1], data.shape[0])
            else:
                first_last = tuple(data.flatten()[-20:])
            
        # パラメータとサイズ、データのサンプルを組み合わせたハッシュを返す
        params_str = (
            f"{self.src_type}_{self.max_max_multiplier}_{self.min_max_multiplier}_"
            f"{self.max_min_multiplier}_{self.min_min_multiplier}_ZAT"
        )
        
        return f"{params_str}_{hash(first_last + (shape,))}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zアダプティブトレンドを計算（高速化版）
        
        Args:
            data: DataFrame または numpy 配列
        
        Returns:
            np.ndarray: 中心線（ZAdaptiveMA）の値
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
            
            # 1. 効率比の計算（CER）
            er = self.cycle_er.calculate(data)
            
            # 2. 動的な乗数値の計算（並列化・ベクトル化版）
            # 2.1. 動的な最大乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                er, 
                self.max_max_multiplier, 
                self.min_max_multiplier
            )
            
            # 2.2. 動的な最小乗数の計算
            min_mult_values = calculate_dynamic_min_multiplier(
                er, 
                self.max_min_multiplier, 
                self.min_min_multiplier
            )
            
            # 2.3. 線形計算を使用して最終的な動的乗数を計算
            dynamic_multiplier = calculate_dynamic_multiplier_vec(er, max_mult_values, min_mult_values)
            
            # 3. ZAdaptiveMAの計算（中心線）
            z_ma = self._z_adaptive_ma.calculate(data, er)
            
            # 4. CATRの計算
            self._c_atr.calculate(data, er)
            
            # 金額ベースのCATRを取得 - 重要: バンド計算には金額ベースのATRを使用する
            z_atr = self._c_atr.get_absolute_atr()
            
            # 5. トレンド判定用のソース価格を取得
            src = self._price_source.get_source(data, self.src_type)
            
            # 6. Zアダプティブトレンドの計算（中心線、上限バンド、下限バンド、トレンド） - 最適化版
            middle, upper_band, lower_band, trend = calculate_z_adaptive_trend_optimized(
                z_ma,
                z_atr,
                dynamic_multiplier,
                src
            )
            
            # 結果をキャッシュ
            result = ZAdaptiveTrendResult(
                middle=middle,
                upper_band=upper_band,
                lower_band=lower_band,
                trend=trend,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                z_atr=z_atr,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values
            )
            
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return middle  # 元のインターフェイスと互換性を保つため中心線を返す
            
        except Exception as e:
            self.logger.error(f"Zアダプティブトレンド計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.middle, result.upper_band, result.lower_band
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド値（1:上昇トレンド、-1:下降トレンド、0:トレンドなし）
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.trend
        except Exception as e:
            self.logger.error(f"トレンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.er
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    # 後方互換性のため
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得（後方互換性のため）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        return self.get_efficiency_ratio(data)
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.dynamic_multiplier
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.z_atr
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最大乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大乗数の値
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.max_mult_values
        except Exception as e:
            self.logger.error(f"動的最大乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最小乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小乗数の値
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
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.min_mult_values
        except Exception as e:
            self.logger.error(f"動的最小乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        # キャッシュをクリア
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存オブジェクトもリセット
        self.cycle_er.reset()
        self._z_adaptive_ma.reset()
        self._c_atr.reset() 