#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import traceback

try:
    from .indicator import Indicator
    from .chop_trend import ChopTrend
    from .chop_er import ChopER
    from .adx_vol import ADXVol
    from .normalized_adx import NormalizedADX
    from .efficiency_ratio import EfficiencyRatio
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class ChopTrend:
        def __init__(self, **kwargs): pass
        def calculate(self, data): 
            class Result: 
                values = np.array([])
            return Result()
        def reset(self): pass
    class ChopER:
        def __init__(self, **kwargs): pass
        def calculate(self, data): 
            class Result: 
                values = np.array([])
            return Result()
        def reset(self): pass
    class ADXVol:
        def __init__(self, **kwargs): pass
        def calculate(self, data): 
            class Result: 
                values = np.array([])
            return Result()
        def reset(self): pass
    class NormalizedADX:
        def __init__(self, **kwargs): pass
        def calculate(self, data): 
            class Result: 
                values = np.array([])
            return Result()
        def reset(self): pass
    class EfficiencyRatio:
        def __init__(self, **kwargs): pass
        def calculate(self, data): 
            class Result: 
                values = np.array([])
            return Result()
        def reset(self): pass


class AdaptivePeriodResult(NamedTuple):
    """適応期間計算結果"""
    periods: np.ndarray
    scaling_factors: np.ndarray  # 使用されたスケーリングファクター
    current_period: float
    avg_period: float
    min_period_used: float
    max_period_used: float


@jit(nopython=True, cache=True)
def calculate_adaptive_periods_numba(
    indicator_values: np.ndarray,
    min_period: float,
    max_period: float,
    power: float = 1.0,
    invert: bool = False,
    reverse_mapping: bool = False
) -> tuple:
    """
    0-1の値から動的期間を計算する (Numba JIT)
    
    KAMAのスムージング定数計算ロジックを正確に適用:
    1. 期間をスムージング定数に変換: sc = 2.0 / (period + 1.0)
    2. 線形補間: adaptive_sc = (indicator_value * (fast_sc - slow_sc) + slow_sc) ** power
    3. 期間に戻す: adaptive_period = round((2.0 / adaptive_sc) - 1.0)
    
    Args:
        indicator_values: 0-1の範囲のインディケーター値配列
        min_period: 最小期間
        max_period: 最大期間
        power: べき乗値（デフォルト: 1.0、KAMAは2.0）
        invert: 値を反転するか（Trueの場合、1 - indicator_valueで計算）
        reverse_mapping: 期間マッピングを逆転するか（Trueの場合、高い値で長い期間）
    
    Returns:
        tuple: (periods, scaling_factors) - periodsは整数配列
    """
    length = len(indicator_values)
    periods = np.full(length, np.nan)
    scaling_factors = np.full(length, np.nan)
    
    if length == 0:
        return periods, scaling_factors
    
    # 期間マッピングの設定
    if reverse_mapping:
        # 逆マッピング: 高い値 → 長い期間, 低い値 → 短い期間
        fast_sc = 2.0 / (max_period + 1.0)  # 長い期間 = 小さなスムージング定数
        slow_sc = 2.0 / (min_period + 1.0)  # 短い期間 = 大きなスムージング定数
    else:
        # 通常マッピング: 高い値 → 短い期間, 低い値 → 長い期間
        fast_sc = 2.0 / (min_period + 1.0)  # 短い期間 = 大きなスムージング定数
        slow_sc = 2.0 / (max_period + 1.0)  # 長い期間 = 小さなスムージング定数
    
    sc_range = fast_sc - slow_sc
    
    for i in range(length):
        value = indicator_values[i]
        
        if np.isnan(value):
            continue
            
        # 値を0-1の範囲にクリップ
        clipped_value = max(0.0, min(1.0, value))
        
        # 反転オプション
        if invert:
            clipped_value = 1.0 - clipped_value
            
        # KAMAスタイルの線形補間（スムージング定数で）
        adaptive_sc = clipped_value * sc_range + slow_sc
        
        # べき乗適用（KAMAでは二乗を使用）
        if power != 1.0:
            # スムージング定数を正規化してべき乗を適用
            normalized = (adaptive_sc - slow_sc) / sc_range if sc_range > 1e-9 else 0.0
            powered = normalized ** power
            final_sc = powered * sc_range + slow_sc
        else:
            final_sc = adaptive_sc
            
        # スムージング定数から期間に戻す
        if final_sc > 1e-9:  # ゼロ除算防止
            calculated_period = (2.0 / final_sc) - 1.0
            # 期間を四捨五入して整数に丸める
            periods[i] = round(calculated_period)
        else:
            periods[i] = 999999  # 実質的に無限大の期間（整数として）
            
        scaling_factors[i] = clipped_value
    
    return periods, scaling_factors


@jit(nopython=True, cache=True)
def calculate_statistics_numba(periods: np.ndarray) -> tuple:
    """
    期間統計を計算する (Numba JIT)
    
    Args:
        periods: 期間の配列
    
    Returns:
        tuple: (current_period, avg_period, min_period, max_period)
    """
    valid_periods = periods[~np.isnan(periods)]
    
    if len(valid_periods) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    current_period = periods[-1] if not np.isnan(periods[-1]) else np.nan
    avg_period = np.mean(valid_periods)
    min_period_used = np.min(valid_periods)
    max_period_used = np.max(valid_periods)
    
    return current_period, avg_period, min_period_used, max_period_used


class AdaptivePeriod(Indicator):
    """
    適応期間インディケーター
    
    0から1の間のインディケーター値を受け取り、最小値・最大値の範囲で
    動的に適応された期間を整数として出力します。
    
    KAMAの正確なスムージング定数計算ロジックを実装：
    1. 期間をスムージング定数に変換: sc = 2.0 / (period + 1.0)
    2. 線形補間: adaptive_sc = (値 * (fast_sc - slow_sc) + slow_sc) ** べき乗
    3. 期間に戻す: adaptive_period = round((2.0 / adaptive_sc) - 1.0)
    
    マッピングオプション:
    - 通常マッピング（reverse_mapping=False）: 高い値 → 短い期間, 低い値 → 長い期間
    - 逆マッピング（reverse_mapping=True）: 高い値 → 長い期間, 低い値 → 短い期間
    
    特徴:
    - 0-1の任意のインディケーター値に対応
    - KAMAと同じスムージング定数による正確な線形補間
    - べき乗パラメータによる非線形変換対応（KAMAは2.0）
    - 値の反転オプション（invert）
    - 期間マッピングの逆転オプション（reverse_mapping）
    - 最終出力は常に整数の期間として返される
    """
    
    def __init__(self,
                 min_period: float = 5.0,
                 max_period: float = 50.0,
                 power: float = 1.0,
                 invert: bool = False,
                 reverse_mapping: bool = False,
                 trigger_indicator: Optional[str] = None,
                 **indicator_params):
        """
        コンストラクタ
        
        Args:
            min_period: 最小期間（デフォルト: 5.0）
            max_period: 最大期間（デフォルト: 50.0）
            power: べき乗値（1.0=線形、2.0=二乗など、デフォルト: 1.0、KAMAは2.0）
            invert: 値を反転するか（Trueの場合、1 - indicator_valueで計算、デフォルト: False）
            reverse_mapping: 期間マッピングを逆転するか（Trueの場合、高い値で長い期間、デフォルト: False）
            trigger_indicator: 使用するトリガーインジケーター（None, 'chop_trend', 'chop_er', 'adx_vol', 'normalized_adx', 'efficiency_ratio'）
            **indicator_params: トリガーインジケーターに渡すパラメータ
        """
        if min_period >= max_period:
            raise ValueError(f"最小期間({min_period})は最大期間({max_period})より小さい必要があります")
        if min_period <= 0:
            raise ValueError(f"最小期間は0より大きい必要があります: {min_period}")
        if power <= 0:
            raise ValueError(f"べき乗値は0より大きい必要があります: {power}")
            
        # トリガーインジケーターの検証
        valid_triggers = [None, 'chop_trend', 'chop_er', 'adx_vol', 'normalized_adx', 'efficiency_ratio']
        if trigger_indicator not in valid_triggers:
            raise ValueError(f"無効なトリガーインジケーター: {trigger_indicator}. 使用可能: {valid_triggers}")
            
        invert_str = "_inv" if invert else ""
        reverse_str = "_rev" if reverse_mapping else ""
        power_str = f"_p{power}" if power != 1.0 else ""
        trigger_str = f"_trigger({trigger_indicator})" if trigger_indicator else ""
        super().__init__(f"AdaptivePeriod(min={min_period},max={max_period}{power_str}{invert_str}{reverse_str}{trigger_str})")
        
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.power = float(power)
        self.invert = invert
        self.reverse_mapping = reverse_mapping
        self.trigger_indicator = trigger_indicator
        
        # トリガーインジケーターの初期化
        self.indicator_instance = None
        if self.trigger_indicator:
            self._initialize_trigger_indicator(**indicator_params)
        
        self._cache = {}
        self._result: Optional[AdaptivePeriodResult] = None

    def _get_data_hash(self, indicator_values: np.ndarray) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(indicator_values, np.ndarray):
                data_hash_val = hash(indicator_values.tobytes())
            else:
                data_hash_val = hash(str(indicator_values))
        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. 文字列表現を使用します。", exc_info=True)
            data_hash_val = hash(str(indicator_values))

        param_str = f"min={self.min_period}_max={self.max_period}_pow={self.power}_inv={self.invert}_rev={self.reverse_mapping}_trigger({self.trigger_indicator})"
        return f"{data_hash_val}_{param_str}"

    def calculate(self, data: Union[np.ndarray, list, pd.DataFrame]) -> AdaptivePeriodResult:
        """
        適応期間を計算する
        
        Args:
            data: インジケーター値配列（0-1の範囲）またはトリガーインジケーター用の価格データ
        
        Returns:
            AdaptivePeriodResult: 適応期間の結果
        """
        if not hasattr(data, '__len__') or len(data) == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            return AdaptivePeriodResult(
                periods=np.array([]),
                scaling_factors=np.array([]),
                current_period=np.nan,
                avg_period=np.nan,
                min_period_used=np.nan,
                max_period_used=np.nan
            )

        try:
            # トリガーインジケーターが設定されている場合は、価格データから値を計算
            if self.trigger_indicator and self.indicator_instance:
                if isinstance(data, (np.ndarray, pd.DataFrame)):
                    # 価格データと判断してインジケーターを計算
                    try:
                        result = self.indicator_instance.calculate(data)
                        indicator_values = result.values
                        self.logger.info(f"トリガーインジケーター '{self.trigger_indicator}' から値を取得: {len(indicator_values)}本")
                    except Exception as e:
                        self.logger.error(f"トリガーインジケーター '{self.trigger_indicator}' の計算に失敗: {e}")
                        current_data_len = len(data) if hasattr(data, '__len__') else 0
                        return AdaptivePeriodResult(
                            periods=np.full(current_data_len, np.nan),
                            scaling_factors=np.full(current_data_len, np.nan),
                            current_period=np.nan,
                            avg_period=np.nan,
                            min_period_used=np.nan,
                            max_period_used=np.nan
                        )
                else:
                    # 1次元配列の場合は直接の値として扱う
                    indicator_values = data
            else:
                # トリガーインジケーターが設定されていない場合は、直接の値として扱う
                indicator_values = data

            # NumPy配列に変換
            if not isinstance(indicator_values, np.ndarray):
                indicator_values = np.array(indicator_values)
            
            current_data_len = len(indicator_values)
            data_hash = self._get_data_hash(indicator_values)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                if len(self._result.periods) == current_data_len:
                    return AdaptivePeriodResult(
                        periods=self._result.periods.copy(),
                        scaling_factors=self._result.scaling_factors.copy(),
                        current_period=self._result.current_period,
                        avg_period=self._result.avg_period,
                        min_period_used=self._result.min_period_used,
                        max_period_used=self._result.max_period_used
                    )
                else:
                    del self._cache[data_hash]
                    self._result = None

            # float64に変換
            if indicator_values.dtype != np.float64:
                try:
                    indicator_values = indicator_values.astype(np.float64)
                except ValueError:
                    self.logger.error("インディケーター値をfloat64に変換できませんでした。")
                    return AdaptivePeriodResult(
                        periods=np.full(current_data_len, np.nan),
                        scaling_factors=np.full(current_data_len, np.nan),
                        current_period=np.nan,
                        avg_period=np.nan,
                        min_period_used=np.nan,
                        max_period_used=np.nan
                    )

            # C-contiguous配列にする
            if not indicator_values.flags['C_CONTIGUOUS']:
                indicator_values = np.ascontiguousarray(indicator_values)

            # 適応期間の計算
            periods, scaling_factors = calculate_adaptive_periods_numba(
                indicator_values,
                self.min_period,
                self.max_period,
                self.power,
                self.invert,
                self.reverse_mapping
            )

            # 統計の計算
            current_period, avg_period, min_period_used, max_period_used = calculate_statistics_numba(periods)

            result = AdaptivePeriodResult(
                periods=periods,
                scaling_factors=scaling_factors,
                current_period=current_period,
                avg_period=avg_period,
                min_period_used=min_period_used,
                max_period_used=max_period_used
            )

            self._result = result
            self._cache[data_hash] = self._result
            return AdaptivePeriodResult(
                periods=result.periods.copy(),
                scaling_factors=result.scaling_factors.copy(),
                current_period=result.current_period,
                avg_period=result.avg_period,
                min_period_used=result.min_period_used,
                max_period_used=result.max_period_used
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"適応期間 '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            current_data_len = len(data) if hasattr(data, '__len__') else 0
            return AdaptivePeriodResult(
                periods=np.full(current_data_len, np.nan),
                scaling_factors=np.full(current_data_len, np.nan),
                current_period=np.nan,
                avg_period=np.nan,
                min_period_used=np.nan,
                max_period_used=np.nan
            )

    def get_periods(self) -> Optional[np.ndarray]:
        """期間値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.periods.copy()
        return None

    def get_scaling_factors(self) -> Optional[np.ndarray]:
        """スケーリングファクターを取得する"""
        if self._result is not None:
            return self._result.scaling_factors.copy()
        return None

    def get_current_period(self) -> float:
        """現在の期間を取得する"""
        if self._result is not None:
            return self._result.current_period
        return np.nan

    def get_statistics(self) -> dict:
        """統計情報を取得する"""
        if self._result is not None:
            return {
                'current': self._result.current_period,
                'average': self._result.avg_period,
                'min_used': self._result.min_period_used,
                'max_used': self._result.max_period_used,
                'range_utilized': (self._result.max_period_used - self._result.min_period_used) if not (np.isnan(self._result.max_period_used) or np.isnan(self._result.min_period_used)) else np.nan
            }
        return {
            'current': np.nan,
            'average': np.nan,
            'min_used': np.nan,
            'max_used': np.nan,
            'range_utilized': np.nan
        }

    def get_trigger_indicator_name(self) -> Optional[str]:
        """使用中のトリガーインジケーター名を取得する"""
        return self.trigger_indicator

    def get_trigger_indicator_instance(self):
        """トリガーインジケーターのインスタンスを取得する"""
        return self.indicator_instance

    def has_trigger_indicator(self) -> bool:
        """トリガーインジケーターが設定されているかを確認する"""
        return self.trigger_indicator is not None and self.indicator_instance is not None

    def reset(self) -> None:
        """インジケータの状態（キャッシュ、結果）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.indicator_instance and hasattr(self.indicator_instance, 'reset'):
            self.indicator_instance.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。")

    def _initialize_trigger_indicator(self, **indicator_params):
        """トリガーインジケーターを初期化する"""
        try:
            if self.trigger_indicator == 'chop_trend':
                self.indicator_instance = ChopTrend(**indicator_params)
            elif self.trigger_indicator == 'chop_er':
                self.indicator_instance = ChopER(**indicator_params)
            elif self.trigger_indicator == 'adx_vol':
                self.indicator_instance = ADXVol(**indicator_params)
            elif self.trigger_indicator == 'normalized_adx':
                self.indicator_instance = NormalizedADX(**indicator_params)
            elif self.trigger_indicator == 'efficiency_ratio':
                self.indicator_instance = EfficiencyRatio(**indicator_params)
            else:
                self.indicator_instance = None
                
        except Exception as e:
            self.logger.error(f"トリガーインジケーター '{self.trigger_indicator}' の初期化に失敗: {e}")
            self.indicator_instance = None 