#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback
import math

try:
    from ..indicator import Indicator
    from .source_calculator import calculate_source_simple
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    from source_calculator import calculate_source_simple

# 条件付きインポート（動的期間用）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        EHLERS_UNIFIED_DC_AVAILABLE = True
    except ImportError:
        EhlersUnifiedDC = None
        EHLERS_UNIFIED_DC_AVAILABLE = False


def calculate_extended_source(data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
    """
    Calculate extended source types as defined in Pine Script
    
    Args:
        data: Price data
        src_type: Extended source type
        
    Returns:
        Calculated source values
    """
    if isinstance(data, pd.DataFrame):
        # Basic sources
        if src_type == 'open':
            return data['open'].values
        elif src_type == 'high':
            return data['high'].values
        elif src_type == 'low':
            return data['low'].values
        elif src_type == 'close':
            return data['close'].values
        elif src_type == 'hlc3':  # Typical Price
            return ((data['high'] + data['low'] + data['close']) / 3).values
        elif src_type == 'hl2':  # Median Price
            return ((data['high'] + data['low']) / 2).values
        elif src_type == 'ohlc4':
            return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
        
        # Extended sources from Pine Script
        elif src_type == 'oo2':
            return ((data['open'] + data['open'].shift(1).fillna(data['open'])) / 2).values
        elif src_type == 'oh2':
            return ((data['open'] + data['high']) / 2).values
        elif src_type == 'ol2':
            return ((data['open'] + data['low']) / 2).values
        elif src_type == 'oc2':
            return ((data['open'] + data['close']) / 2).values
        elif src_type == 'hh2':
            return ((data['high'] + data['high'].shift(1).fillna(data['high'])) / 2).values
        elif src_type == 'hc2':
            return ((data['high'] + data['close']) / 2).values
        elif src_type == 'll2':
            return ((data['low'] + data['low'].shift(1).fillna(data['low'])) / 2).values
        elif src_type == 'lc2':
            return ((data['low'] + data['close']) / 2).values
        elif src_type == 'cc2':
            return ((data['close'] + data['close'].shift(1).fillna(data['close'])) / 2).values
        elif src_type == 'wc':  # Weighted Close
            return ((2 * data['close'] + data['high'] + data['low']) / 4).values
        else:
            return data['close'].values  # Default to close
    else:
        # For numpy arrays, use source calculator for basic types
        if src_type in ['open', 'high', 'low', 'close', 'hlc3', 'hl2', 'ohlc4']:
            return calculate_source_simple(data, src_type)
        else:
            # For extended types, fallback to close
            return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data


@dataclass
class SuperSmootherResult:
    """Super Smoother Filter calculation result"""
    values: np.ndarray      # Super Smoother filtered values
    raw_values: np.ndarray  # Original source values for comparison


@njit(fastmath=True, cache=True)
def calculate_super_smoother_2pole(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate 2-pole Super Smoother Filter
    
    Args:
        source: Input price series
        length: Filter length
        
    Returns:
        Filtered values array
    """
    data_length = len(source)
    result = np.zeros(data_length)
    
    if data_length < 3 or length < 2:
        return result
    
    # Calculate coefficients
    PI = 2 * math.asin(1)
    arg = math.sqrt(2) * PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(arg)
    
    coef3 = -math.pow(a1, 2)
    coef2 = b1
    coef1 = 1 - coef2 - coef3
    
    # Initialize first values
    result[0] = source[0]
    if data_length > 1:
        result[1] = source[1]
    
    # Calculate Super Smoother values
    for i in range(2, data_length):
        if not np.isnan(source[i]):
            result[i] = (coef1 * source[i] + 
                        coef2 * result[i-1] + 
                        coef3 * result[i-2])
        else:
            result[i] = result[i-1]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_super_smoother_3pole(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate 3-pole Super Smoother Filter
    
    Args:
        source: Input price series
        length: Filter length
        
    Returns:
        Filtered values array
    """
    data_length = len(source)
    result = np.zeros(data_length)
    
    if data_length < 4 or length < 2:
        return result
    
    # Calculate coefficients
    PI = 2 * math.asin(1)
    arg = PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(1.738 * arg)
    c1 = math.pow(a1, 2)
    
    coef4 = math.pow(c1, 2)
    coef3 = -(c1 + b1 * c1)
    coef2 = b1 + c1
    coef1 = 1 - coef2 - coef3 - coef4
    
    # Initialize first values
    result[0] = source[0]
    if data_length > 1:
        result[1] = source[1]
    if data_length > 2:
        result[2] = source[2]
    
    # Calculate Super Smoother values
    for i in range(3, data_length):
        if not np.isnan(source[i]):
            result[i] = (coef1 * source[i] + 
                        coef2 * result[i-1] + 
                        coef3 * result[i-2] + 
                        coef4 * result[i-3])
        else:
            result[i] = result[i-1]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_super_smoother_numba(source: np.ndarray, length: int, num_poles: int) -> np.ndarray:
    """
    Calculate Super Smoother Filter with specified number of poles
    
    Args:
        source: Input price series
        length: Filter length
        num_poles: Number of poles (2 or 3)
        
    Returns:
        Filtered values array
    """
    if num_poles == 2:
        return calculate_super_smoother_2pole(source, length)
    elif num_poles == 3:
        return calculate_super_smoother_3pole(source, length)
    else:
        # Default to 2-pole
        return calculate_super_smoother_2pole(source, length)


@njit(fastmath=True, cache=True)
def calculate_super_smoother_dynamic(
    source: np.ndarray,
    length: int,
    num_poles: int,
    dynamic_periods: np.ndarray = None
) -> np.ndarray:
    """
    動的期間対応SuperSmoother Filter計算
    
    Args:
        source: 入力価格系列
        length: 基本フィルター長
        num_poles: ポール数 (2または3)
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        フィルターされた値の配列
    """
    data_length = len(source)
    result = np.zeros(data_length, dtype=np.float64)
    
    if data_length < 3:
        return result
    
    # 初期値設定
    result[0] = source[0]
    if data_length > 1:
        result[1] = source[1]
    if data_length > 2 and num_poles == 3:
        result[2] = source[2]
    
    # 動的期間を使用してスムージング
    for i in range(max(2, 3 if num_poles == 3 else 2), data_length):
        # 現在の期間を決定
        current_length = length
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_length = max(2, min(int(dynamic_periods[i]), 50))  # 2-50期間に制限
        
        # 係数計算（現在の期間で）
        PI = 2 * math.asin(1)
        
        if num_poles == 2:
            # 2-pole coefficients
            arg = math.sqrt(2) * PI / current_length
            a1 = math.exp(-arg)
            b1 = 2 * a1 * math.cos(arg)
            
            coef3 = -math.pow(a1, 2)
            coef2 = b1
            coef1 = 1 - coef2 - coef3
            
            if not np.isnan(source[i]):
                result[i] = (coef1 * source[i] + 
                            coef2 * result[i-1] + 
                            coef3 * result[i-2])
            else:
                result[i] = result[i-1]
        
        elif num_poles == 3 and i >= 3:
            # 3-pole coefficients
            arg = PI / current_length
            a1 = math.exp(-arg)
            b1 = 2 * a1 * math.cos(1.738 * arg)
            c1 = math.pow(a1, 2)
            
            coef4 = math.pow(c1, 2)
            coef3 = -(c1 + b1 * c1)
            coef2 = b1 + c1
            coef1 = 1 - coef2 - coef3 - coef4
            
            if not np.isnan(source[i]):
                result[i] = (coef1 * source[i] + 
                            coef2 * result[i-1] + 
                            coef3 * result[i-2] + 
                            coef4 * result[i-3])
            else:
                result[i] = result[i-1]
        else:
            # フォールバック
            result[i] = result[i-1]
    
    return result


class SuperSmoother(Indicator):
    """
    Ehlers Super Smoother Filter
    
    A sophisticated digital filter that provides superior smoothing with minimal lag.
    Developed by John Ehlers, it uses poles in the z-domain to create a smooth output
    while maintaining responsiveness to price changes.
    
    Features:
    - 2-pole or 3-pole filter options
    - Extended price source support (cc2, oo2, oh2, ol2, oc2, hh2, hl2, hc2, ll2, lc2, wc)
    - Numba optimization for high performance
    - Proper handling of missing values
    
    Usage:
    - 2-pole: Faster response, good for shorter timeframes
    - 3-pole: Smoother output, better for longer timeframes
    """
    
    # Extended source types from Pine Script
    SRC_TYPES = [
        'open', 'high', 'low', 'close', 'oo2', 'oh2', 'ol2', 'oc2', 
        'hh2', 'hl2', 'hc2', 'll2', 'lc2', 'cc2', 'hlc3', 'ohlc4', 'wc'
    ]
    
    def __init__(self, 
                 length: int = 15,
                 num_poles: int = 2,
                 src_type: str = 'cc2',
                 # 動的期間パラメータ
                 period_mode: str = 'fixed',
                 cycle_detector_type: str = 'hody_e',
                 lp_period: int = 13,
                 hp_period: int = 124,
                 cycle_part: float = 0.5,
                 max_cycle: int = 124,
                 min_cycle: int = 13,
                 max_output: int = 124,
                 min_output: int = 13):
        """
        Constructor
        
        Args:
            length: Filter length (default: 15, minimum: 2)
            num_poles: Number of poles (2 or 3, default: 2)
            src_type: Price source type (default: 'cc2')
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
        """
        if length < 2:
            raise ValueError("Length must be at least 2")
        if num_poles not in [2, 3]:
            raise ValueError("Number of poles must be 2 or 3")
        if src_type not in self.SRC_TYPES:
            raise ValueError(f"Invalid source type: {src_type}. Valid options: {', '.join(self.SRC_TYPES)}")
        
        # 動的期間文字列の作成
        dynamic_str = f"_dynamic({cycle_detector_type})" if period_mode == 'dynamic' else ""
        
        super().__init__(f"SuperSmoother(length={length}, poles={num_poles}, src={src_type}{dynamic_str})")
        
        self.length = length
        self.num_poles = num_poles
        self.src_type = src_type.lower()
        
        # 動的期間パラメータ
        self.period_mode = period_mode.lower()
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # 動的期間検証
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な期間モード: {period_mode}")
        
        # ドミナントサイクル検出器の初期化
        self.dc_detector = None
        self._last_dc_values = None
        if self.period_mode == 'dynamic' and EHLERS_UNIFIED_DC_AVAILABLE:
            try:
                self.dc_detector = EhlersUnifiedDC(
                    detector_type=self.cycle_detector_type,
                    cycle_part=self.cycle_part,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type=self.src_type,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            except Exception as e:
                self.logger.warning(f"ドミナントサイクル検出器の初期化に失敗しました: {e}")
                self.period_mode = 'fixed'
        elif self.period_mode == 'dynamic' and not EHLERS_UNIFIED_DC_AVAILABLE:
            self.logger.warning("EhlersUnifiedDCが利用できません。固定期間モードに変更します。")
            self.period_mode = 'fixed'
        
        # Result cache
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        Calculate hash value for data caching
        
        Args:
            data: Price data
            
        Returns:
            Data hash string
        """
        try:
            # Get data information
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
            
            # Parameter signature
            params_sig = f"{self.length}_{self.num_poles}_{self.src_type}_{self.period_mode}_{self.cycle_detector_type}"
            
            # Fast hash
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # Fallback
            return f"{id(data)}_{self.length}_{self.num_poles}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SuperSmootherResult:
        """
        Calculate Super Smoother Filter
        
        Args:
            data: Price data (DataFrame or NumPy array)
                For DataFrame, columns required by the selected source type are needed
        
        Returns:
            SuperSmootherResult: Filtered values and original source values
        """
        try:
            # Cache check
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # Update cache key order
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return SuperSmootherResult(
                    values=cached_result.values.copy(),
                    raw_values=cached_result.raw_values.copy()
                )
            
            # Get source data using extended source calculation
            source_values = calculate_extended_source(data, self.src_type)
            
            # Ensure it's a numpy array of float64
            if not isinstance(source_values, np.ndarray):
                source_values = np.array(source_values)
            if source_values.dtype != np.float64:
                source_values = source_values.astype(np.float64)
            
            # Validate data length
            data_length = len(source_values)
            if data_length == 0:
                raise ValueError("Input data is empty")
            
            # Check minimum data length requirement
            min_len_required = max(self.length, 4 if self.num_poles == 3 else 3)
            if data_length < min_len_required:
                self.logger.warning(f"Data length ({data_length}) is shorter than required minimum ({min_len_required}). Returning raw values.")
                # Return raw values when insufficient data
                result = SuperSmootherResult(
                    values=source_values.copy(),
                    raw_values=source_values.copy()
                )
                return result
            
            # 動的期間の計算（オプション）
            dynamic_periods = None
            if self.period_mode == 'dynamic' and self.dc_detector is not None:
                try:
                    dc_result = self.dc_detector.calculate(data)
                    if dc_result is not None:
                        dynamic_periods = np.asarray(dc_result, dtype=np.float64)
                        self._last_dc_values = dynamic_periods.copy()
                except Exception as e:
                    self.logger.warning(f"ドミナントサイクル検出に失敗しました: {e}")
                    # フォールバック: 前回の値を使用
                    if self._last_dc_values is not None:
                        dynamic_periods = self._last_dc_values
            
            # SuperSmootherの計算
            if self.period_mode == 'dynamic' and dynamic_periods is not None:
                # 動的期間対応版を使用
                smoothed_values = calculate_super_smoother_dynamic(
                    source_values, self.length, self.num_poles, dynamic_periods
                )
            else:
                # 固定期間版を使用
                smoothed_values = calculate_super_smoother_numba(
                    source_values, self.length, self.num_poles
                )
            
            # Create result
            result = SuperSmootherResult(
                values=smoothed_values.copy(),
                raw_values=source_values.copy()
            )
            
            # Update cache
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # Remove oldest cache
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = smoothed_values  # For base class compatibility
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"SuperSmoother calculation error: {error_msg}\n{stack_trace}")
            
            # Return empty result on error
            error_result = SuperSmootherResult(
                values=np.array([]),
                raw_values=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """Get Super Smoother values only (for backward compatibility)"""
        if not self._result_cache:
            return None
            
        # Use latest cache
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_raw_values(self) -> Optional[np.ndarray]:
        """Get original source values"""
        if not self._result_cache:
            return None
            
        # Use latest cache
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.raw_values.copy()
    
    def reset(self) -> None:
        """Reset indicator state"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._last_dc_values = None
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        self.logger.debug(f"SuperSmoother '{self.name}' has been reset.")
    
    def __str__(self) -> str:
        """String representation"""
        return self.name