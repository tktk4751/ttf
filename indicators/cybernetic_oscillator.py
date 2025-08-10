#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import traceback
import math

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource:
        @staticmethod
        def calculate_source(data, src_type):
            if isinstance(data, pd.DataFrame):
                if src_type == 'close': return data['close'].values
                elif src_type == 'open': return data['open'].values
                elif src_type == 'high': return data['high'].values
                elif src_type == 'low': return data['low'].values
                elif src_type == 'hl2': return ((data['high'] + data['low']) / 2).values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                elif src_type == 'ohlc4': return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
                else: return data['close'].values # Default to close
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data


class CyberneticOscillatorResult(NamedTuple):
    """Cybernetic Oscillator calculation result"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@njit(fastmath=True, cache=True)
def calculate_filter_coefficients(period: int, is_hp: bool) -> tuple:
    """
    Calculate coefficients for highpass and lowpass filters
    
    Args:
        period: The critical period of the filter
        is_hp: If True, coefficients are for highpass filter, else lowpass
    
    Returns:
        tuple: (c1, c2, c3) coefficients
    """
    a0 = 1.414 * math.pi / period
    a1 = math.exp(-a0)
    c2 = 2.0 * a1 * math.cos(a0)
    c3 = -a1 * a1
    
    if is_hp:
        c1 = (1.0 + c2 - c3) * 0.25
    else:
        c1 = 1.0 - c2 - c3
    
    return c1, c2, c3


@njit(fastmath=True, cache=True)
def calculate_highpass_filter(source: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate second-order highpass filter
    
    Args:
        source: The series of values to process
        period: The length of the filter's critical period
    
    Returns:
        The filtered source values
    """
    length = len(source)
    result = np.zeros(length)
    
    if length < 4:
        return result
    
    c1, c2, c3 = calculate_filter_coefficients(period, True)
    
    for i in range(4, length):
        result[i] = (c1 * (source[i] - 2.0 * source[i-1] + source[i-2]) + 
                    c2 * result[i-1] + 
                    c3 * result[i-2])
    
    return result


@njit(fastmath=True, cache=True)
def calculate_super_smoother(source: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Super Smoother filter (second-order lowpass)
    
    Args:
        source: The series of values to process
        period: The length of the filter's critical period
    
    Returns:
        The filtered source values
    """
    length = len(source)
    result = np.copy(source)
    
    if length < 4:
        return result
    
    c1, c2, c3 = calculate_filter_coefficients(period, False)
    
    for i in range(4, length):
        result[i] = (c1 * 0.5 * (source[i] + source[i-1]) + 
                    c2 * result[i-1] + 
                    c3 * result[i-2])
    
    return result


@njit(fastmath=True, cache=True)
def calculate_rms(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate the root mean square (RMS) of a series
    
    Args:
        source: The series of values to process
        length: The number of bars in the calculation
    
    Returns:
        The RMS of the source values over length bars
    """
    data_length = len(source)
    result = np.full(data_length, np.nan)
    
    if data_length < length:
        return result
    
    for i in range(length - 1, data_length):
        sum_squares = 0.0
        for j in range(i - length + 1, i + 1):
            sum_squares += source[j] * source[j]
        result[i] = math.sqrt(sum_squares / length)
    
    return result


@njit(fastmath=True, cache=True)
def calculate_cybernetic_oscillator_numba(
    source: np.ndarray, 
    hp_period: int, 
    lp_period: int, 
    rms_length: int
) -> np.ndarray:
    """
    Calculate the Cybernetic Oscillator
    
    Args:
        source: The series of values to process
        hp_period: The highpass filter critical period
        lp_period: The lowpass filter critical period
        rms_length: The number of bars in the RMS calculation
    
    Returns:
        The Cybernetic Oscillator values
    """
    length = len(source)
    result = np.zeros(length)
    
    if length < max(hp_period, lp_period, rms_length):
        return result
    
    # Apply highpass filter
    hp_values = calculate_highpass_filter(source, hp_period)
    
    # Apply lowpass filter (Super Smoother) to highpass output
    lp_values = calculate_super_smoother(hp_values, lp_period)
    
    # Calculate RMS of the lowpass output
    rms_values = calculate_rms(lp_values, rms_length)
    
    # Calculate the final oscillator values
    for i in range(length):
        if not np.isnan(rms_values[i]) and rms_values[i] != 0.0:
            result[i] = lp_values[i] / rms_values[i]
        else:
            result[i] = 0.0
    
    return result


@jit(nopython=True, cache=True)
def calculate_trend_signals_with_range(values: np.ndarray, slope_index: int, range_threshold: float = 0.005) -> np.ndarray:
    """
    Calculate trend signals (range state supported) (Numba JIT)
    
    Args:
        values: Indicator values array
        slope_index: Slope judgment period
        range_threshold: Range judgment threshold (relative change rate)
    
    Returns:
        trend_signals: 1=up, -1=down, 0=range NumPy array
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    # Window size for statistical threshold calculation (fixed)
    stats_window = 21
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            current = values[i]
            previous = values[i - slope_index]
            
            # Basic change amount
            change = current - previous
            
            # Calculate relative change rate
            base_value = max(abs(current), abs(previous), 1e-10)  # Prevent division by zero
            relative_change = abs(change) / base_value
            
            # Calculate statistical threshold (standard deviation of past fluctuations)
            start_idx = max(slope_index, i - stats_window + 1)
            if start_idx < i - slope_index:
                # Calculate past change rates
                historical_changes = np.zeros(i - start_idx)
                for j in range(start_idx, i):
                    if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                        hist_current = values[j]
                        hist_previous = values[j - slope_index]
                        hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                        historical_changes[j - start_idx] = abs(hist_current - hist_previous) / hist_base
                
                # Use only standard deviation-based threshold
                if len(historical_changes) > 0:
                    # Standard deviation-based threshold
                    std_threshold = np.std(historical_changes) * 0.5  # 0.5 times standard deviation
                    
                    # Final range threshold is the larger of fixed threshold and std threshold
                    effective_threshold = max(range_threshold, std_threshold)
                else:
                    effective_threshold = range_threshold
            else:
                effective_threshold = range_threshold
            
            # Trend judgment
            if relative_change < effective_threshold:
                # If change is too small, it's range
                trend_signals[i] = 0  # range
            elif change > 0:
                # Upward trend
                trend_signals[i] = 1  # up
            else:
                # Downward trend
                trend_signals[i] = -1  # down
    
    return trend_signals


@jit(nopython=True, cache=True)
def calculate_current_trend_with_range(trend_signals: np.ndarray) -> tuple:
    """
    Calculate current trend state (range supported) (Numba JIT)
    
    Args:
        trend_signals: Trend signal array (1=up, -1=down, 0=range)
    
    Returns:
        tuple: (current_trend_index, current_trend_value)
               current_trend_index: 0=range, 1=up, 2=down (index for trend_names)
               current_trend_value: 0=range, 1=up, -1=down (actual trend value)
    """
    length = len(trend_signals)
    if length == 0:
        return 0, 0  # range
    
    # Get the latest value
    latest_trend = trend_signals[-1]
    
    if latest_trend == 1:  # up
        return 1, 1   # up
    elif latest_trend == -1:  # down
        return 2, -1   # down
    else:  # range
        return 0, 0  # range


class CyberneticOscillator(Indicator):
    """
    Cybernetic Oscillator indicator
    
    A flexible oscillator that measures the filtered price movement normalized by its RMS.
    The indicator applies a highpass filter followed by a lowpass filter (Super Smoother)
    and then normalizes the result by its RMS to create an oscillator.
    
    Features:
    - Highpass filter to remove low-frequency noise
    - Super Smoother (lowpass) for additional smoothing
    - RMS normalization for consistent scaling
    - Trend detection with range state support
    
    Usage:
    - Values above threshold: Strong directional movement
    - Values below threshold: Range-bound or weak movement
    - Zero line crossovers: Potential trend changes
    """
    
    def __init__(self, 
                 hp_period: int = 30,
                 lp_period: int = 20,
                 rms_length: int = 100,
                 src_type: str = 'close',
                 slope_index: int = 3,
                 range_threshold: float = 0.005):
        """
        Constructor
        
        Args:
            hp_period: Highpass filter critical period (default: 30)
            lp_period: Lowpass filter critical period (default: 20)
            rms_length: Number of bars in RMS calculation (default: 100)
            src_type: Price source ('close', 'hlc3', etc.) (default: 'close')
            slope_index: Trend judgment period (default: 3)
            range_threshold: Range judgment threshold (default: 0.005 = 0.5%)
        """
        if hp_period < lp_period:
            raise ValueError("The highpass period cannot be less than the lowpass period.")
        
        super().__init__(f"CyberneticOscillator(hp={hp_period},lp={lp_period},rms={rms_length},src={src_type},slope={slope_index},range_th={range_threshold:.3f})")
        
        self.hp_period = hp_period
        self.lp_period = lp_period
        self.rms_length = rms_length
        self.src_type = src_type
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        self._cache = {}
        self._result: Optional[CyberneticOscillatorResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Calculate hash value based on data and parameters"""
        try:
            if isinstance(data, pd.DataFrame):
                # For DataFrame, calculate hash based on shape and endpoints
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                # For NumPy array, use byte representation for hash
                data_hash_val = hash(data.tobytes())
            else:
                # For other data types, use string representation for hash
                data_hash_val = hash(str(data))

        except Exception as e:
            self.logger.warning(f"Error during data hash calculation: {e}. Using string representation of entire data.", exc_info=True)
            data_hash_val = hash(str(data)) # fallback

        # Create parameter string
        param_str = f"hp={self.hp_period}_lp={self.lp_period}_rms={self.rms_length}_src={self.src_type}_slope={self.slope_index}_range_th={self.range_threshold:.3f}"

        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CyberneticOscillatorResult:
        """
        Calculate the Cybernetic Oscillator
        
        Args:
            data: Price data (DataFrame or NumPy array)
                For DataFrame, columns required by the selected source type are needed
        
        Returns:
            CyberneticOscillatorResult: CO values and trend information
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("Input data is empty. Returning empty result.")
            empty_result = CyberneticOscillatorResult(
                values=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return empty_result
            
        try:
            data_hash = self._get_data_hash(data)

            # Cache check
            if data_hash in self._cache and self._result is not None:
                # Check if data length matches
                if len(self._result.values) == current_data_len:
                    return CyberneticOscillatorResult(
                        values=self._result.values.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_trend=self._result.current_trend,
                        current_trend_value=self._result.current_trend_value
                    )
                else:
                    self.logger.debug(f"Cache data length differs, recalculating.")
                    # Invalidate cache
                    del self._cache[data_hash]
                    self._result = None

            # Get price data using PriceSource
            prices = PriceSource.calculate_source(data, self.src_type)
            
            # Ensure it's a numpy array of float64 for Numba
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            if prices.dtype != np.float64:
                prices = prices.astype(np.float64)

            # Validate data length
            data_length = len(prices)
            if data_length == 0:
                self.logger.warning("Price data is empty. Returning empty result.")
                empty_result = CyberneticOscillatorResult(
                    values=np.array([]),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            # Check minimum data length requirement
            min_len_required = max(self.hp_period, self.lp_period, self.rms_length) + 4
            if data_length < min_len_required:
                self.logger.warning(f"Data length ({data_length}) is shorter than required minimum ({min_len_required}). Cannot calculate.")
                error_result = CyberneticOscillatorResult(
                    values=np.full(current_data_len, np.nan),
                    trend_signals=np.zeros(current_data_len, dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                return error_result

            # Calculate Cybernetic Oscillator
            co_values = calculate_cybernetic_oscillator_numba(
                prices, self.hp_period, self.lp_period, self.rms_length
            )

            # Calculate trend signals
            trend_signals = calculate_trend_signals_with_range(co_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = CyberneticOscillatorResult(
                values=co_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            # Save calculation result
            self._result = result
            self._cache[data_hash] = self._result
            return CyberneticOscillatorResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value
            )
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CyberneticOscillator '{self.name}' calculation error: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            error_result = CyberneticOscillatorResult(
                values=np.full(current_data_len, np.nan),
                trend_signals=np.zeros(current_data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """Get CO values only (for backward compatibility)"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """Get trend signals"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None

    def get_current_trend(self) -> str:
        """Get current trend state"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'

    def get_current_trend_value(self) -> int:
        """Get current trend value"""
        if self._result is not None:
            return self._result.current_trend_value
        return 0

    def reset(self) -> None:
        """Reset indicator state (cache, results)"""
        super().reset()
        self._result = None
        self._cache = {}
        self.logger.debug(f"Indicator '{self.name}' has been reset.")

    def __str__(self) -> str:
        """String representation"""
        return self.name