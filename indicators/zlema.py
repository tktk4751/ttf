#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import traceback # For detailed error logging

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    # Fallback for potential execution context issues, adjust as needed
    # This might happen if running the file directly without the package structure
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    # Define dummy classes if needed for static analysis, but real execution needs correct imports
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
    # Dummy EhlersUnifiedDC class for fallback
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 20.0)


class ZLEMAResult(NamedTuple):
    """ZLEMA計算結果"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@jit(nopython=True, cache=True)
def calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    EMA (Exponential Moving Average) を計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period: 期間
    
    Returns:
        EMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # EMAの平滑化係数
    alpha = 2.0 / (period + 1.0)
    
    # 最初の値を設定（最初の有効な値を使用）
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(prices[i]):
            result[i] = prices[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result  # 全てNaNの場合
    
    # EMAの計算
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(prices[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha * prices[i] + (1.0 - alpha) * result[i-1]
            else:
                result[i] = prices[i]
        # prices[i]がNaNの場合、result[i]はNaNのまま
    
    return result


@jit(nopython=True, cache=True)
def calculate_zlema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    ZLEMA (Zero Lag EMA) を計算する（Numba JIT）- 改良版
    
    ZLEMA計算手順:
    1. lag = (period - 1) / 2
    2. adjusted_src = src + (src - src[lag])  // ラグ補正
    3. zlema = ema(adjusted_src, period)
    
    Args:
        prices: 価格の配列
        period: 期間
    
    Returns:
        ZLEMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # 期間の調整（最小値と最大値の制限）
    effective_period = max(2, min(period, length // 2))
    
    # ラグ期間の計算
    lag = int((effective_period - 1) / 2)
    
    if lag >= length:
        # ラグ期間がデータ長以上の場合は通常のEMAで代替
        return calculate_ema_numba(prices, effective_period)
    
    # ラグ補正された価格系列を作成
    adjusted_prices = np.full(length, np.nan)
    
    # ラグ補正の適用
    for i in range(lag, length):
        if not np.isnan(prices[i]) and not np.isnan(prices[i - lag]):
            # adjusted_src = src + (src - src[lag])
            adjusted_prices[i] = prices[i] + (prices[i] - prices[i - lag])
        elif not np.isnan(prices[i]):
            # ラグ位置の値がNaNの場合は元の価格を使用
            adjusted_prices[i] = prices[i]
    
    # 調整された価格系列にEMAを適用
    if not np.all(np.isnan(adjusted_prices)):
        result = calculate_ema_numba(adjusted_prices, effective_period)
    
    # NaN値の補間処理（前方補間＋後方補間）
    # 前方補間
    last_valid = np.nan
    for i in range(length):
        if not np.isnan(result[i]):
            last_valid = result[i]
        elif not np.isnan(last_valid):
            result[i] = last_valid
    
    # 後方補間（最初のNaN値を埋める）
    first_valid = np.nan
    for i in range(length):
        if not np.isnan(result[i]):
            first_valid = result[i]
            break
    
    if not np.isnan(first_valid):
        for i in range(length):
            if np.isnan(result[i]):
                result[i] = first_valid
            else:
                break
    
    return result


@jit(nopython=True, cache=True)
def calculate_dynamic_zlema_numba(
    prices: np.ndarray,
    period_array: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    動的期間でZLEMAを計算する（Numba JIT）- 改良版
    
    Args:
        prices: 価格の配列
        period_array: 各時点での期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        ZLEMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    # 計算開始位置を調整（max_periodが大きすぎる場合の対策）
    effective_start = min(max_period, length // 4)  # データ長の1/4または最大期間の小さい方
    
    # 各時点での動的ZLEMAを計算
    for i in range(effective_start, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(period_array[i]) if not np.isnan(period_array[i]) else 10
        curr_period = max(2, min(curr_period, i // 2))  # 期間の範囲制限
            
        # ラグ期間の計算
        lag = int((curr_period - 1) / 2)
        
        # 必要なデータ長を計算
        required_length = curr_period + lag
        
        if i >= required_length:
            # 現在位置から必要なデータを取得
            start_idx = i - required_length + 1
            window_prices = prices[start_idx:i+1]
            
            # ラグ補正された価格系列を作成
            adjusted_prices = np.full(len(window_prices), np.nan)
            
            for j in range(lag, len(window_prices)):
                if not np.isnan(window_prices[j]) and not np.isnan(window_prices[j - lag]):
                    adjusted_prices[j] = window_prices[j] + (window_prices[j] - window_prices[j - lag])
            
            # 調整された価格系列にEMAを適用
            if not np.all(np.isnan(adjusted_prices)):
                ema_values = calculate_ema_numba(adjusted_prices, curr_period)
                
                # 最後の有効値をZLEMAとして使用
                if len(ema_values) > 0 and not np.isnan(ema_values[-1]):
                    result[i] = ema_values[-1]
        elif i >= curr_period:
            # データが不足している場合は通常のEMAでフォールバック
            start_idx = max(0, i - curr_period + 1)
            window_prices = prices[start_idx:i+1]
            ema_values = calculate_ema_numba(window_prices, curr_period)
            if len(ema_values) > 0 and not np.isnan(ema_values[-1]):
                result[i] = ema_values[-1]
    
    # NaN値の補間処理（前方補間）
    last_valid = np.nan
    for i in range(length):
        if not np.isnan(result[i]):
            last_valid = result[i]
        elif not np.isnan(last_valid) and i < effective_start:
            # 計算開始前の値を補間
            result[i] = last_valid
    
    return result


@jit(nopython=True, cache=True)
def calculate_trend_signals_with_range(values: np.ndarray, slope_index: int, range_threshold: float = 0.005) -> np.ndarray:
    """
    トレンド信号を計算する（range状態対応版）(Numba JIT)
    
    Args:
        values: インジケーター値の配列
        slope_index: スロープ判定期間
        range_threshold: range判定の閾値（相対的変化率）
    
    Returns:
        trend_signals: 1=up, -1=down, 0=range のNumPy配列
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    # 統計的閾値計算用のウィンドウサイズ（固定）
    stats_window = 21
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            current = values[i]
            previous = values[i - slope_index]
            
            # 基本的な変化量
            change = current - previous
            
            # 相対的変化率の計算
            base_value = max(abs(current), abs(previous), 1e-10)  # ゼロ除算防止
            relative_change = abs(change) / base_value
            
            # 統計的閾値の計算（過去の変動の標準偏差）
            start_idx = max(slope_index, i - stats_window + 1)
            if start_idx < i - slope_index:
                # 過去の変化率を計算
                historical_changes = np.zeros(i - start_idx)
                for j in range(start_idx, i):
                    if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                        hist_current = values[j]
                        hist_previous = values[j - slope_index]
                        hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                        historical_changes[j - start_idx] = abs(hist_current - hist_previous) / hist_base
                
                # 標準偏差ベースの閾値のみを使用
                if len(historical_changes) > 0:
                    # 標準偏差ベースの閾値
                    std_threshold = np.std(historical_changes) * 0.5  # 0.5倍の標準偏差
                    
                    # 最終的なrange閾値は、固定閾値と標準偏差閾値の大きい方
                    effective_threshold = max(range_threshold, std_threshold)
                else:
                    effective_threshold = range_threshold
            else:
                effective_threshold = range_threshold
            
            # トレンド判定
            if relative_change < effective_threshold:
                # 変化が小さすぎる場合はrange
                trend_signals[i] = 0  # range
            elif change > 0:
                # 上昇トレンド
                trend_signals[i] = 1  # up
            else:
                # 下降トレンド
                trend_signals[i] = -1  # down
    
    return trend_signals


@jit(nopython=True, cache=True)
def calculate_current_trend_with_range(trend_signals: np.ndarray) -> tuple:
    """
    現在のトレンド状態を計算する（range対応版）(Numba JIT)
    
    Args:
        trend_signals: トレンド信号配列 (1=up, -1=down, 0=range)
    
    Returns:
        tuple: (current_trend_index, current_trend_value)
               current_trend_index: 0=range, 1=up, 2=down (trend_names用のインデックス)
               current_trend_value: 0=range, 1=up, -1=down (実際のトレンド値)
    """
    length = len(trend_signals)
    if length == 0:
        return 0, 0  # range
    
    # 最新の値を取得
    latest_trend = trend_signals[-1]
    
    if latest_trend == 1:  # up
        return 1, 1   # up
    elif latest_trend == -1:  # down
        return 2, -1   # down
    else:  # range
        return 0, 0  # range


class ZLEMA(Indicator):
    """
    ZLEMA (Zero Lag Exponential Moving Average) インジケーター
    
    ZLEMAは通常のEMAにラグ補正を適用したもので、価格変動により迅速に反応します。
    
    計算手順:
    1. lag = (period - 1) / 2
    2. adjusted_src = src + (src - src[lag])  # ラグ補正
    3. zlema = ema(adjusted_src, period)
    
    特徴:
    - 通常のEMAよりも価格変動に素早く反応
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)
    - 固定期間または動的期間（ドミナントサイクル）での計算に対応
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    """
    
    def __init__(self, 
                 period: int = 24, 
                 src_type: str = 'hlc3',
                 use_dynamic_period: bool = False,
                 cycle_part: float = 1.0,
                 detector_type: str = 'cycle_period2',
                 max_cycle: int = 233,
                 min_cycle: int = 13,
                 max_output: int = 144,
                 min_output: int = 13,
                 slope_index: int = 1,
                 range_threshold: float = 0.005,
                 lp_period: int = 10,
                 hp_period: int = 48):
        """
        コンストラクタ
        
        Args:
            period: 期間（固定期間モードで使用）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_dynamic_period: 動的期間を使用するかどうか
            cycle_part: サイクル部分の倍率（動的期間モード用）
            detector_type: 検出器タイプ（動的期間モード用）
            max_cycle: 最大サイクル期間（動的期間モード用）
            min_cycle: 最小サイクル期間（動的期間モード用）
            max_output: 最大出力値（動的期間モード用）
            min_output: 最小出力値（動的期間モード用）
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            lp_period: ローパスフィルター期間（動的期間モード用、デフォルト: 10）
            hp_period: ハイパスフィルター期間（動的期間モード用、デフォルト: 48）
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        super().__init__(f"ZLEMA(p={period},src={src_type}{dynamic_str},slope={slope_index},range_th={range_threshold:.3f})")
        
        self.period = period
        self.src_type = src_type
        self.use_dynamic_period = use_dynamic_period
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        # 動的期間モード用パラメータ
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        self.price_source_extractor = PriceSource()
        
        # ドミナントサイクル検出器（動的期間モード用）
        self.dc_detector = None
        self._last_dc_values = None  # 最後に計算されたDC値を保存
        if self.use_dynamic_period:
            self.dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.cycle_part,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                max_output=self.max_output,
                min_output=self.min_output,
                src_type=self.src_type,
                lp_period=self.lp_period,
                hp_period=self.hp_period
            )
        
        self._cache = {}
        self._result: Optional[ZLEMAResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        if self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        else:
            required_cols.add('close') # Default

        if isinstance(data, pd.DataFrame):
            relevant_cols = [col for col in data.columns if col.lower() in required_cols]
            present_cols = [col for col in relevant_cols if col in data.columns]
            if not present_cols:
                try:
                    shape_tuple = data.shape
                    first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row, last_row)
                    data_hash_val = hash(data_repr_tuple)
                except Exception:
                    data_hash_val = hash(str(data))
            else:
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            col_indices = []
            if 'open' in required_cols: col_indices.append(0)
            if 'high' in required_cols: col_indices.append(1)
            if 'low' in required_cols: col_indices.append(2)
            if 'close' in required_cols: col_indices.append(3)
            col_indices = sorted(list(set(col_indices)))
            if data.ndim == 2 and data.shape[1] > max(col_indices if col_indices else [-1]):
                data_values = data[:, col_indices]
                data_hash_val = hash(data_values.tobytes())
            else:
                data_hash_val = hash(data.tobytes())
        else:
            data_hash_val = hash(str(data))

        if self.use_dynamic_period:
            param_str = (f"p={self.period}_src={self.src_type}_"
                        f"dynamic={self.detector_type}_{self.max_output}_{self.min_output}")
        else:
            param_str = f"p={self.period}_src={self.src_type}"
        return f"{data_hash_val}_{param_str}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZLEMAResult:
        """
        ZLEMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）または直接価格の配列
        
        Returns:
            ZLEMAResult: ZLEMA値とトレンド情報を含む結果
        """
        try:
            # データチェック - 1次元配列が直接渡された場合はそのまま使用（固定期間のみ）
            if isinstance(data, np.ndarray) and data.ndim == 1 and not self.use_dynamic_period:
                src_prices = data
                data_hash = hash(data.tobytes()) # シンプルなハッシュ
                data_hash_key = f"{data_hash}_{self.period}_{self.slope_index}_{self.range_threshold}"
                
                if data_hash_key in self._cache and self._result is not None:
                    return self._result
                    
                # 直接1次元配列に対してZLEMAを計算
                zlema_values = calculate_zlema_numba(src_prices, self.period)
                
                # トレンド判定
                trend_signals = calculate_trend_signals_with_range(zlema_values, self.slope_index, self.range_threshold)
                trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
                trend_names = ['range', 'up', 'down']
                current_trend = trend_names[trend_index]
                
                result = ZLEMAResult(
                    values=zlema_values,
                    trend_signals=trend_signals,
                    current_trend=current_trend,
                    current_trend_value=trend_value
                )
                
                self._result = result
                self._cache[data_hash_key] = self._result
                return self._result
            
            # 通常のハッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result

            # PriceSourceを使ってソース価格を取得
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # データ長の検証
            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_result = ZLEMAResult(
                    values=np.array([]),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            if self.use_dynamic_period:
                # 動的期間モード
                if self.dc_detector is None:
                    self.logger.error("動的期間モードですが、ドミナントサイクル検出器が初期化されていません。")
                    error_result = ZLEMAResult(
                        values=np.full(data_length, np.nan),
                        trend_signals=np.zeros(data_length, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # ドミナントサイクルの計算
                dc_values = self.dc_detector.calculate(data)
                period_array = np.asarray(dc_values, dtype=np.float64)
                
                # DC値を保存（get_dynamic_periods用）
                self._last_dc_values = period_array.copy()
                
                # デバッグ情報の出力
                valid_dc = period_array[~np.isnan(period_array)]
                if len(valid_dc) > 0:
                    self.logger.info(f"動的期間統計 - 平均: {valid_dc.mean():.1f}, 範囲: {valid_dc.min():.0f} - {valid_dc.max():.0f}")
                else:
                    self.logger.warning("ドミナントサイクル値が全てNaNです。")
                
                # 最大期間の取得
                max_period_value_float = np.nanmax(period_array)
                if np.isnan(max_period_value_float):
                    max_period_value = self.period  # デフォルト期間を使用
                    self.logger.warning("ドミナントサイクルが全てNaNです。デフォルト期間を使用します。")
                else:
                    max_period_value = int(max_period_value_float)
                    if max_period_value < 1:
                        max_period_value = self.period
                
                # 有効期間の計算
                effective_period = max(2, min(max_period_value, data_length // 3))
                
                # データ長の検証（ZLEMAにはラグ分の余裕が必要）
                min_required_length = max(max_period_value, effective_period * 2)
                if data_length < min_required_length:
                    self.logger.warning(f"データ長({data_length})が推奨最小長({min_required_length})より短いですが、計算を試行します。")
                    # 計算を継続（警告のみ）
                
                # 動的ZLEMAの計算
                zlema_values = calculate_dynamic_zlema_numba(
                    src_prices, period_array, max_period_value
                )
                
                # 結果が全てNaNの場合のフォールバック
                if np.all(np.isnan(zlema_values)):
                    self.logger.warning("動的ZLEMA計算が失敗しました。固定期間でフォールバックします。")
                    fallback_period = int(np.nanmean(period_array)) if len(period_array) > 0 else self.period
                    fallback_period = max(2, min(fallback_period, data_length // 4))
                    zlema_values = calculate_zlema_numba(src_prices, fallback_period)
            else:
                # 固定期間モード
                lag_required = int((self.period - 1) / 2)
                if lag_required >= data_length:
                    self.logger.warning(f"期間 ({self.period}) に対するラグ期間 ({lag_required}) がデータ長 ({data_length}) 以上です。期間を調整します。")
                    # 期間を調整してフォールバック
                    adjusted_period = max(2, min(self.period, data_length // 3))
                    zlema_values = calculate_zlema_numba(src_prices, adjusted_period)
                else:
                    # 固定期間でのZLEMA計算
                    zlema_values = calculate_zlema_numba(src_prices, self.period)
                
                # 結果が全てNaNの場合のフォールバック
                if np.all(np.isnan(zlema_values)):
                    self.logger.warning("固定期間ZLEMA計算が失敗しました。EMAでフォールバックします。")
                    fallback_period = max(2, min(self.period, data_length // 4))
                    zlema_values = calculate_ema_numba(src_prices, fallback_period)

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(zlema_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = ZLEMAResult(
                values=zlema_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            self._result = result
            self._cache[data_hash] = self._result
            return self._result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None # エラー時は結果をクリア
            error_result = ZLEMAResult(
                values=np.full(data_len, np.nan),
                trend_signals=np.zeros(data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """ZLEMA値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得する"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None

    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'

    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得する"""
        if self._result is not None:
            return self._result.current_trend_value
        return 0

    def get_dynamic_periods(self) -> np.ndarray:
        """
        動的期間の値を取得する（動的期間モードのみ）
        
        Returns:
            動的期間の配列
        """
        if not self.use_dynamic_period:
            return np.array([])
        
        # 最後に計算されたドミナントサイクル値を返す
        if self._last_dc_values is not None:
            return self._last_dc_values.copy()
        
        return np.array([])

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        self._last_dc_values = None
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset() 