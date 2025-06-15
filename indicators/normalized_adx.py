#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import traceback

try:
    from .indicator import Indicator
    from .ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 14.0)
        def reset(self): pass


class NormalizedADXResult(NamedTuple):
    """正規化ADX計算結果"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        prev_close: 1期前の終値の配列
    
    Returns:
        True Range の配列
    """
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(tr1, tr2)
    tr = np.maximum(tr, tr3)
    
    return tr


@jit(nopython=True)
def calculate_dm(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Directional Movement（+DM, -DM）を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if (up > down) and (up > 0):
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        
        if (down > up) and (down > 0):
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    return plus_dm, minus_dm


@jit(nopython=True)
def calculate_normalized_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    正規化されたADXを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 計算期間
    
    Returns:
        正規化されたADX値の配列（0-1の範囲）
    """
    length = len(high)
    tr = np.zeros(length)
    tr[0] = high[0] - low[0]
    tr[1:] = calculate_tr(high[1:], low[1:], close[:-1])
    
    # +DM, -DMの計算
    plus_dm, minus_dm = calculate_dm(high, low)
    
    # 指数移動平均の計算用の係数
    alpha = 2.0 / (period + 1.0)
    
    # TR, +DM, -DMの平滑化
    smoothed_tr = np.zeros(length)
    smoothed_plus_dm = np.zeros(length)
    smoothed_minus_dm = np.zeros(length)
    
    # 最初の値の初期化
    smoothed_tr[0] = tr[0]
    smoothed_plus_dm[0] = plus_dm[0]
    smoothed_minus_dm[0] = minus_dm[0]
    
    # 指数移動平均による平滑化
    for i in range(1, length):
        smoothed_tr[i] = (tr[i] * alpha) + (smoothed_tr[i-1] * (1 - alpha))
        smoothed_plus_dm[i] = (plus_dm[i] * alpha) + (smoothed_plus_dm[i-1] * (1 - alpha))
        smoothed_minus_dm[i] = (minus_dm[i] * alpha) + (smoothed_minus_dm[i-1] * (1 - alpha))
    
    # +DI, -DIの計算
    nonzero_tr = np.where(smoothed_tr == 0, 1e-10, smoothed_tr)
    plus_di = smoothed_plus_dm / nonzero_tr
    minus_di = smoothed_minus_dm / nonzero_tr
    
    # DXの計算
    dx_sum = plus_di + minus_di
    # ゼロ除算を防ぐ
    nonzero_sum = np.where(dx_sum == 0, 1e-10, dx_sum)
    dx = np.abs(plus_di - minus_di) / nonzero_sum
    
    # ADXの計算（DXの平滑化）
    adx = np.zeros(length)
    adx[0] = dx[0]
    
    for i in range(1, length):
        adx[i] = (dx[i] * alpha) + (adx[i-1] * (1 - alpha))
    
    return adx


@jit(nopython=True, cache=True)
def calculate_dynamic_nadx_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period_array: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    動的期間で正規化ADXを計算する（Numba JIT）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period_array: 各時点での期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        正規化ADX値の配列
    """
    length = len(high)
    result = np.full(length, np.nan)
    
    # 各時点での動的正規化ADXを計算
    for i in range(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(period_array[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置までのデータを取得
        start_idx = max(0, i - curr_period * 2)  # ADXには余裕をもったウィンドウが必要
        window_high = high[start_idx:i+1]
        window_low = low[start_idx:i+1]
        window_close = close[start_idx:i+1]
        
        # 現在の期間で正規化ADXを計算
        if len(window_high) >= curr_period:
            nadx_values = calculate_normalized_adx(window_high, window_low, window_close, curr_period)
            
            # 最後の値を正規化ADXとして使用
            if len(nadx_values) > 0 and not np.isnan(nadx_values[-1]):
                result[i] = nadx_values[-1]
    
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


class NormalizedADX(Indicator):
    """
    正規化ADX（Normalized Average Directional Index）インジケーター
    
    通常のADXを0-1の範囲に正規化したインジケーター
    トレンドの強さを測定する
    - 0.25以上: 強いトレンド
    - 0.20以下: トレンドなし
    
    特徴:
    - 固定期間または動的期間（ドミナントサイクル）での計算に対応
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    """
    
    def __init__(self, 
                 period: int = 13,
                 use_dynamic_period: bool = True,
                 cycle_part: float = 0.5,
                 detector_type: str = 'cycle_period2',
                 max_cycle: int = 55,
                 min_cycle: int = 5,
                 max_output: int = 15,
                 min_output: int = 3,
                 slope_index: int = 1,
                 range_threshold: float = 0.005,
                 lp_period: int = 10,
                 hp_period: int = 48):
        """
        コンストラクタ
        
        Args:
            period: 期間（デフォルト: 13）
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
        super().__init__(f"NADX(p={period}{dynamic_str},slope={slope_index},range_th={range_threshold:.3f})")
        
        self.period = period
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
                src_type='hlc3',  # NADXはHLCデータを使用
                lp_period=self.lp_period,
                hp_period=self.hp_period
            )
        
        self._cache = {}
        self._result: Optional[NormalizedADXResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は形状と端点でハッシュを計算
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                # NumPy配列の場合はバイト表現でハッシュ
                data_hash_val = hash(data.tobytes())
            else:
                # その他のデータ型は文字列表現でハッシュ化
                data_hash_val = hash(str(data))

        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. データ全体の文字列表現を使用します。", exc_info=True)
            data_hash_val = hash(str(data)) # fallback

        # パラメータ文字列の作成
        if self.use_dynamic_period:
            param_str = (f"p={self.period}_dynamic={self.detector_type}_{self.max_output}_{self.min_output}_"
                        f"slope={self.slope_index}_range_th={self.range_threshold:.3f}")
        else:
            param_str = f"p={self.period}_slope={self.slope_index}_range_th={self.range_threshold:.3f}"

        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> NormalizedADXResult:
        """
        正規化ADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            NormalizedADXResult: 正規化ADX値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            empty_result = NormalizedADXResult(
                values=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return empty_result
            
        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                    return NormalizedADXResult(
                        values=self._result.values.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_trend=self._result.current_trend,
                        current_trend_value=self._result.current_trend_value
                    )
                else:
                    self.logger.debug(f"キャッシュのデータ長が異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3] # close

            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(high, np.ndarray):
                high = np.array(high)
            if not isinstance(low, np.ndarray):
                low = np.array(low)
            if not isinstance(close, np.ndarray):
                close = np.array(close)
            
            if high.dtype != np.float64:
                high = high.astype(np.float64)
            if low.dtype != np.float64:
                low = low.astype(np.float64)
            if close.dtype != np.float64:
                close = close.astype(np.float64)

            data_length = len(high)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の結果を返します。")
                empty_result = NormalizedADXResult(
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
                    error_result = NormalizedADXResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # ドミナントサイクルの計算
                df_data = data if isinstance(data, pd.DataFrame) else pd.DataFrame({'open': data[:, 0], 'high': high, 'low': low, 'close': close})
                dc_values = self.dc_detector.calculate(df_data)
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
                
                # データ長の検証
                min_len_required = max_period_value + 1
                if data_length < min_len_required:
                    self.logger.warning(f"データ長({data_length})が必要な最大期間({min_len_required})より短いため、計算できません。")
                    error_result = NormalizedADXResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # 動的正規化ADXの計算
                nadx_values = calculate_dynamic_nadx_numba(high, low, close, period_array, max_period_value)
            else:
                # 固定期間モード
                if self.period > data_length:
                    self.logger.warning(f"期間 ({self.period}) がデータ長 ({data_length}) より大きいです。")
                
                # 正規化ADXの計算（高速化版）
                nadx_values = calculate_normalized_adx(high, low, close, self.period)

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(nadx_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = NormalizedADXResult(
                values=nadx_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            # 計算結果を保存
            self._result = result
            self._cache[data_hash] = self._result
            self._values = nadx_values  # Indicatorクラスの標準出力
            return NormalizedADXResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value
            )
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"NormalizedADX '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            error_result = NormalizedADXResult(
                values=np.full(current_data_len, np.nan),
                trend_signals=np.zeros(current_data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """正規化ADX値のみを取得する（後方互換性のため）"""
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
        """インジケータの状態（キャッシュ、結果）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        self._last_dc_values = None
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 