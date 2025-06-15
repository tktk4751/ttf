#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Literal
import numpy as np
import pandas as pd
from numba import jit
import traceback

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .hma import HMA
    from .alma import ALMA
    from .zlema import ZLEMA
    from .hyperma import HyperMA
    from .efficiency_ratio import calculate_efficiency_ratio_for_period
except ImportError:
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
                else: return data['close'].values
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data
    
    # Dummy MA classes
    class HMA:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return type('', (), {'values': np.zeros(len(data))})()
    class ALMA:
        def __init__(self, **kwargs): pass 
        def calculate(self, data): return type('', (), {'values': np.zeros(len(data))})()
    class ZLEMA:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return type('', (), {'values': np.zeros(len(data))})()
    class HyperMA:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return type('', (), {'values': np.zeros(len(data))})()
    
    def calculate_efficiency_ratio_for_period(prices, period): return [0.5]


class AdaptiveMAResult(NamedTuple):
    """AdaptiveMA計算結果"""
    values: np.ndarray
    dynamic_periods: np.ndarray
    efficiency_ratio: np.ndarray
    is_bullish: np.ndarray
    is_bearish: np.ndarray
    current_trend: str
    is_currently_bullish: bool
    is_currently_bearish: bool


@jit(nopython=True, cache=True)
def calculate_dynamic_periods(er: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    """
    効率比に基づいて動的期間を計算する（表示用・複雑MAタイプ用）
    
    Args:
        er: 効率比の配列
        min_period: 最小期間（高効率時）
        max_period: 最大期間（低効率時）
    
    Returns:
        動的期間の配列
    """
    length = len(er)
    periods = np.full(length, np.nan)
    
    for i in range(length):
        if not np.isnan(er[i]):
            # ERが高い（1に近い）ほど短い期間、ERが低い（0に近い）ほど長い期間
            periods[i] = max_period - er[i] * (max_period - min_period)
            # 整数に丸める
            periods[i] = round(periods[i])
            # 範囲制限
            if periods[i] < min_period:
                periods[i] = min_period
            elif periods[i] > max_period:
                periods[i] = max_period
    
    return periods


@jit(nopython=True, cache=True)
def calculate_efficiency_ratio_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    効率比を計算する (Numba JIT)
    
    Args:
        prices: 価格の配列
        period: 計算期間
    
    Returns:
        効率比の配列
    """
    length = len(prices)
    er = np.full(length, np.nan)
    
    for i in range(period - 1, length):
        # 期間全体の価格変動
        total_change = abs(prices[i] - prices[i - period + 1])
        
        # 日々の変動の総和
        daily_changes = 0.0
        for j in range(i - period + 2, i + 1):
            daily_changes += abs(prices[j] - prices[j - 1])
        
        # 効率比の計算
        if daily_changes > 1e-9:
            er[i] = total_change / daily_changes
        else:


            
            er[i] = 0.0
        
        # ERを0-1の範囲にクランプ
        if er[i] > 1.0:
            er[i] = 1.0
        elif er[i] < 0.0:
            er[i] = 0.0
    
    return er


@jit(nopython=True, cache=True)
def calculate_adaptive_ma(prices: np.ndarray, er: np.ndarray, min_period: int, max_period: int, ma_type: str) -> np.ndarray:
    """
    効率比ベースの期間適応移動平均を計算する（KAMAロジック - SMA/EMA用）
    
    Args:
        prices: 価格の配列
        er: 効率比の配列
        min_period: 最小期間（高効率時）
        max_period: 最大期間（低効率時）
        ma_type: MAタイプ ('sma' or 'ema')
    
    Returns:
        適応移動平均値の配列
    """
    length = len(prices)
    adaptive_ma = np.full(length, np.nan)
    
    if length == 0:
        return adaptive_ma
    
    # スムージング定数の計算（KAMAスタイル）
    fast_sc = 2.0 / (min_period + 1.0)  # 高効率時のスムージング定数
    slow_sc = 2.0 / (max_period + 1.0)  # 低効率時のスムージング定数
    
    for i in range(length):
        if np.isnan(er[i]):
            continue
            
        # 効率比に基づいてスムージング定数を動的に調整
        # ERが高い(1.0)時は高感度(fast_sc)、ERが低い(0.0)時は低感度(slow_sc)
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == 0 or np.isnan(adaptive_ma[i-1]):
            # 初期値は現在価格
            adaptive_ma[i] = prices[i]
        else:
            if ma_type == 'ema':
                # EMAスタイル: adaptive_ma[i] = sc * prices[i] + (1 - sc) * adaptive_ma[i-1]
                adaptive_ma[i] = sc * prices[i] + (1.0 - sc) * adaptive_ma[i-1]
            else:  # SMAまたはその他
                # KAMAスタイル: adaptive_ma[i] = adaptive_ma[i-1] + sc * (prices[i] - adaptive_ma[i-1])
                adaptive_ma[i] = adaptive_ma[i-1] + sc * (prices[i] - adaptive_ma[i-1])
    
    return adaptive_ma


@jit(nopython=True, cache=True)
def calculate_trend_signals(values: np.ndarray, slope_index: int) -> tuple:
    """
    トレンド信号を計算する (Numba JIT)
    
    Args:
        values: インジケーター値の配列
        slope_index: スロープ判定期間
    
    Returns:
        tuple: (is_bullish, is_bearish) のNumPy配列
    """
    length = len(values)
    is_bullish = np.full(length, False)
    is_bearish = np.full(length, False)
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            if values[i] > values[i - slope_index]:
                is_bullish[i] = True
                is_bearish[i] = False
            elif values[i] < values[i - slope_index]:
                is_bullish[i] = False
                is_bearish[i] = True
    
    return is_bullish, is_bearish


@jit(nopython=True, cache=True)
def calculate_current_trend(is_bullish: np.ndarray, is_bearish: np.ndarray) -> tuple:
    """
    現在のトレンド状態を計算する (Numba JIT)
    
    Args:
        is_bullish: 上昇トレンド判定配列
        is_bearish: 下降トレンド判定配列
    
    Returns:
        tuple: (current_trend_index, is_currently_bullish, is_currently_bearish)
               current_trend_index: 0=neutral, 1=bullish, 2=bearish
    """
    length = len(is_bullish)
    if length == 0:
        return 0, False, False
    
    latest_bullish = is_bullish[-1]
    latest_bearish = is_bearish[-1]
    
    if latest_bullish:
        return 1, True, False   # bullish
    elif latest_bearish:
        return 2, False, True   # bearish
    else:
        return 0, False, False  # neutral


@jit(nopython=True, cache=True)
def calculate_adaptive_ma_kama_style(theoretical_values: np.ndarray, er: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    """
    理論値に基づいてKAMAスタイルの適応移動平均を計算する
    
    Args:
        theoretical_values: 各時点での理論MA値（HMA, ALMA, ZLEMA, HyperMAなど）
        er: 効率比の配列
        min_period: 最小期間（高効率時）
        max_period: 最大期間（低効率時）
    
    Returns:
        滑らかな適応MA値の配列
    """
    length = len(theoretical_values)
    adaptive_ma = np.full(length, np.nan)
    
    if length == 0:
        return adaptive_ma
    
    # スムージング定数の計算（KAMAスタイル）
    fast_sc = 2.0 / (min_period + 1.0)  # 高効率時のスムージング定数
    slow_sc = 2.0 / (max_period + 1.0)  # 低効率時のスムージング定数
    
    for i in range(length):
        if np.isnan(er[i]) or np.isnan(theoretical_values[i]):
            continue
            
        # 効率比に基づいてスムージング定数を動的に調整
        # ERが高い(1.0)時は高感度(fast_sc)、ERが低い(0.0)時は低感度(slow_sc)
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == 0 or np.isnan(adaptive_ma[i-1]):
            # 初期値は理論値
            adaptive_ma[i] = theoretical_values[i]
        else:
            # KAMAスタイル: adaptive_ma[i] = adaptive_ma[i-1] + sc * (theoretical_value[i] - adaptive_ma[i-1])
            adaptive_ma[i] = adaptive_ma[i-1] + sc * (theoretical_values[i] - adaptive_ma[i-1])
    
    return adaptive_ma


class AdaptiveMA(Indicator):
    """
    期間適応移動平均線 (Period-Adaptive MA) インジケーター
    
    効率比（ER）に基づいて動的に調整する適応移動平均線。
    全MAタイプでKAMAロジックによる統一的な滑らかな適応動作を実現。
    
    特徴:
    - SMA/EMA: KAMAロジック（効率比→スムージング定数変換）で連続的計算
    - HMA/ALMA/ZLEMA/HyperMA: 理論値をKAMAロジックでスムージング
    - 全MAタイプで統一的な滑らかな適応動作
    - 効率的なNumba実装
    - ギザギザを完全に抑制した美しい曲線
    
    計算原理:
    - 高効率時（ER≈1.0）: 価格変動に敏感に反応
    - 低効率時（ER≈0.0）: 安定した平滑化動作
    - 中間効率時: 効率比に応じて連続的にブレンド
    """
    
    SUPPORTED_MA_TYPES = {
        'sma': 'simple',
        'ema': 'exponential', 
        'hma': HMA,
        'alma': ALMA,
        'zlema': ZLEMA,
        'hyperma': HyperMA
    }
    
    def __init__(self,
                 ma_type: str = 'sma',
                 min_period: int = 2,
                 max_period: int = 50,
                 er_period: int = 10,
                 src_type: str = 'close',
                 slope_index: int = 1):
        """
        コンストラクタ
        
        Args:
            ma_type: MAのタイプ ('sma', 'ema', 'hma', 'alma', 'zlema', 'hyperma')
            min_period: 最小期間（高効率時）
            max_period: 最大期間（低効率時）
            er_period: 効率比の計算期間
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間
        """
        if ma_type.lower() not in self.SUPPORTED_MA_TYPES:
            raise ValueError(f"サポートされていないMAタイプ: {ma_type}. サポート: {list(self.SUPPORTED_MA_TYPES.keys())}")
        
        if not isinstance(slope_index, int) or slope_index < 1:
            raise ValueError(f"slope_indexは1以上の整数である必要があります: {slope_index}")
        
        if not isinstance(min_period, int) or min_period < 1:
            raise ValueError(f"min_periodは1以上の整数である必要があります: {min_period}")
        
        if not isinstance(max_period, int) or max_period < min_period:
            raise ValueError(f"max_periodはmin_period以上である必要があります: {max_period}")
            
        super().__init__(f"AdaptiveMA(type={ma_type},min={min_period},max={max_period},er={er_period},src={src_type},slope={slope_index})")
        
        self.ma_type = ma_type.lower()
        self.min_period = min_period
        self.max_period = max_period
        self.er_period = er_period
        self.src_type = src_type.lower()
        self.slope_index = slope_index
        
        self._cache = {}
        self._result: Optional[AdaptiveMAResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(data, pd.DataFrame):
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                data_hash_val = hash(data.tobytes())
            else:
                data_hash_val = hash(str(data))
        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. fallbackを使用します。")
            data_hash_val = hash(str(data))

        param_str = f"type={self.ma_type}_min={self.min_period}_max={self.max_period}_er={self.er_period}_src={self.src_type}_slope={self.slope_index}"
        return f"{data_hash_val}_{param_str}"

    def _calculate_theoretical_ma_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        各MAタイプの理論値を計算する（KAMAスムージング用）
        
        Args:
            data: 価格データ
            
        Returns:
            理論MA値の配列
        """
        try:
            # 中間期間を使用して理論値を計算
            mid_period = (self.min_period + self.max_period) // 2
            
            if self.ma_type == 'hma':
                ma_instance = HMA(period=mid_period, src_type=self.src_type)
            elif self.ma_type == 'alma':
                ma_instance = ALMA(period=mid_period, src_type=self.src_type)
            elif self.ma_type == 'zlema':
                ma_instance = ZLEMA(period=mid_period, src_type=self.src_type)
            elif self.ma_type == 'hyperma':
                ma_instance = HyperMA(length=mid_period, src_type=self.src_type)
            else:
                return np.full(len(data), np.nan)
            
            # 理論値を計算
            result = ma_instance.calculate(data)
            theoretical_values = result.values if hasattr(result, 'values') else result
            
            if theoretical_values is None:
                self.logger.warning(f"MAタイプ'{self.ma_type}'の理論値計算結果がNoneです。")
                return np.full(len(data), np.nan)
                
            return theoretical_values
            
        except Exception as e:
            self.logger.error(f"理論MA値計算エラー ({self.ma_type}): {e}")
            return np.full(len(data), np.nan)

    def _calculate_adaptive_complex_ma(self, data: Union[pd.DataFrame, np.ndarray], 
                                       efficiency_ratio: np.ndarray) -> np.ndarray:
        """
        複雑MAタイプのKAMA風適応計算（滑らか版）
        
        KAMAロジック: 理論値に基づく連続的スムージング
        
        Args:
            data: 価格データ
            efficiency_ratio: 効率比配列
        
        Returns:
            滑らかな適応MAの配列
        """
        try:
            # 理論MA値を計算
            theoretical_values = self._calculate_theoretical_ma_values(data)
            
            # C-contiguous配列に変換（Numba用）
            if not theoretical_values.flags['C_CONTIGUOUS']:
                theoretical_values = np.ascontiguousarray(theoretical_values)
            
            # KAMAスタイルで適応MA計算
            adaptive_values = calculate_adaptive_ma_kama_style(
                theoretical_values, efficiency_ratio, self.min_period, self.max_period
            )
            
            # 有効値の統計
            valid_count = np.sum(~np.isnan(adaptive_values))
            self.logger.debug(f"MAタイプ'{self.ma_type}' (KAMAロジック): 有効値 {valid_count}/{len(adaptive_values)}")
            
            return adaptive_values
            
        except Exception as e:
            self.logger.error(f"複雑MA適応計算エラー ({self.ma_type}): {e}")
            return np.full(len(efficiency_ratio), np.nan)

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> AdaptiveMAResult:
        """
        期間適応移動平均を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            AdaptiveMAResult: 適応MA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。")
            return self._empty_result(0)

        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                if len(self._result.values) == current_data_len:
                    return AdaptiveMAResult(
                        values=self._result.values.copy(),
                        dynamic_periods=self._result.dynamic_periods.copy(),
                        efficiency_ratio=self._result.efficiency_ratio.copy(),
                        is_bullish=self._result.is_bullish.copy(),
                        is_bearish=self._result.is_bearish.copy(),
                        current_trend=self._result.current_trend,
                        is_currently_bullish=self._result.is_currently_bullish,
                        is_currently_bearish=self._result.is_currently_bearish
                    )
                else:
                    del self._cache[data_hash]
                    self._result = None

            # 価格ソースを取得
            src_prices = PriceSource.calculate_source(data, self.src_type)

            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗しました。")
                return self._empty_result(current_data_len)

            # Numba用にfloat64配列に変換
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                    return self._empty_result(current_data_len)

            data_length = len(src_prices)
            if data_length < self.er_period:
                self.logger.warning(f"データ長({data_length})が効率比計算期間({self.er_period})より短いです。")
                return self._empty_result(current_data_len)

            # C-contiguous配列に変換
            if not src_prices.flags['C_CONTIGUOUS']:
                src_prices = np.ascontiguousarray(src_prices)

            # 効率比の計算
            efficiency_ratio = calculate_efficiency_ratio_numba(src_prices, self.er_period)

            # 動的期間の計算（表示用のみ）
            dynamic_periods = calculate_dynamic_periods(efficiency_ratio, self.min_period, self.max_period)

            # 適応移動平均の計算（全MAタイプで統一的な滑らかロジック）
            if self.ma_type in ['sma', 'ema']:
                # SMA/EMA: 価格ベースKAMAロジック（スムージング定数変換）
                adaptive_ma_values = calculate_adaptive_ma(src_prices, efficiency_ratio, 
                                                         self.min_period, self.max_period, self.ma_type)
            else:
                # 複雑MAタイプ: 理論値ベースKAMAロジック（連続的スムージング）
                adaptive_ma_values = self._calculate_adaptive_complex_ma(data, efficiency_ratio)

            # トレンド信号の計算
            is_bullish, is_bearish = calculate_trend_signals(adaptive_ma_values, self.slope_index)

            # 現在のトレンド状態を計算
            trend_index, currently_bullish, currently_bearish = calculate_current_trend(is_bullish, is_bearish)
            trend_names = ['neutral', 'bullish', 'bearish']
            current_trend = trend_names[trend_index]

            result = AdaptiveMAResult(
                values=adaptive_ma_values,
                dynamic_periods=dynamic_periods,
                efficiency_ratio=efficiency_ratio,
                is_bullish=is_bullish,
                is_bearish=is_bearish,
                current_trend=current_trend,
                is_currently_bullish=currently_bullish,
                is_currently_bearish=currently_bearish
            )

            self._result = result
            self._cache[data_hash] = self._result
            
            return AdaptiveMAResult(
                values=result.values.copy(),
                dynamic_periods=result.dynamic_periods.copy(),
                efficiency_ratio=result.efficiency_ratio.copy(),
                is_bullish=result.is_bullish.copy(),
                is_bearish=result.is_bearish.copy(),
                current_trend=result.current_trend,
                is_currently_bullish=result.is_currently_bullish,
                is_currently_bearish=result.is_currently_bearish
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AdaptiveMA計算中にエラー: {error_msg}\n{stack_trace}")
            return self._empty_result(current_data_len)

    def _empty_result(self, length: int) -> AdaptiveMAResult:
        """空の結果を生成する"""
        return AdaptiveMAResult(
            values=np.full(length, np.nan),
            dynamic_periods=np.full(length, np.nan),
            efficiency_ratio=np.full(length, np.nan),
            is_bullish=np.full(length, False, dtype=bool),
            is_bearish=np.full(length, False, dtype=bool),
            current_trend='neutral',
            is_currently_bullish=False,
            is_currently_bearish=False
        )

    def get_values(self) -> Optional[np.ndarray]:
        """適応MA値のみを取得する"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """動的期間を取得する"""
        if self._result is not None:
            return self._result.dynamic_periods.copy()
        return None

    def get_efficiency_ratio(self) -> Optional[np.ndarray]:
        """効率比を取得する"""
        if self._result is not None:
            return self._result.efficiency_ratio.copy()
        return None

    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if self._result is not None:
            return self._result.current_trend
        return 'neutral'

    def is_currently_bullish_trend(self) -> bool:
        """現在が上昇トレンドかどうか"""
        if self._result is not None:
            return self._result.is_currently_bullish
        return False

    def is_currently_bearish_trend(self) -> bool:
        """現在が下降トレンドかどうか"""
        if self._result is not None:
            return self._result.is_currently_bearish
        return False

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 