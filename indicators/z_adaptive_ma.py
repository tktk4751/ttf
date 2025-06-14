#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
from numba.np.unsafe.ndarray import to_fixed_tuple

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class ZAdaptiveMAResult:
    """ZAdaptiveMAの計算結果"""
    values: np.ndarray        # ZAdaptiveMAの値
    er: np.ndarray            # サイクル効率比（CER）
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str        # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@njit(['float64(float64, float64, float64)'], fastmath=True, cache=True, inline='always')
def calculate_smoothing_constant(er_val: float, fast_const: float, slow_const: float) -> float:
    """
    スムージング定数を計算する（インライン化高速版）
    
    Args:
        er_val: 効率比の値
        fast_const: 速い移動平均の定数
        slow_const: 遅い移動平均の定数
    
    Returns:
        スムージング定数（0-1の範囲）
    """
    # スムージング定数の計算
    sc = (er_val * (fast_const - slow_const) + slow_const) ** 2
    
    # 0-1の範囲に制限
    return max(0.0, min(1.0, sc))


@njit(['float64[:](float64[:], float64[:], float64, float64)'], fastmath=True, parallel=True, cache=True)
def calculate_smoothing_constants_batch(er: np.ndarray, sc_vec: np.ndarray, fast_constant: float, slow_constant: float) -> np.ndarray:
    """
    スムージング定数を一括計算する（並列高速版）
    
    Args:
        er: 効率比の配列
        sc_vec: 出力用スムージング定数の配列（既に確保済み）
        fast_constant: 速い移動平均の定数
        slow_constant: 遅い移動平均の定数
    
    Returns:
        計算済みのスムージング定数配列
    """
    length = len(er)
    
    # 最初の点は特別扱い
    if length > 1 and not np.isnan(er[1]):
        sc_vec[1] = calculate_smoothing_constant(er[1], fast_constant, slow_constant)
    
    # 残りの点を並列計算
    for i in prange(2, length):
        if np.isnan(er[i]):
            sc_vec[i] = sc_vec[i-1]  # 前回の値を使用
        else:
            sc_vec[i] = calculate_smoothing_constant(er[i], fast_constant, slow_constant)
    
    return sc_vec


@njit(['float64[:](float64[:], float64[:], float64[:], int32, int32)'], fastmath=True, parallel=False, cache=True)
def calculate_z_adaptive_ma(prices: np.ndarray, er: np.ndarray, z_ma: np.ndarray, fast_period: int, slow_period: int) -> np.ndarray:
    """
    ZAdaptiveMAを計算する（最適化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        er: 効率比の配列（ERまたはCER）
        z_ma: 出力用配列（既に確保済み）
        fast_period: 速い移動平均の期間（固定値）
        slow_period: 遅い移動平均の期間（固定値）
    
    Returns:
        ZAdaptiveMAの配列
    """
    length = len(prices)
    
    # 最初のZMAは最初の価格
    if length > 0:
        z_ma[0] = prices[0]
    
    # 定数の計算
    fast_constant = 2.0 / (fast_period + 1.0)
    slow_constant = 2.0 / (slow_period + 1.0)
    
    # スムージング定数用の配列を確保
    sc_vec = np.zeros(length, dtype=np.float64)
    
    # スムージング定数を一括計算（最適化版）
    calculate_smoothing_constants_batch(er, sc_vec, fast_constant, slow_constant)
    
    # 各時点でのZMAを計算（最適化版）
    for i in range(1, length):
        z_ma[i] = z_ma[i-1] + sc_vec[i] * (prices[i] - z_ma[i-1])
    
    return z_ma


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


class ZAdaptiveMA(Indicator):
    """
    ZAdaptiveMA (Z Adaptive Moving Average) インジケーター
    
    外部から提供されるサイクル効率比（CER）に基づいて移動平均を調整する適応型移動平均線：
    - fast期間とslow期間は固定値を使用
    - ERの値に応じてスムージング係数を動的に調整
    
    特徴:
    - トレンドが強い時：速い反応（fast係数に近づく）
    - レンジ相場時：ノイズ除去（slow係数に近づく）
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    """
    
    def __init__(
        self,
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 120,            # 遅い移動平均の期間（固定値）
        src_type: str = 'hlc3',           # ソースタイプ
        slope_index: int = 1,             # トレンド判定期間
        range_threshold: float = 0.005    # range判定の閾値
    ):
        """
        コンストラクタ
        
        Args:
            fast_period: 速い移動平均の期間（固定値）（デフォルト: 2）
            slow_period: 遅い移動平均の期間（固定値）（デフォルト: 30）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3（デフォルト）
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
        """
        # 指標名の作成
        indicator_name = f"ZAdaptiveMA(fast={fast_period}, slow={slow_period}, {src_type}, slope={slope_index}, range_th={range_threshold:.3f})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.src_type = src_type.lower()
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20  # キャッシュの最大サイズを増加
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # メモリ効率化のために事前確保したバッファ
        self._last_length = 0
        self._z_ma_buffer = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            external_er: 外部効率比データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
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
            
            # 外部ERの情報
            er_info = (0.0, 0.0)
            if external_er is not None and len(external_er) > 0:
                er_info = (float(external_er[0]), float(external_er[-1]))
            
            # 最小限のパラメータ情報
            params_sig = f"{self.fast_period}_{self.slow_period}_{self.src_type}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(er_info)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.fast_period}_{self.slow_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> ZAdaptiveMAResult:
        """
        ZAdaptiveMAを計算する（超高速版）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            ZAdaptiveMAResult: ZAdaptiveMAの値とトレンド情報を含む結果
        """
        try:
            # サイクル効率比（CER）の検証
            if external_er is None:
                raise ValueError("サイクル効率比（CER）は必須です。external_erパラメータを指定してください")
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data, external_er)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return ZAdaptiveMAResult(
                    values=cached_result.values.copy(),
                    er=cached_result.er.copy(),
                    trend_signals=cached_result.trend_signals.copy(),
                    current_trend=cached_result.current_trend,
                    current_trend_value=cached_result.current_trend_value
                )
            
            # 指定されたソースタイプの価格データを取得
            prices = PriceSource.calculate_source(data, self.src_type)
            
            # データ長の検証
            data_length = len(prices)
            
            # NumPy配列に変換（可能な限り同じメモリ領域を再利用）
            prices_np = np.asarray(prices, dtype=np.float64)
            er_np = np.asarray(external_er, dtype=np.float64)
            
            # 外部CERの長さが一致するか確認
            if len(er_np) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er_np)})がデータ長({data_length})と一致しません")
            
            # 出力用バッファを再利用または新規確保
            if self._z_ma_buffer is None or self._last_length != data_length:
                self._z_ma_buffer = np.full(data_length, np.nan, dtype=np.float64)
                self._last_length = data_length
            else:
                # 既存バッファを再利用（ゼロクリア）
                self._z_ma_buffer.fill(np.nan)
            
            # ZAdaptiveMAの計算（最適化版）- 既存バッファを再利用
            z_ma_values = calculate_z_adaptive_ma(
                prices_np,
                er_np,
                self._z_ma_buffer,
                np.int32(self.fast_period),
                np.int32(self.slow_period)
            )
            
            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(z_ma_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            # メモリ効率のため、キャッシュに保存する前にコピーを作成
            result = ZAdaptiveMAResult(
                values=z_ma_values.copy(),  # コピーを作成
                er=er_np.copy(),  # 効率比は既にnumpy配列なのでコピーして保持
                trend_signals=trend_signals.copy(),
                current_trend=current_trend,
                current_trend_value=trend_value
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = z_ma_values
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZAdaptiveMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = ZAdaptiveMAResult(
                values=np.array([]),
                er=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """ZAdaptiveMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.trend_signals.copy()
    
    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if not self._result_cache:
            return 'range'
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.current_trend

    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得する"""
        if not self._result_cache:
            return 0
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.current_trend_value
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if not self._result_cache:
            return np.array([])
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.er
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._z_ma_buffer = None
        self._last_length = 0 