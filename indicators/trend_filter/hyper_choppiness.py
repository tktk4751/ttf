#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Any, Union, Tuple, Optional, List
from indicators.indicator import Indicator
from indicators.price_source import PriceSource


@njit(cache=True)
def roofing_filter_core(prices: np.ndarray, 
                        hp_cutoff: float, 
                        ss_band_edge: float) -> np.ndarray:
    """
    ルーフィングフィルターのコア計算
    
    Args:
        prices: 価格データ
        hp_cutoff: ハイパスフィルターのカットオフ周波数
        ss_band_edge: スーパースムーサーのバンドエッジ周波数
    
    Returns:
        フィルター済み価格データ
    """
    n = len(prices)
    if n < 10:
        return np.full(n, np.nan)
    
    # ハイパスフィルター
    alpha1 = (np.cos(2 * np.pi / hp_cutoff) + np.sin(2 * np.pi / hp_cutoff) - 1) / np.cos(2 * np.pi / hp_cutoff)
    hp_values = np.full(n, np.nan)
    
    for i in range(2, n):
        hp_values[i] = (1 - alpha1 / 2) * (1 - alpha1 / 2) * (prices[i] - 2 * prices[i-1] + prices[i-2])
        if i >= 2:
            hp_values[i] += 2 * (1 - alpha1) * hp_values[i-1] - (1 - alpha1) * (1 - alpha1) * hp_values[i-2]
    
    # スーパースムーサー
    alpha2 = np.exp(-np.sqrt(2) * np.pi / ss_band_edge)
    c3 = -alpha2 * alpha2
    c2 = 2 * alpha2 * np.cos(np.sqrt(2) * np.pi / ss_band_edge)
    c1 = 1 - c2 - c3
    
    roofing = np.full(n, np.nan)
    
    for i in range(2, n):
        if not np.isnan(hp_values[i]):
            roofing[i] = c1 * hp_values[i]
            if i >= 1 and not np.isnan(roofing[i-1]):
                roofing[i] += c2 * roofing[i-1]
            if i >= 2 and not np.isnan(roofing[i-2]):
                roofing[i] += c3 * roofing[i-2]
    
    return roofing


@njit(cache=True) 
def calculate_cycle_period_core(roofing_values: np.ndarray,
                                min_period: int = 10,
                                max_period: int = 50) -> np.ndarray:
    """
    ルーフィングフィルター済み価格からサイクル期間を計算
    
    Args:
        roofing_values: ルーフィングフィルター済み価格
        min_period: 最小サイクル期間
        max_period: 最大サイクル期間
    
    Returns:
        サイクル期間の配列
    """
    n = len(roofing_values)
    periods = np.full(n, np.nan)
    
    for i in range(max_period, n):
        max_corr = 0.0
        best_period = min_period
        
        for period in range(min_period, max_period + 1):
            # 自己相関を計算
            correlation = 0.0
            count = 0
            
            for j in range(period):
                if (i - j >= 0 and i - j - period >= 0 and 
                    not np.isnan(roofing_values[i - j]) and 
                    not np.isnan(roofing_values[i - j - period])):
                    correlation += roofing_values[i - j] * roofing_values[i - j - period]
                    count += 1
            
            if count > 0:
                correlation /= count
                if correlation > max_corr:
                    max_corr = correlation
                    best_period = period
        
        periods[i] = best_period
    
    return periods


@njit(cache=True)
def calculate_choppiness_core(high_filtered: np.ndarray,
                             low_filtered: np.ndarray,
                             close_filtered: np.ndarray,
                             periods: np.ndarray) -> np.ndarray:
    """
    ルーフィングフィルター済み価格でチョピネスインデックスを計算
    
    Args:
        high_filtered: フィルター済み高値
        low_filtered: フィルター済み安値  
        close_filtered: フィルター済み終値
        periods: 動的サイクル期間
    
    Returns:
        チョピネスインデックス値
    """
    n = len(high_filtered)
    choppiness = np.full(n, np.nan)
    
    # より早い開始点に変更し、NaN値のチェックを改善
    for i in range(20, n):  # より早い開始点
        period = int(periods[i]) if not np.isnan(periods[i]) else 14
        period = max(5, min(30, period))  # 期間を緩和
        
        if i < period:
            continue
            
        # ATRの計算（NaN値のチェックを改善）
        atr_sum = 0.0
        atr_count = 0
        
        for j in range(i - period + 1, i + 1):
            if j > 0:
                # フィルター済み価格がNaNの場合は元の価格を使用
                h_val = high_filtered[j] if not np.isnan(high_filtered[j]) else 0.0
                l_val = low_filtered[j] if not np.isnan(low_filtered[j]) else 0.0
                c_prev = close_filtered[j-1] if not np.isnan(close_filtered[j-1]) else 0.0
                
                # 全てゼロの場合はスキップ
                if h_val == 0.0 and l_val == 0.0 and c_prev == 0.0:
                    continue
                
                tr1 = abs(h_val - l_val)
                tr2 = abs(h_val - c_prev)
                tr3 = abs(l_val - c_prev)
                tr = max(tr1, max(tr2, tr3))
                
                if tr > 0:
                    atr_sum += tr
                    atr_count += 1
        
        if atr_count == 0 or atr_sum == 0:
            continue
            
        atr = atr_sum / atr_count
        
        # 期間内の最高値と最低値（NaN値を適切に処理）
        period_high = -float('inf')
        period_low = float('inf')
        valid_count = 0
        
        for j in range(i - period + 1, i + 1):
            h_val = high_filtered[j] if not np.isnan(high_filtered[j]) else 0.0
            l_val = low_filtered[j] if not np.isnan(low_filtered[j]) else 0.0
            
            if h_val > 0 and l_val > 0:
                period_high = max(period_high, h_val)
                period_low = min(period_low, l_val)
                valid_count += 1
        
        if (valid_count > 0 and 
            period_high > period_low and 
            period_high != -float('inf') and 
            period_low != float('inf') and
            atr > 0):
            
            # チョピネスインデックス計算
            try:
                chop_raw = 100 * np.log10(atr * period / (period_high - period_low)) / np.log10(period)
                choppiness[i] = max(0, min(100, chop_raw))  # 0-100に制限
            except:
                continue
    
    return choppiness


@njit(cache=True) 
def smooth_values_core(values: np.ndarray, period: int) -> np.ndarray:
    """
    単純移動平均で値を平滑化
    
    Args:
        values: 平滑化する値
        period: 平滑化期間
    
    Returns:
        平滑化された値
    """
    n = len(values)
    smoothed = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sum_val = 0.0
        count = 0
        
        for j in range(i - period + 1, i + 1):
            if not np.isnan(values[j]):
                sum_val += values[j]
                count += 1
        
        if count > 0:
            smoothed[i] = sum_val / count
    
    return smoothed


class HyperChoppiness(Indicator):
    """
    Hyper Choppiness Index - HyperERのルーフィングフィルターを適用したチョピネスインデックス
    
    X-Choppinessにルーフィングフィルターとサイクル検出を追加した版
    計算フロー:
    1. ソース価格からサイクル期間を計算（オプション）
    2. 高値・安値それぞれにルーフィングフィルターを適用
    3. サイクル期間とフィルター済み価格でチョピネスインデックス計算
    4. 結果を平滑化
    """
    
    def __init__(self,
                 # ルーフィングフィルターパラメータ
                 use_roofing_filter: bool = True,
                 roofing_hp_cutoff: float = 48.0,
                 roofing_ss_band_edge: float = 10.0,
                 
                 # サイクル検出パラメータ  
                 use_cycle_detection: bool = True,
                 min_cycle_period: int = 10,
                 max_cycle_period: int = 50,
                 default_period: int = 14,
                 
                 # 平滑化パラメータ
                 smoothing_period: int = 3,
                 
                 # 価格ソース
                 source_type: str = 'hlc3',
                 
                 **kwargs):
        """
        初期化
        
        Args:
            use_roofing_filter: ルーフィングフィルターを使用するか
            roofing_hp_cutoff: ルーフィングフィルターのハイパスカットオフ
            roofing_ss_band_edge: ルーフィングフィルターのスーパースムーサーバンドエッジ
            use_cycle_detection: サイクル検出を使用するか
            min_cycle_period: 最小サイクル期間
            max_cycle_period: 最大サイクル期間
            default_period: デフォルトのチョピネス計算期間
            smoothing_period: 最終結果の平滑化期間
            source_type: 価格ソースタイプ
        """
        super().__init__(name='HyperChoppiness')
        
        # パラメータの保存
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        
        self.use_cycle_detection = use_cycle_detection
        self.min_cycle_period = min_cycle_period
        self.max_cycle_period = max_cycle_period
        self.default_period = default_period
        
        self.smoothing_period = smoothing_period
        self.source_type = source_type
        
        # 結果の保存
        self.hyper_choppiness = None
        self.cycle_periods = None
        self.roofing_high = None
        self.roofing_low = None
        self.raw_choppiness = None
        
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Hyper Choppinessの計算
        
        Args:
            data: OHLCV データ
            
        Returns:
            計算結果の辞書
        """
        try:
            # データの準備
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")
                    
                open_prices = data['open'].values
                high_prices = data['high'].values
                low_prices = data['low'].values
                close_prices = data['close'].values
                volume = data['volume'].values if 'volume' in data.columns else None
                
            elif isinstance(data, np.ndarray):
                if data.shape[1] < 4:
                    raise ValueError("Array must have at least 4 columns (OHLC)")
                open_prices = data[:, 0]
                high_prices = data[:, 1]
                low_prices = data[:, 2]
                close_prices = data[:, 3]
                volume = data[:, 4] if data.shape[1] > 4 else None
            else:
                raise ValueError("Data must be pandas DataFrame or numpy array")
                
            n = len(close_prices)
            
            # ソース価格の計算
            if isinstance(data, pd.DataFrame):
                source_prices = PriceSource.calculate_source(data, self.source_type)
            else:
                # NumPy配列の場合は手動でhlc3を計算
                if self.source_type == 'hlc3':
                    source_prices = (high_prices + low_prices + close_prices) / 3
                elif self.source_type == 'close':
                    source_prices = close_prices
                elif self.source_type == 'hl2':
                    source_prices = (high_prices + low_prices) / 2
                else:
                    source_prices = close_prices
            
            # サイクル期間の計算
            if self.use_cycle_detection and self.use_roofing_filter:
                # ルーフィングフィルターをソース価格に適用してからサイクル検出
                roofing_source = roofing_filter_core(
                    source_prices, 
                    self.roofing_hp_cutoff, 
                    self.roofing_ss_band_edge
                )
                self.cycle_periods = calculate_cycle_period_core(
                    roofing_source,
                    self.min_cycle_period,
                    self.max_cycle_period
                )
            else:
                # 固定期間を使用
                self.cycle_periods = np.full(n, float(self.default_period))
            
            # 高値・安値にルーフィングフィルターを適用
            if self.use_roofing_filter:
                self.roofing_high = roofing_filter_core(
                    high_prices, 
                    self.roofing_hp_cutoff, 
                    self.roofing_ss_band_edge
                )
                self.roofing_low = roofing_filter_core(
                    low_prices, 
                    self.roofing_hp_cutoff, 
                    self.roofing_ss_band_edge
                )
                # ルーフィングフィルターを終値にも適用（ATR計算用）
                roofing_close = roofing_filter_core(
                    close_prices, 
                    self.roofing_hp_cutoff, 
                    self.roofing_ss_band_edge
                )
                
                # ルーフィングフィルターが全てNaNの場合は元の価格を使用
                if np.all(np.isnan(self.roofing_high)):
                    self.logger.warning("Roofing filter returned all NaN values, using original prices")
                    self.roofing_high = high_prices.copy()
                    self.roofing_low = low_prices.copy()
                    roofing_close = close_prices.copy()
            else:
                # フィルターなしの場合は元の価格を使用
                self.roofing_high = high_prices.copy()
                self.roofing_low = low_prices.copy()
                roofing_close = close_prices.copy()
            
            # チョピネスインデックスの計算
            self.raw_choppiness = calculate_choppiness_core(
                self.roofing_high,
                self.roofing_low,
                roofing_close,
                self.cycle_periods
            )
            
            # 最終的な平滑化
            self.hyper_choppiness = smooth_values_core(
                self.raw_choppiness,
                self.smoothing_period
            )
            
            # メインの結果配列を返す（基底クラスの要求に合わせる）
            self.logger.info(f"Hyper Choppiness calculation completed. Data points: {n}")
            return self.hyper_choppiness
            
        except Exception as e:
            self.logger.error(f"Error in Hyper Choppiness calculation: {str(e)}")
            raise
    
    def get_choppiness_values(self) -> np.ndarray:
        """
        平滑化済みのHyper Choppiness値を取得
        
        Returns:
            Hyper Choppiness値の配列
        """
        if self.hyper_choppiness is None:
            raise ValueError("Hyper Choppiness not calculated. Call calculate() first.")
        return self.hyper_choppiness.copy()
    
    def get_raw_choppiness(self) -> np.ndarray:
        """
        生のチョピネス値（平滑化前）を取得
        
        Returns:
            生のチョピネス値の配列
        """
        if self.raw_choppiness is None:
            raise ValueError("Raw choppiness not calculated. Call calculate() first.")
        return self.raw_choppiness.copy()
    
    def get_cycle_periods(self) -> np.ndarray:
        """
        検出されたサイクル期間を取得
        
        Returns:
            サイクル期間の配列
        """
        if self.cycle_periods is None:
            raise ValueError("Cycle periods not calculated. Call calculate() first.")
        return self.cycle_periods.copy()
    
    def get_roofing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ルーフィングフィルター済みの高値・安値を取得
        
        Returns:
            (フィルター済み高値, フィルター済み安値)のタプル
        """
        if self.roofing_high is None or self.roofing_low is None:
            raise ValueError("Roofing filter data not calculated. Call calculate() first.")
        return self.roofing_high.copy(), self.roofing_low.copy()
    
    def is_choppy(self, threshold: float = 61.8) -> np.ndarray:
        """
        チョピネス判定（閾値以上でチョピー相場）
        
        Args:
            threshold: チョピネス判定の閾値
            
        Returns:
            チョピー相場の判定（True/False）
        """
        if self.hyper_choppiness is None:
            raise ValueError("Hyper Choppiness not calculated. Call calculate() first.")
        return self.hyper_choppiness >= threshold
    
    def is_trending(self, threshold: float = 38.2) -> np.ndarray:
        """
        トレンド判定（閾値以下でトレンド相場）
        
        Args:
            threshold: トレンド判定の閾値
            
        Returns:
            トレンド相場の判定（True/False）
        """
        if self.hyper_choppiness is None:
            raise ValueError("Hyper Choppiness not calculated. Call calculate() first.")
        return self.hyper_choppiness <= threshold
    
    def get_market_regime(self, 
                         choppy_threshold: float = 61.8,
                         trending_threshold: float = 38.2) -> np.ndarray:
        """
        相場レジーム分類（チョピー/中立/トレンド）
        
        Args:
            choppy_threshold: チョピー相場の閾値
            trending_threshold: トレンド相場の閾値
            
        Returns:
            相場レジーム (-1: トレンド, 0: 中立, 1: チョピー)
        """
        if self.hyper_choppiness is None:
            raise ValueError("Hyper Choppiness not calculated. Call calculate() first.")
        
        regime = np.zeros_like(self.hyper_choppiness)
        regime[self.hyper_choppiness >= choppy_threshold] = 1  # チョピー
        regime[self.hyper_choppiness <= trending_threshold] = -1  # トレンド
        # 0はそのまま中立
        
        return regime


# テスト用の関数
def test_hyper_choppiness():
    """基本的なテスト"""
    # サンプルデータの生成
    np.random.seed(42)
    n = 200
    
    # トレンドデータとレンジデータを組み合わせ
    trend_data = np.cumsum(np.random.randn(n//2) * 0.5) + 100
    range_data = np.random.randn(n//2) * 2 + trend_data[-1]
    prices = np.concatenate([trend_data, range_data])
    
    # OHLCVデータの作成
    data = np.zeros((n, 5))
    for i in range(n):
        price = prices[i]
        high = price + abs(np.random.randn() * 0.5)
        low = price - abs(np.random.randn() * 0.5)
        open_price = price + np.random.randn() * 0.2
        close = price + np.random.randn() * 0.2
        volume = 1000 + abs(np.random.randn() * 200)
        
        data[i] = [open_price, high, low, close, volume]
    
    # Hyper Choppinessの計算
    hc = HyperChoppiness()
    results = hc.calculate(data)
    
    print("=== Hyper Choppiness Test Results ===")
    print(f"Data points: {len(data)}")
    print(f"Hyper Choppiness range: {np.nanmin(results['hyper_choppiness']):.2f} - {np.nanmax(results['hyper_choppiness']):.2f}")
    print(f"Average cycle period: {np.nanmean(results['cycle_periods']):.1f}")
    print(f"Choppy periods: {np.sum(hc.is_choppy())}")
    print(f"Trending periods: {np.sum(hc.is_trending())}")


if __name__ == "__main__":
    test_hyper_choppiness()