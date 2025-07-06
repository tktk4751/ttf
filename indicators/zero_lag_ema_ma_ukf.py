#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Zero Lag EMA with Market-Adaptive UKF v1.0** 🎯

HLC3価格を市場適応無香料カルマンフィルター（MA-UKF）でフィルタリングし、
そのフィルタリング済み価格を使用してゼロラグEMAを計算するインジケーター。

特徴:
- HLC3価格の取得と前処理
- 市場適応無香料カルマンフィルター（MA-UKF）による高精度ノイズ除去
- ゼロラグEMAによる遅延のないトレンド追跡
- 動的トレンド判定（up/down/range）
- Numba最適化による高速計算
- 包括的なエラーハンドリング
"""

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .kalman_filter_unified import KalmanFilterUnified
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from kalman_filter_unified import KalmanFilterUnified


class ZeroLagEMAResult(NamedTuple):
    """Zero Lag EMA計算結果"""
    values: np.ndarray                    # ゼロラグEMA値
    ema_values: np.ndarray               # 通常のEMA値
    filtered_source: np.ndarray          # MA-UKFフィルタリング済みソース価格
    raw_source: np.ndarray               # 生のソース価格（HLC3）
    trend_signals: np.ndarray            # 1=up, -1=down, 0=range
    current_trend: str                   # 'up', 'down', 'range'
    current_trend_value: int             # 1, -1, 0
    market_regimes: Optional[np.ndarray] # 市場レジーム状態（MA-UKF用）
    confidence_scores: Optional[np.ndarray] # MA-UKF信頼度スコア


@njit(fastmath=True, cache=True)
def calculate_zero_lag_ema_numba(
    prices: np.ndarray, 
    period: int,
    lag_adjustment: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ゼロラグEMAを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列（フィルタリング済み）
        period: EMA期間
        lag_adjustment: 遅延調整係数（デフォルト: 2.0）
    
    Returns:
        Tuple[zero_lag_ema, regular_ema]: ゼロラグEMAと通常のEMA
    """
    n = len(prices)
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    
    zero_lag_ema = np.full(n, np.nan)
    regular_ema = np.full(n, np.nan)
    
    if period <= 0:
        return zero_lag_ema, regular_ema
    
    # EMAの平滑化係数
    alpha = 2.0 / (period + 1.0)
    
    # 初期値設定
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(prices[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return zero_lag_ema, regular_ema
    
    # 初期値
    regular_ema[first_valid_idx] = prices[first_valid_idx]
    zero_lag_ema[first_valid_idx] = prices[first_valid_idx]
    
    # EMAとゼロラグEMAの計算
    for i in range(first_valid_idx + 1, n):
        if np.isnan(prices[i]):
            # NaN値の場合は前の値を維持
            regular_ema[i] = regular_ema[i-1]
            zero_lag_ema[i] = zero_lag_ema[i-1]
            continue
        
        # 通常のEMA計算
        regular_ema[i] = alpha * prices[i] + (1 - alpha) * regular_ema[i-1]
        
        # ゼロラグEMA計算
        # ゼロラグEMA = EMA + lag_adjustment * (EMA - EMA[-1])
        # これにより遅延を補正
        lag_correction = lag_adjustment * (regular_ema[i] - regular_ema[i-1])
        candidate_value = regular_ema[i] + lag_correction
        
        # 厳格な数値安定性チェック（発散防止強化）
        # 1. 候補値が有限でない場合
        if not np.isfinite(candidate_value):
            zero_lag_ema[i] = regular_ema[i]
            continue
        
        # 2. 異常に大きな値（元の価格の1.5倍を超える）
        max_reasonable_value = abs(prices[i]) * 1.5 + 1e-6
        if abs(candidate_value) > max_reasonable_value:
            zero_lag_ema[i] = regular_ema[i]
            continue
        
        # 3. 異常な変化量（前の値からの変化が価格変化の3倍を超える）
        if i > first_valid_idx:
            price_change = abs(prices[i] - prices[i-1])
            ema_change = abs(candidate_value - zero_lag_ema[i-1])
            max_reasonable_change = max(price_change * 3.0, abs(prices[i]) * 0.1)
            
            if ema_change > max_reasonable_change:
                zero_lag_ema[i] = regular_ema[i]
                continue
        
        # 4. 候補値が合理的な範囲内の場合のみ採用
        zero_lag_ema[i] = candidate_value
    
    return zero_lag_ema, regular_ema


@njit(fastmath=True, cache=True)
def calculate_trend_signals_zero_lag(
    values: np.ndarray, 
    slope_period: int = 1,
    range_threshold: float = 0.003
) -> np.ndarray:
    """
    ゼロラグEMAのトレンド信号を計算（Numba最適化版）
    
    Args:
        values: ゼロラグEMA値配列
        slope_period: スロープ計算期間
        range_threshold: レンジ判定閾値
    
    Returns:
        trend_signals: 1=up, -1=down, 0=range
    """
    n = len(values)
    trend_signals = np.zeros(n, dtype=np.int8)
    
    if n < slope_period + 1:
        return trend_signals
    
    # 統計的閾値計算用のウィンドウ
    stats_window = min(21, n // 2)
    
    for i in range(slope_period, n):
        if np.isnan(values[i]) or np.isnan(values[i - slope_period]):
            trend_signals[i] = 0  # NaNの場合はレンジ
            continue
        
        current = values[i]
        previous = values[i - slope_period]
        
        # 基本変化量
        change = current - previous
        
        # 相対変化率（ゼロ除算保護強化）
        base_value = max(abs(current), abs(previous), 1e-8)
        if base_value <= 1e-10:
            relative_change = 0.0
        else:
            relative_change = abs(change) / base_value
        
        # 動的閾値計算（過去の変動性に基づく）
        if i >= stats_window + slope_period:
            start_idx = i - stats_window
            historical_changes = np.zeros(stats_window)
            
            count = 0
            for j in range(start_idx, i):
                if (not np.isnan(values[j]) and 
                    not np.isnan(values[j - slope_period])):
                    hist_curr = values[j]
                    hist_prev = values[j - slope_period]
                    hist_base = max(abs(hist_curr), abs(hist_prev), 1e-8)
                    if hist_base <= 1e-10:
                        hist_change = 0.0
                    else:
                        hist_change = abs(hist_curr - hist_prev) / hist_base
                    historical_changes[count] = hist_change
                    count += 1
            
            if count > 0:
                # 有効なデータのみを使用して統計を計算
                valid_changes = historical_changes[:count]
                std_threshold = np.std(valid_changes) * 0.7
                effective_threshold = max(range_threshold, std_threshold)
            else:
                effective_threshold = range_threshold
        else:
            effective_threshold = range_threshold
        
        # トレンド判定
        if relative_change < effective_threshold:
            trend_signals[i] = 0  # レンジ
        elif change > 0:
            trend_signals[i] = 1  # 上昇
        else:
            trend_signals[i] = -1  # 下降
    
    return trend_signals


@njit(fastmath=True, cache=True)
def get_current_trend_state(trend_signals: np.ndarray) -> Tuple[int, int]:
    """
    現在のトレンド状態を取得（Numba最適化版）
    
    Args:
        trend_signals: トレンド信号配列
    
    Returns:
        Tuple[trend_index, trend_value]: (インデックス, 値)
    """
    if len(trend_signals) == 0:
        return 0, 0  # レンジ
    
    latest_trend = trend_signals[-1]
    
    if latest_trend == 1:    # 上昇
        return 1, 1
    elif latest_trend == -1: # 下降
        return 2, -1
    else:                    # レンジ
        return 0, 0


class ZeroLagEMAWithMAUKF(Indicator):
    """
    🎯 Market-Adaptive UKF Zero Lag EMA インジケーター
    
    HLC3価格を市場適応無香料カルマンフィルター（MA-UKF）でフィルタリングし、
    そのフィルタリング済み価格を使用してゼロラグEMAを計算します。
    
    特徴:
    - HLC3価格の自動取得
    - MA-UKFによる高精度ノイズ除去
    - ゼロラグEMAによる遅延のないトレンド追跡
    - 動的トレンド判定（レンジ相場検出含む）
    - 市場レジーム情報の提供
    - Numba最適化による高速計算
    """
    
    def __init__(
        self,
        ema_period: int = 14,
        lag_adjustment: float = 1.0,
        slope_period: int = 1,
        range_threshold: float = 0.003,
        # MA-UKFパラメータ
        ukf_alpha: float = 0.001,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        ukf_base_process_noise: float = 0.001,
        ukf_base_measurement_noise: float = 0.01,
        ukf_volatility_window: int = 10
    ):
        """
        コンストラクタ
        
        Args:
            ema_period: EMA期間
            lag_adjustment: ゼロラグ調整係数
            slope_period: トレンド判定期間
            range_threshold: レンジ判定基本閾値
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            ukf_base_process_noise: UKF基本プロセスノイズ
            ukf_base_measurement_noise: UKF基本測定ノイズ
            ukf_volatility_window: UKFボラティリティウィンドウ
        """
        name = (f"ZeroLagEMA_MAUKF(period={ema_period}, "
                f"lag_adj={lag_adjustment}, slope={slope_period}, "
                f"range_th={range_threshold:.4f})")
        super().__init__(name)
        
        # パラメータ保存（安全な範囲に制限）
        self.ema_period = max(1, ema_period)
        # lag_adjustmentを安全な範囲に制限（過度な発散を防ぐ）
        self.lag_adjustment = max(0.1, min(3.0, lag_adjustment))
        self.slope_period = max(1, slope_period)
        self.range_threshold = max(0.0001, range_threshold)
        
        # MA-UKFフィルター初期化
        self.ma_ukf = KalmanFilterUnified(
            filter_type='market_adaptive_unscented',
            src_type='hlc3',
            base_process_noise=ukf_base_process_noise,
            base_measurement_noise=ukf_base_measurement_noise,
            volatility_window=ukf_volatility_window,
            ukf_alpha=ukf_alpha,
            ukf_beta=ukf_beta,
            ukf_kappa=ukf_kappa
        )
        
        # キャッシュ
        self._cache = {}
        self._result: Optional[ZeroLagEMAResult] = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算"""
        try:
            if isinstance(data, pd.DataFrame):
                # HLC3に必要なカラムのみを考慮
                required_cols = ['high', 'low', 'close']
                available_cols = [col for col in required_cols if col in data.columns]
                if available_cols:
                    data_values = data[available_cols].values
                    data_hash = hash(data_values.tobytes())
                else:
                    data_hash = hash(str(data.shape))
            else:
                data_hash = hash(data.tobytes())
            
            param_str = (f"ema={self.ema_period}_lag={self.lag_adjustment}_"
                        f"slope={self.slope_period}_thresh={self.range_threshold}")
            return f"{data_hash}_{param_str}"
        except Exception:
            return str(hash(str(data)))
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZeroLagEMAResult:
        """
        Zero Lag EMA with MA-UKFを計算
        
        Args:
            data: 価格データ（OHLC形式）
        
        Returns:
            ZeroLagEMAResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result
            
            # データ検証
            if isinstance(data, np.ndarray) and data.ndim == 1:
                raise ValueError("1次元配列は使用できません。OHLC形式のデータが必要です。")
            
            # HLC3価格を取得
            try:
                raw_hlc3 = PriceSource.calculate_source(data, 'hlc3')
            except Exception as e:
                self.logger.error(f"HLC3価格の取得に失敗: {e}")
                return self._create_empty_result(len(data) if hasattr(data, '__len__') else 0)
            
            if len(raw_hlc3) == 0:
                self.logger.warning("HLC3価格データが空です")
                return self._create_empty_result(0)
            
            # MA-UKFでHLC3をフィルタリング
            try:
                ukf_result = self.ma_ukf.calculate(data)
                filtered_hlc3 = ukf_result.filtered_values
                
                # MA-UKF固有データの取得
                market_regimes = None
                confidence_scores = None
                
                if hasattr(self.ma_ukf, 'get_market_regimes'):
                    market_regimes = self.ma_ukf.get_market_regimes()
                if hasattr(self.ma_ukf, 'get_confidence_scores'):
                    confidence_scores = self.ma_ukf.get_confidence_scores()
                
            except Exception as e:
                self.logger.error(f"MA-UKFフィルタリングに失敗: {e}")
                # フォールバック：フィルタリングなしで継続
                filtered_hlc3 = raw_hlc3.copy()
                market_regimes = None
                confidence_scores = None
            
            # データ長チェック
            if len(filtered_hlc3) < self.ema_period:
                self.logger.warning(f"データ長({len(filtered_hlc3)})が"
                                  f"EMA期間({self.ema_period})より短いです")
            
            # ゼロラグEMAの計算
            zero_lag_ema, regular_ema = calculate_zero_lag_ema_numba(
                filtered_hlc3, self.ema_period, self.lag_adjustment
            )
            
            # トレンド信号の計算
            trend_signals = calculate_trend_signals_zero_lag(
                zero_lag_ema, self.slope_period, self.range_threshold
            )
            
            # 現在のトレンド状態
            trend_index, trend_value = get_current_trend_state(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]
            
            # 結果の作成
            result = ZeroLagEMAResult(
                values=zero_lag_ema,
                ema_values=regular_ema,
                filtered_source=filtered_hlc3,
                raw_source=raw_hlc3,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value,
                market_regimes=market_regimes,
                confidence_scores=confidence_scores
            )
            
            # キャッシュ保存
            self._result = result
            self._cache[data_hash] = result
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算エラー: {error_msg}\n{stack_trace}")
            
            data_len = len(data) if hasattr(data, '__len__') else 0
            return self._create_empty_result(data_len)
    
    def _create_empty_result(self, length: int) -> ZeroLagEMAResult:
        """空の結果を作成"""
        return ZeroLagEMAResult(
            values=np.full(length, np.nan, dtype=np.float64),
            ema_values=np.full(length, np.nan, dtype=np.float64),
            filtered_source=np.full(length, np.nan, dtype=np.float64),
            raw_source=np.full(length, np.nan, dtype=np.float64),
            trend_signals=np.zeros(length, dtype=np.int8),
            current_trend='range',
            current_trend_value=0,
            market_regimes=None,
            confidence_scores=None
        )
    
    # === 後方互換性とデータアクセスメソッド ===
    
    def get_values(self) -> Optional[np.ndarray]:
        """ゼロラグEMA値を取得（後方互換性用）"""
        if self._result is not None:
            return self._result.values.copy()
        return None
    
    def get_ema_values(self) -> Optional[np.ndarray]:
        """通常のEMA値を取得"""
        if self._result is not None:
            return self._result.ema_values.copy()
        return None
    
    def get_filtered_source(self) -> Optional[np.ndarray]:
        """MA-UKFフィルタリング済みソース価格を取得"""
        if self._result is not None:
            return self._result.filtered_source.copy()
        return None
    
    def get_raw_source(self) -> Optional[np.ndarray]:
        """生のソース価格（HLC3）を取得"""
        if self._result is not None:
            return self._result.raw_source.copy()
        return None
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None
    
    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'
    
    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得"""
        if self._result is not None:
            return self._result.current_trend_value
        return 0
    
    def get_market_regimes(self) -> Optional[np.ndarray]:
        """市場レジーム状態を取得（MA-UKF由来）"""
        if self._result is not None and self._result.market_regimes is not None:
            return self._result.market_regimes.copy()
        return None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """MA-UKF信頼度スコアを取得"""
        if self._result is not None and self._result.confidence_scores is not None:
            return self._result.confidence_scores.copy()
        return None
    
    def get_filter_performance(self) -> dict:
        """フィルター性能の統計を取得"""
        if self._result is None:
            return {}
        
        stats = {
            'data_points': len(self._result.values),
            'valid_points': np.sum(~np.isnan(self._result.values)),
            'trend_up_ratio': np.mean(self._result.trend_signals == 1),
            'trend_down_ratio': np.mean(self._result.trend_signals == -1),
            'range_ratio': np.mean(self._result.trend_signals == 0),
            'current_trend': self._result.current_trend
        }
        
        # 価格統計
        valid_zero_lag = self._result.values[~np.isnan(self._result.values)]
        valid_raw = self._result.raw_source[~np.isnan(self._result.raw_source)]
        
        if len(valid_zero_lag) > 0:
            stats.update({
                'zero_lag_ema_mean': np.mean(valid_zero_lag),
                'zero_lag_ema_std': np.std(valid_zero_lag),
                'zero_lag_ema_range': np.max(valid_zero_lag) - np.min(valid_zero_lag)
            })
        
        if len(valid_raw) > 0:
            stats.update({
                'raw_price_mean': np.mean(valid_raw),
                'raw_price_std': np.std(valid_raw)
            })
        
        # フィルタリング効果
        if len(valid_zero_lag) > 0 and len(valid_raw) > 0:
            # 最小長に合わせる
            min_len = min(len(valid_zero_lag), len(valid_raw))
            if min_len > 1:
                filtered_volatility = np.std(np.diff(valid_zero_lag[:min_len]))
                raw_volatility = np.std(np.diff(valid_raw[:min_len]))
                if raw_volatility > 0:
                    stats['noise_reduction_ratio'] = 1.0 - (filtered_volatility / raw_volatility)
        
        # MA-UKF固有統計
        if self._result.market_regimes is not None:
            valid_regimes = self._result.market_regimes[~np.isnan(self._result.market_regimes)]
            if len(valid_regimes) > 0:
                stats.update({
                    'avg_market_regime': np.mean(valid_regimes),
                    'trend_market_ratio': np.mean(valid_regimes > 0.5),
                    'range_market_ratio': np.mean(np.abs(valid_regimes) < 0.3)
                })
        
        if self._result.confidence_scores is not None:
            valid_conf = self._result.confidence_scores[~np.isnan(self._result.confidence_scores)]
            if len(valid_conf) > 0:
                stats['avg_ma_ukf_confidence'] = np.mean(valid_conf)
        
        return stats
    
    def reset(self) -> None:
        """インジケーターの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        if hasattr(self.ma_ukf, 'reset'):
            self.ma_ukf.reset()


# === デモンストレーション機能 ===

def demo_zero_lag_ema_ma_ukf():
    """
    Zero Lag EMA with MA-UKFのデモ
    
    基本的な使用方法と性能を示します。
    """
    print("🎯 Zero Lag EMA with Market-Adaptive UKF デモ")
    print("=" * 60)
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 150
    t = np.arange(n)
    
    # 複雑な価格データをシミュレート
    base_price = 100
    trend = 0.05 * t
    cyclical = 5 * np.sin(t * 0.1) + 2 * np.sin(t * 0.05)
    noise = np.random.normal(0, 1, n)
    
    # トレンド変化点を追加
    trend_changes = [40, 80, 120]
    for change_point in trend_changes:
        if change_point < n:
            trend[change_point:] += np.random.choice([-5, 5])
    
    close_prices = base_price + trend + cyclical + noise
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, n))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, n))
    
    # DataFrame作成
    data = pd.DataFrame({
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'open': close_prices + np.random.normal(0, 0.2, n)
    })
    
    print(f"サンプルデータ生成: {len(data)}期間")
    print(f"価格範囲: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    # Zero Lag EMA with MA-UKFを計算
    zero_lag_ema = ZeroLagEMAWithMAUKF(
        ema_period=14,
        lag_adjustment=2.0,
        slope_period=1,
        range_threshold=0.003
    )
    
    print(f"\n{zero_lag_ema.name} 計算中...")
    
    try:
        result = zero_lag_ema.calculate(data)
        
        # 基本統計
        valid_values = result.values[~np.isnan(result.values)]
        print(f"有効なデータポイント: {len(valid_values)}/{len(result.values)}")
        
        if len(valid_values) > 0:
            print(f"Zero Lag EMA範囲: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"現在のトレンド: {result.current_trend}")
            
            # トレンド統計
            trend_stats = {
                'up': np.mean(result.trend_signals == 1),
                'down': np.mean(result.trend_signals == -1), 
                'range': np.mean(result.trend_signals == 0)
            }
            print(f"トレンド分布: up={trend_stats['up']:.1%}, "
                  f"down={trend_stats['down']:.1%}, range={trend_stats['range']:.1%}")
            
            # MA-UKF統計
            if result.market_regimes is not None:
                valid_regimes = result.market_regimes[~np.isnan(result.market_regimes)]
                if len(valid_regimes) > 0:
                    print(f"平均市場レジーム: {np.mean(valid_regimes):.3f}")
                    print(f"トレンド市場比率: {np.mean(valid_regimes > 0.5):.1%}")
            
            if result.confidence_scores is not None:
                valid_conf = result.confidence_scores[~np.isnan(result.confidence_scores)]
                if len(valid_conf) > 0:
                    print(f"平均MA-UKF信頼度: {np.mean(valid_conf):.3f}")
            
            # フィルタリング効果の評価
            valid_raw = result.raw_source[~np.isnan(result.raw_source)]
            valid_filtered = result.filtered_source[~np.isnan(result.filtered_source)]
            
            if len(valid_raw) > 1 and len(valid_filtered) > 1:
                min_len = min(len(valid_raw), len(valid_filtered))
                raw_volatility = np.std(np.diff(valid_raw[:min_len]))
                filtered_volatility = np.std(np.diff(valid_filtered[:min_len]))
                
                if raw_volatility > 0:
                    noise_reduction = (1.0 - filtered_volatility / raw_volatility) * 100
                    print(f"ノイズ除去効果: {noise_reduction:.1f}%")
        
        # 性能統計
        perf_stats = zero_lag_ema.get_filter_performance()
        print(f"\n詳細性能統計:")
        for key, value in perf_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Zero Lag EMA with MA-UKF デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    demo_zero_lag_ema_ma_ukf() 