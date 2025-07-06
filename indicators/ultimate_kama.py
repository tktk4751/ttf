#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import njit, jit

from .indicator import Indicator
from .price_source import PriceSource
from .ultimate_smoother import UltimateSmoother
from .ultimate_efficiency_ratio import UltimateEfficiencyRatio

class UltimateKAMAResult(NamedTuple):
    """UltimateKAMA計算結果"""
    values: np.ndarray              # 最終フィルター済み価格
    raw_values: np.ndarray          # 元の価格
    ukf_values: np.ndarray          # UKF_HLC3フィルター後
    ultimate_smooth_values: np.ndarray # アルティメットスムーザー後
    kama_values: np.ndarray         # KAMA後
    zero_lag_values: np.ndarray     # ゼロラグEMA後
    er: np.ndarray                 # アルティメットER
    trend_signals: np.ndarray      # 1=up, -1=down, 0=range
    current_trend: str             # 'up', 'down', 'range'
    current_trend_value: int       # 1, -1, 0

@jit(nopython=True, cache=True)
def kama_with_er_numba(prices: np.ndarray, er: np.ndarray, fast_period: int, slow_period: int) -> np.ndarray:
    """
    アルティメットERを使ったKAMA計算（Numba最適化）
    prices: 入力価格
    er: アルティメット効率比（0-1）
    fast_period, slow_period: KAMAのパラメータ
    """
    n = len(prices)
    kama = np.zeros(n)
    if n == 0:
        return kama
    # 平滑化定数
    fast_alpha = 2.0 / (fast_period + 1.0)
    slow_alpha = 2.0 / (slow_period + 1.0)
    kama[0] = prices[0]
    for i in range(1, n):
        sc = (er[i] * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    return kama

@jit(nopython=True, cache=True)
def zero_lag_ema_numba(prices: np.ndarray, period: int = 21) -> np.ndarray:
    n = len(prices)
    zero_lag = np.zeros(n)
    if n < 2:
        return prices.copy()
    alpha = 2.0 / (period + 1.0)
    zero_lag[0] = prices[0]
    for i in range(1, n):
        ema = alpha * prices[i] + (1 - alpha) * zero_lag[i-1]
        if i >= 2:
            momentum = prices[i] - prices[i-1]
            lag_correction = alpha * momentum
            zero_lag[i] = ema + lag_correction
        else:
            zero_lag[i] = ema
    return zero_lag

@jit(nopython=True, cache=True)
def real_time_trend_detector_numba(prices: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(prices)
    trend_signals = np.zeros(n)
    if n < 3:
        return trend_signals
    smoothed = np.zeros(n)
    smoothed[0] = prices[0]
    alpha = 0.3
    for i in range(1, n):
        if np.isnan(prices[i]):
            smoothed[i] = smoothed[i-1]
            continue
        smoothed[i] = alpha * prices[i] + (1 - alpha) * smoothed[i-1]
    for i in range(3, n):
        if np.isnan(prices[i]):
            trend_signals[i] = 0.0
            continue
        trend_1 = smoothed[i] - smoothed[i-1]
        trend_2 = (smoothed[i] - smoothed[i-2]) / 2.0
        trend_3 = (smoothed[i] - smoothed[i-3]) / 3.0
        combined_trend = trend_1 * 0.4 + trend_2 * 0.35 + trend_3 * 0.25
        noise_threshold = 0.0
        if i >= 5:
            recent_noise = abs(prices[i-1] - prices[i-2]) + abs(prices[i-2] - prices[i-3])
            avg_noise = recent_noise / 2.0
            noise_threshold = avg_noise * 0.5
        consistency = 0.0
        if abs(trend_1) > 0 and abs(trend_2) > 0 and abs(trend_3) > 0:
            direction_1 = 1 if trend_1 > 0 else -1
            direction_2 = 1 if trend_2 > 0 else -1
            direction_3 = 1 if trend_3 > 0 else -1
            main_direction = 1 if combined_trend > 0 else -1
            matches = 0
            if direction_1 == main_direction: matches += 1
            if direction_2 == main_direction: matches += 1
            if direction_3 == main_direction: matches += 1
            consistency = matches / 3.0
        trend_strength = abs(combined_trend)
        if trend_strength <= noise_threshold:
            trend_signals[i] = 0.0
            continue
        if consistency < 0.6:
            trend_strength *= 0.5
        long_term_boost = 1.0
        if i >= min(window, 8):
            long_trend = (smoothed[i] - smoothed[i-min(window, 8)]) / min(window, 8)
            if (combined_trend > 0 and long_trend > 0) or (combined_trend < 0 and long_trend < 0):
                long_term_boost = 1.3
        final_strength = trend_strength * long_term_boost
        min_strength = max(noise_threshold * 2.0, abs(combined_trend) * 0.1)
        if final_strength > min_strength:
            trend_signals[i] = final_strength * (1 if combined_trend > 0 else -1)
        else:
            trend_signals[i] = 0.0
    return trend_signals

@jit(nopython=True, cache=True)
def calculate_trend_signals_with_range_numba(values: np.ndarray, slope_index: int, range_threshold: float = 0.005) -> np.ndarray:
    """
    🚀 **超高精度AI風トレンド判定アルゴリズム V3.0** 🚀
    
    最新の金融工学技術を統合した次世代判定システム:
    - **適応的指数重み付け統計**: 最新データ重視の動的閾値
    - **多時間軸モメンタム分析**: 1期・2期・3期・5期の複合解析
    - **AI風動的信頼度スコア**: 4指標の重み付き総合評価
    - **高精度ボラティリティ分析**: 市場状況の自動判定・適応
    - **予測的判定システム**: 先読み機能による早期検出
    - **緊急事態検出**: 極端変化への瞬時対応
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    # 超低遅延統計ウィンドウ（最小限に短縮）
    stats_window = max(13, slope_index * 2)  # 大幅短縮
    confirmation_window = 5  # 固定2期間で即応性重視
    
    for i in range(stats_window, length):
        if np.isnan(values[i]):
            trend_signals[i] = 0
            continue
        
        current = values[i]
        previous = values[i - slope_index]
        
        if np.isnan(previous):
            trend_signals[i] = 0
            continue
        
        # 基本的な変化量
        change = current - previous
        base_value = max(abs(current), abs(previous), 1e-10)
        relative_change = change / base_value
        abs_relative_change = abs(relative_change)
        
        # 🔥 1. 適応的指数重み付け統計（最新データ重視）
        start_idx = max(slope_index, i - stats_window + 1)
        
        # 指数重み付けによる高精度閾値計算
        weighted_changes = 0.0
        weighted_sum = 0.0
        weighted_variance = 0.0
        
        for j in range(start_idx + slope_index, i):
            if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                hist_current = values[j]
                hist_previous = values[j - slope_index]
                hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                hist_change = abs(hist_current - hist_previous) / hist_base
                
                # 指数重み（最新データほど重要）
                age = i - j
                weight = np.exp(-age * 0.15)  # 指数減衰
                
                weighted_changes += hist_change * weight
                weighted_sum += weight
        
        # 動的閾値の計算
        if weighted_sum > 0:
            avg_weighted_change = weighted_changes / weighted_sum
            dynamic_threshold = max(range_threshold, avg_weighted_change * 0.8)
        else:
            dynamic_threshold = range_threshold
        
        # 🔥 2. 多時間軸モメンタム分析
        momentum_1 = 0.0
        momentum_2 = 0.0
        momentum_3 = 0.0
        momentum_5 = 0.0
        
        if i >= slope_index + 1:
            momentum_1 = (values[i] - values[i-1]) / max(abs(values[i-1]), 1e-10)
        if i >= slope_index + 2:
            momentum_2 = (values[i] - values[i-2]) / max(abs(values[i-2]), 1e-10) / 2.0
        if i >= slope_index + 3:
            momentum_3 = (values[i] - values[i-3]) / max(abs(values[i-3]), 1e-10) / 3.0
        if i >= slope_index + 5:
            momentum_5 = (values[i] - values[i-5]) / max(abs(values[i-5]), 1e-10) / 5.0
        
        # 重み付きモメンタム合成
        composite_momentum = (momentum_1 * 0.4 + momentum_2 * 0.3 + 
                             momentum_3 * 0.2 + momentum_5 * 0.1)
        
        # 🔥 3. AI風動的信頼度スコア
        confidence_score = 0.0
        
        # 方向性の一貫性チェック
        direction_consistency = 0
        if abs(momentum_1) > 0 and abs(momentum_2) > 0:
            if (momentum_1 > 0 and momentum_2 > 0) or (momentum_1 < 0 and momentum_2 < 0):
                direction_consistency += 1
        if abs(momentum_2) > 0 and abs(momentum_3) > 0:
            if (momentum_2 > 0 and momentum_3 > 0) or (momentum_2 < 0 and momentum_3 < 0):
                direction_consistency += 1
        if abs(momentum_3) > 0 and abs(momentum_5) > 0:
            if (momentum_3 > 0 and momentum_5 > 0) or (momentum_3 < 0 and momentum_5 < 0):
                direction_consistency += 1
        
        confidence_score += direction_consistency * 0.25
        
        # 変化量の安定性チェック
        if abs_relative_change > dynamic_threshold:
            confidence_score += 0.25
        
        # モメンタムの強度チェック
        if abs(composite_momentum) > dynamic_threshold * 0.5:
            confidence_score += 0.25
        
        # 統計的異常値チェック
        if abs_relative_change > dynamic_threshold * 1.5:
            confidence_score += 0.25
        
        # 🔥 4. 高精度ボラティリティ分析
        volatility_factor = 1.0
        if i >= 10:
            recent_volatility = 0.0
            for k in range(1, 10):
                if not np.isnan(values[i-k]) and not np.isnan(values[i-k-1]):
                    vol_change = abs(values[i-k] - values[i-k-1]) / max(abs(values[i-k-1]), 1e-10)
                    recent_volatility += vol_change
            recent_volatility /= 9.0
            
            if recent_volatility > dynamic_threshold * 2.0:
                volatility_factor = 0.7  # 高ボラティリティ時は判定を厳しく
            elif recent_volatility < dynamic_threshold * 0.5:
                volatility_factor = 1.3  # 低ボラティリティ時は判定を緩く
        
        # 🔥 5. 予測的判定システム
        predictive_boost = 1.0
        if i >= slope_index + 3:
            # 短期トレンドの加速度を計算
            short_accel = momentum_1 - momentum_2
            if abs(short_accel) > dynamic_threshold * 0.3:
                if (composite_momentum > 0 and short_accel > 0) or (composite_momentum < 0 and short_accel < 0):
                    predictive_boost = 1.2  # 加速トレンド
        
        # 🔥 6. 緊急事態検出
        extreme_threshold = dynamic_threshold * 3.0
        if abs_relative_change > extreme_threshold:
            # 極端な変化の場合は即座に判定
            trend_signals[i] = 1 if relative_change > 0 else -1
            continue
        
        # 🔥 最終判定ロジック
        final_threshold = dynamic_threshold * volatility_factor
        adjusted_momentum = composite_momentum * predictive_boost
        
        if confidence_score >= 0.5 and abs(adjusted_momentum) > final_threshold:
            if adjusted_momentum > 0:
                trend_signals[i] = 1  # up
            else:
                trend_signals[i] = -1  # down
        else:
            trend_signals[i] = 0  # range
    
    return trend_signals

@jit(nopython=True, cache=True)
def calculate_current_trend_with_range_numba(trend_signals: np.ndarray) -> tuple:
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

class UltimateKAMA(Indicator):
    """
    🚀 Ultimate KAMA - アルティメット効率比とKAMAを融合した次世代適応型移動平均
    
    - プライスソース: UKF_HLC3
    - アルティメットスムーサーで平滑化
    - アルティメットERで効率比を算出
    - KAMAロジックで適応平滑化
    - ゼロラグEMAで遅延補正
    - UltimateMAと同じリアルタイムトレンド検出
    """
    def __init__(self,
                 kama_fast: int = 2,
                 kama_slow: int = 30,
                 zero_lag_period: int = 21,
                 smoother_period: float = 13.0,
                 src_type: str = 'ukf_hlc3',
                 trend_window: int = 21,
                 slope_index: int = 1,
                 range_threshold: float = 0.005):
        """
        コンストラクタ
        Args:
            kama_fast: KAMAのfast期間
            kama_slow: KAMAのslow期間
            zero_lag_period: ゼロラグEMA期間
            smoother_period: アルティメットスムーサー期間
            src_type: プライスソース
            trend_window: トレンド検出ウィンドウ
            slope_index: トレンド判定期間
            range_threshold: range判定の閾値
        """
        name = f"UltimateKAMA(kama_fast={kama_fast},kama_slow={kama_slow},zl={zero_lag_period},us={smoother_period},{src_type},trend={trend_window},slope={slope_index},range_th={range_threshold})"
        super().__init__(name)
        self.kama_fast = kama_fast
        self.kama_slow = kama_slow
        self.zero_lag_period = zero_lag_period
        self.smoother_period = smoother_period
        self.src_type = src_type
        self.trend_window = trend_window
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        self.smoother = UltimateSmoother(period=smoother_period, src_type=src_type)
        self.er_indicator = UltimateEfficiencyRatio(period=14, smoother_period=smoother_period, src_type=src_type)
        self._result: Optional[UltimateKAMAResult] = None
        self._cache = {}

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
        elif self.src_type == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
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

        param_str = (f"kama_fast={self.kama_fast}_kama_slow={self.kama_slow}"
                    f"_zl={self.zero_lag_period}_us={self.smoother_period}"
                    f"_src={self.src_type}_trend={self.trend_window}"
                    f"_slope={self.slope_index}_range_th={self.range_threshold}")
        return f"{data_hash_val}_{param_str}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        🚀 Ultimate KAMA を計算する（アルティメットER + KAMA + ゼロラグEMA）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            np.ndarray: 最終フィルター済み価格（ゼロラグEMA値）
        """
        try:
            # データチェック - 1次元配列が直接渡された場合は使用できない（UKF_HLC3にはOHLCが必要）
            if isinstance(data, np.ndarray) and data.ndim == 1:
                raise ValueError("1次元配列は直接使用できません。UKF_HLC3にはOHLCデータが必要です。")
            else:
                # 通常のハッシュチェック
                data_hash = self._get_data_hash(data)
                if data_hash in self._cache and self._result is not None:
                    return self._result.values

                # UKF_HLC3を使用して価格を取得
                ukf_prices = PriceSource.calculate_source(data, 'ukf_hlc3')
                ukf_prices = ukf_prices.astype(np.float64)  # 明示的にfloat64に変換
                data_hash_key = data_hash

            # データ長の検証
            data_length = len(ukf_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_result = UltimateKAMAResult(
                    values=np.array([], dtype=np.float64),
                    raw_values=np.array([], dtype=np.float64),
                    ukf_values=np.array([], dtype=np.float64),
                    ultimate_smooth_values=np.array([], dtype=np.float64),
                    kama_values=np.array([], dtype=np.float64),
                    zero_lag_values=np.array([], dtype=np.float64),
                    er=np.array([], dtype=np.float64),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash_key] = self._result
                self._values = np.array([], dtype=np.float64)
                return np.array([], dtype=np.float64)

            # 🚀 アルティメットKAMA計算処理
            self.logger.info("🚀 Ultimate KAMA - アルティメットER + KAMA + ゼロラグEMA実行中...")
            
            # ①元の価格（比較用）
            src_prices = PriceSource.calculate_source(data, self.src_type)
            src_prices = src_prices.astype(np.float64)
            
            # ②アルティメットスムーザーフィルター
            self.logger.debug("🌊 アルティメットスムーザーフィルター適用中...")
            ultimate_smooth_result = self.smoother.calculate(data)
            ultimate_smoothed = ultimate_smooth_result.values
            
            # ③アルティメットER計算
            self.logger.debug("⚡ アルティメットER計算中...")
            er_result = self.er_indicator.calculate(data)
            er_values = er_result.values
            
            # ④KAMA計算（アルティメットER使用）
            self.logger.debug("📈 KAMA計算中（アルティメットER使用）...")
            kama_values = kama_with_er_numba(ultimate_smoothed, er_values, self.kama_fast, self.kama_slow)
            
            # ⑤ゼロラグEMA
            self.logger.debug("⚡ ゼロラグEMA処理中...")
            zero_lag_values = zero_lag_ema_numba(kama_values, self.zero_lag_period)
            
            # ⑥リアルタイムトレンド検出
            self.logger.debug("🎯 リアルタイムトレンド検出中...")
            realtime_trends = real_time_trend_detector_numba(zero_lag_values, self.trend_window)
            
            # ⑦超高精度AI風トレンド判定
            self.logger.debug("🚀 超高精度AI風トレンド判定実行中...")
            trend_signals = calculate_trend_signals_with_range_numba(zero_lag_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range_numba(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            # 最終的な処理済み価格系列
            final_values = zero_lag_values

            result = UltimateKAMAResult(
                values=final_values,
                raw_values=src_prices,
                ukf_values=ukf_prices,
                ultimate_smooth_values=ultimate_smoothed,
                kama_values=kama_values,
                zero_lag_values=zero_lag_values,
                er=er_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            self._result = result
            self._cache[data_hash_key] = self._result
            
            # 基底クラスの要件に合わせて最終値を設定
            self._values = final_values
            
            self.logger.info(f"✅ Ultimate KAMA 計算完了 - トレンド: {current_trend}")
            return final_values

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None # エラー時は結果をクリア
            error_values = np.full(data_len, np.nan, dtype=np.float64)
            self._values = error_values
            return error_values

    def get_values(self) -> Optional[np.ndarray]:
        """最終フィルター済み値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_raw_values(self) -> Optional[np.ndarray]:
        """元の価格値を取得する"""
        if self._result is not None:
            return self._result.raw_values.copy()
        return None

    def get_ukf_values(self) -> Optional[np.ndarray]:
        """UKF_HLC3フィルター後の値を取得する"""
        if self._result is not None:
            return self._result.ukf_values.copy()
        return None

    def get_ultimate_smooth_values(self) -> Optional[np.ndarray]:
        """アルティメットスムーザーフィルター後の値を取得する"""
        if self._result is not None:
            return self._result.ultimate_smooth_values.copy()
        return None

    def get_kama_values(self) -> Optional[np.ndarray]:
        """KAMA後の値を取得する"""
        if self._result is not None:
            return self._result.kama_values.copy()
        return None

    def get_zero_lag_values(self) -> Optional[np.ndarray]:
        """ゼロラグEMA後の値を取得する"""
        if self._result is not None:
            return self._result.zero_lag_values.copy()
        return None

    def get_er_values(self) -> Optional[np.ndarray]:
        """アルティメットER値を取得する"""
        if self._result is not None:
            return self._result.er.copy()
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

    def get_noise_reduction_stats(self) -> dict:
        """ノイズ除去統計を取得する"""
        if self._result is None:
            return {}
        
        raw_std = np.nanstd(self._result.raw_values)
        final_std = np.nanstd(self._result.values)
        noise_reduction_ratio = (raw_std - final_std) / raw_std if raw_std > 0 else 0.0
        
        return {
            'raw_volatility': raw_std,
            'filtered_volatility': final_std,
            'noise_reduction_ratio': noise_reduction_ratio,
            'noise_reduction_percentage': noise_reduction_ratio * 100,
            'smoothing_effectiveness': min(noise_reduction_ratio * 100, 100.0)
        }

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if hasattr(self, 'smoother'):
            self.smoother.reset()
        if hasattr(self, 'er_indicator'):
            self.er_indicator.reset() 