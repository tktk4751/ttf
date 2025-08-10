#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
    from .ultimate_smoother import UltimateSmoother
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC
    from ultimate_smoother import UltimateSmoother


@dataclass
class SupremeTrendStateResult:
    """Supreme Trend State結果"""
    trend_state: np.ndarray          # 1=トレンド, 0=レンジ (バイナリ出力)
    confidence: np.ndarray           # 信頼度 (0-1)
    entropy_score: np.ndarray        # 情報エントロピー
    fractal_dimension: np.ndarray    # フラクタル次元
    spectral_power: np.ndarray       # スペクトル密度
    neural_adaptation: np.ndarray    # 神経適応学習係数
    ml_prediction: np.ndarray        # 機械学習予測値
    efficiency_ratio: np.ndarray     # 効率比
    volatility_regime: np.ndarray    # ボラティリティ体制
    cycle_strength: np.ndarray       # サイクル強度
    composite_score: np.ndarray      # 複合スコア (0-1)


@njit(fastmath=True, cache=True)
def calculate_information_entropy(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    情報エントロピーを計算（価格変動の不確実性を測定）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        エントロピー値配列（高いほど不確実性が高い＝レンジ）
    """
    n = len(prices)
    entropy = np.zeros(n)
    
    for i in range(window, n):
        # 価格変化率を計算
        returns = np.zeros(window-1)
        for j in range(window-1):
            if prices[i-window+j] != 0:
                returns[j] = (prices[i-window+j+1] - prices[i-window+j]) / prices[i-window+j]
        
        # 変化率を離散化（5段階）
        if len(returns) > 0:
            sorted_returns = np.sort(returns)
            q1 = sorted_returns[len(sorted_returns)//4]
            q2 = sorted_returns[len(sorted_returns)//2]  
            q3 = sorted_returns[3*len(sorted_returns)//4]
            
            # 5段階に分類
            bins = np.array([0, 0, 0, 0, 0])
            for ret in returns:
                if ret <= q1:
                    bins[0] += 1
                elif ret <= q2:
                    bins[1] += 1
                elif ret <= q3:
                    bins[2] += 1
                else:
                    bins[3] += 1
            
            # エントロピー計算
            total = np.sum(bins)
            if total > 0:
                entropy_val = 0.0
                for count in bins:
                    if count > 0:
                        p = count / total
                        entropy_val -= p * np.log2(p)
                entropy[i] = entropy_val
    
    return entropy


@njit(fastmath=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    フラクタル次元を計算（価格パターンの複雑さを測定）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        フラクタル次元配列（1.5に近いほどトレンド、2.0に近いほどレンジ）
    """
    n = len(prices)
    fractal_dim = np.zeros(n)
    
    for i in range(window, n):
        # Higuchi's methodを使用
        price_window = prices[i-window:i]
        
        # 複数のスケールでの長さを計算
        scales = np.array([2, 3, 4, 5, 6])
        lengths = np.zeros(len(scales))
        
        for s_idx, scale in enumerate(scales):
            length = 0.0
            for m in range(scale):
                max_idx = (window - m - 1) // scale
                if max_idx > 0:
                    curve_length = 0.0
                    for j in range(max_idx):
                        idx1 = m + j * scale
                        idx2 = m + (j + 1) * scale
                        if idx1 < window and idx2 < window:
                            curve_length += abs(price_window[idx2] - price_window[idx1])
                    
                    if max_idx > 0:
                        curve_length = curve_length * (window - 1) / (max_idx * scale)
                        length += curve_length
            
            lengths[s_idx] = length / scale
        
        # 対数回帰でフラクタル次元を計算
        if np.all(lengths > 0):
            log_scales = np.log(scales.astype(np.float64))
            log_lengths = np.log(lengths)
            
            # 線形回帰の傾きを計算
            n_points = len(log_scales)
            sum_x = np.sum(log_scales)
            sum_y = np.sum(log_lengths)
            sum_xy = np.sum(log_scales * log_lengths)
            sum_x2 = np.sum(log_scales * log_scales)
            
            if n_points * sum_x2 - sum_x * sum_x != 0:
                slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x)
                fractal_dim[i] = 2.0 - slope  # フラクタル次元
    
    return fractal_dim


@njit(fastmath=True, cache=True)
def calculate_spectral_power(prices: np.ndarray, window: int = 32) -> np.ndarray:
    """
    スペクトル密度を計算（周波数領域での信号強度）
    
    Args:
        prices: 価格配列
        window: 計算窓（2の冪乗が望ましい）
    
    Returns:
        スペクトル密度配列（高いほど周期性が強い＝トレンド）
    """
    n = len(prices)
    spectral_power = np.zeros(n)
    
    for i in range(window, n):
        # 価格変化率を計算
        returns = np.zeros(window-1)
        valid_count = 0
        for j in range(window-1):
            if prices[i-window+j] != 0:
                returns[valid_count] = (prices[i-window+j+1] - prices[i-window+j]) / prices[i-window+j]
                valid_count += 1
        
        if valid_count > 5:  # 最小限のデータが必要
            # 有効なデータのみを使用
            valid_returns = returns[:valid_count]
            mean_return = np.mean(valid_returns)
            
            # 各周波数成分の強度を計算（簡易DFT）
            max_power = 0.0
            for freq in range(1, min(valid_count//2, 8)):  # 低周波数のみ検討
                cos_sum = 0.0
                sin_sum = 0.0
                for j in range(valid_count):
                    angle = 2.0 * math.pi * freq * j / valid_count
                    cos_sum += (valid_returns[j] - mean_return) * math.cos(angle)
                    sin_sum += (valid_returns[j] - mean_return) * math.sin(angle)
                
                power = (cos_sum * cos_sum + sin_sum * sin_sum) / valid_count
                max_power = max(max_power, power)
            
            spectral_power[i] = max_power
    
    return spectral_power


@njit(fastmath=True, cache=True)
def calculate_neural_adaptation(prices: np.ndarray, window: int = 20, learning_rate: float = 0.1) -> np.ndarray:
    """
    神経適応学習係数を計算（価格予測誤差の学習）
    
    Args:
        prices: 価格配列
        window: 計算窓
        learning_rate: 学習率
    
    Returns:
        神経適応係数配列（高いほど予測可能＝トレンド）
    """
    n = len(prices)
    neural_coeff = np.zeros(n)
    
    # 簡易的な単層パーセプトロンの重みを動的に調整
    weight = 1.0
    
    for i in range(window, n):
        # 過去のデータから予測を計算
        prediction_error = 0.0
        valid_count = 0
        
        for j in range(i-window, i-1):
            if j >= 1:
                # 単純な線形予測
                predicted = prices[j-1] * weight
                actual = prices[j]
                error = actual - predicted
                prediction_error += error * error
                valid_count += 1
                
                # 重みを更新
                if prices[j-1] != 0:
                    weight += learning_rate * error * prices[j-1] / (prices[j-1] * prices[j-1] + 1e-8)
        
        # 予測精度を神経適応係数として使用
        if valid_count > 0:
            mse = prediction_error / valid_count
            neural_coeff[i] = 1.0 / (1.0 + mse)  # 誤差が小さいほど高い係数
    
    return neural_coeff


@njit(fastmath=True, cache=True)
def calculate_ml_prediction(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    機械学習予測値を計算（適応的線形回帰）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        予測精度配列（高いほど予測可能＝トレンド）
    """
    n = len(prices)
    ml_score = np.zeros(n)
    
    for i in range(window, n):
        # 線形回帰で予測
        x_vals = np.arange(window, dtype=np.float64)
        y_vals = prices[i-window:i]
        
        # 最小二乗法で回帰係数を計算
        sum_x = np.sum(x_vals)
        sum_y = np.sum(y_vals)
        sum_xy = np.sum(x_vals * y_vals)
        sum_x2 = np.sum(x_vals * x_vals)
        
        if window * sum_x2 - sum_x * sum_x != 0:
            slope = (window * sum_xy - sum_x * sum_y) / (window * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / window
            
            # 予測精度を計算（R²係数）
            y_mean = sum_y / window
            ss_tot = 0.0
            ss_res = 0.0
            
            for j in range(window):
                y_pred = slope * x_vals[j] + intercept
                ss_res += (y_vals[j] - y_pred) ** 2
                ss_tot += (y_vals[j] - y_mean) ** 2
            
            if ss_tot > 0:
                r_squared = 1.0 - (ss_res / ss_tot)
                ml_score[i] = max(0.0, r_squared)
    
    return ml_score


@njit(fastmath=True, cache=True)
def calculate_efficiency_ratio_simple(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    効率比を計算（価格変動の効率性）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        効率比配列（高いほど効率的＝トレンド）
    """
    n = len(prices)
    efficiency = np.zeros(n)
    
    for i in range(window, n):
        # 全体的な価格変化
        total_change = abs(prices[i] - prices[i-window])
        
        # 各期間の価格変化の合計
        cumulative_change = 0.0
        for j in range(i-window, i):
            cumulative_change += abs(prices[j+1] - prices[j])
        
        # 効率比を計算
        if cumulative_change > 0:
            efficiency[i] = total_change / cumulative_change
        else:
            efficiency[i] = 0.0
    
    return efficiency


@njit(fastmath=True, cache=True)
def calculate_volatility_regime(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    ボラティリティ体制を計算（変動性の状態）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        ボラティリティ体制配列（高いほど高ボラティリティ）
    """
    n = len(prices)
    vol_regime = np.zeros(n)
    
    for i in range(window, n):
        # 価格変化率を計算
        returns = np.zeros(window-1)
        for j in range(window-1):
            if prices[i-window+j] != 0:
                returns[j] = (prices[i-window+j+1] - prices[i-window+j]) / prices[i-window+j]
        
        # 標準偏差を計算
        if len(returns) > 0:
            mean_return = np.mean(returns)
            variance = 0.0
            for ret in returns:
                variance += (ret - mean_return) ** 2
            variance /= len(returns)
            vol_regime[i] = np.sqrt(variance)
    
    return vol_regime


@njit(fastmath=True, cache=True)
def calculate_cycle_strength(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    サイクル強度を計算（周期性の強さ）
    
    Args:
        prices: 価格配列
        window: 計算窓
    
    Returns:
        サイクル強度配列（高いほど強い周期性＝トレンド）
    """
    n = len(prices)
    cycle_strength = np.zeros(n)
    
    for i in range(window, n):
        # 価格変化率を計算
        returns = np.zeros(window-1)
        valid_count = 0
        for j in range(window-1):
            if prices[i-window+j] != 0:
                returns[valid_count] = (prices[i-window+j+1] - prices[i-window+j]) / prices[i-window+j]
                valid_count += 1
        
        if valid_count > 5:  # 最小限のデータが必要
            valid_returns = returns[:valid_count]
            mean_return = np.mean(valid_returns)
            variance = 0.0
            for j in range(valid_count):
                variance += (valid_returns[j] - mean_return) ** 2
            variance /= valid_count
            
            if variance > 1e-12:  # ゼロ除算防止
                max_correlation = 0.0
                
                # 複数のラグでの自己相関を計算
                max_lag = min(valid_count//2, 10)
                for lag in range(1, max_lag):
                    if lag < valid_count:
                        correlation = 0.0
                        count = valid_count - lag
                        
                        for j in range(count):
                            correlation += (valid_returns[j] - mean_return) * (valid_returns[j + lag] - mean_return)
                        
                        if count > 0:
                            correlation = (correlation / count) / variance
                            max_correlation = max(max_correlation, abs(correlation))
                
                cycle_strength[i] = max_correlation
    
    return cycle_strength


@njit(fastmath=True, cache=True)  
def calculate_supreme_trend_state(
    entropy: np.ndarray,
    fractal_dim: np.ndarray,
    spectral_power: np.ndarray,
    neural_coeff: np.ndarray,
    ml_score: np.ndarray,
    efficiency: np.ndarray,
    vol_regime: np.ndarray,
    cycle_strength: np.ndarray,
    threshold: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最終的なトレンド状態を計算
    
    Args:
        entropy: エントロピー値
        fractal_dim: フラクタル次元
        spectral_power: スペクトル密度
        neural_coeff: 神経適応係数
        ml_score: 機械学習スコア
        efficiency: 効率比
        vol_regime: ボラティリティ体制
        cycle_strength: サイクル強度
        threshold: 閾値
    
    Returns:
        (trend_state, confidence, composite_score)
    """
    n = len(entropy)
    trend_state = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n)
    composite_score = np.zeros(n)
    
    for i in range(n):
        # 各指標を0-1に正規化してスコア化
        scores = np.zeros(8)
        
        # エントロピー（低いほどトレンド）
        scores[0] = 1.0 - min(1.0, entropy[i] / 3.0)
        
        # フラクタル次元（1.5に近いほどトレンド）
        if fractal_dim[i] > 0:
            scores[1] = 1.0 - min(1.0, abs(fractal_dim[i] - 1.5) / 0.5)
        
        # スペクトル密度（高いほどトレンド）
        scores[2] = min(1.0, spectral_power[i] * 10.0)
        
        # 神経適応係数（高いほどトレンド）
        scores[3] = min(1.0, neural_coeff[i])
        
        # 機械学習スコア（高いほどトレンド）
        scores[4] = min(1.0, ml_score[i])
        
        # 効率比（高いほどトレンド）
        scores[5] = min(1.0, efficiency[i])
        
        # ボラティリティ体制（適度な値がトレンド）
        if vol_regime[i] > 0:
            scores[6] = 1.0 - min(1.0, abs(vol_regime[i] - 0.02) / 0.02)
        
        # サイクル強度（高いほどトレンド）
        scores[7] = min(1.0, cycle_strength[i] * 5.0)
        
        # 重み付き合成スコア（実証的に効果的な指標により高い重み）
        # 効率比、機械学習、神経適応を重視し、エントロピーとフラクタル次元で補強
        weights = np.array([0.15, 0.15, 0.10, 0.20, 0.20, 0.15, 0.03, 0.02])
        composite_score[i] = np.sum(scores * weights)
        
        # 信頼度を計算（標準偏差の逆数）
        score_std = np.std(scores)
        confidence[i] = 1.0 / (1.0 + score_std)
        
        # バイナリ判定
        trend_state[i] = 1 if composite_score[i] >= threshold else 0
    
    return trend_state, confidence, composite_score


class SupremeTrendState(Indicator):
    """
    Supreme Trend State Indicator
    
    超高精度・超低遅延・超適応性のトレンド状態判別インジケーター
    複数の学問分野から厳選した最強アルゴリズムを多角的に組み合わせ、
    トレンドかレンジかを1/0のバイナリで出力
    
    採用アルゴリズム:
    1. 情報エントロピー（情報理論）
    2. フラクタル次元（カオス理論）
    3. スペクトル密度（デジタル信号処理）
    4. 神経適応学習（人工知能）
    5. 機械学習予測（統計学習）
    6. 効率比（金融工学）
    7. ボラティリティ体制（リスク管理）
    8. サイクル強度（時系列解析）
    """
    
    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.6,
        src_type: str = 'hlc3',
        use_dynamic_period: bool = True,
        detector_type: str = 'absolute_ultimate',
        max_cycle: int = 50,
        min_cycle: int = 8,
        learning_rate: float = 0.1
    ):
        """
        コンストラクタ
        
        Args:
            window: 計算窓サイズ
            threshold: トレンド判定閾値 (0.5-0.8推奨)
            src_type: 価格ソースタイプ
            use_dynamic_period: 動的期間を使用するか
            detector_type: サイクル検出器タイプ
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
            learning_rate: 神経学習率
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        super().__init__(f"SupremeTrendState(w={window},th={threshold:.2f}{dynamic_str})")
        
        self.window = window
        self.threshold = threshold
        self.src_type = src_type
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.learning_rate = learning_rate
        
        # 動的期間用サイクル検出器
        self.cycle_detector = None
        if self.use_dynamic_period:
            self.cycle_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                src_type=self.src_type
            )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    first_val = float(data[0, -1]) if data.ndim > 1 else float(data[0])
                    last_val = float(data[-1, -1]) if data.ndim > 1 else float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            params_sig = f"{self.window}_{self.threshold}_{self.src_type}_{self.use_dynamic_period}_{self.detector_type}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.window}_{self.threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SupremeTrendStateResult:
        """
        Supreme Trend Stateを計算
        
        Args:
            data: 価格データ
        
        Returns:
            SupremeTrendStateResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # 価格データを取得
            if isinstance(data, pd.DataFrame):
                prices = PriceSource.calculate_source(data, self.src_type)
            else:
                prices = data[:, 3] if data.ndim > 1 else data
            
            prices = np.asarray(prices, dtype=np.float64)
            
            # 動的期間の計算
            if self.use_dynamic_period and self.cycle_detector is not None:
                dynamic_cycles = self.cycle_detector.calculate(data)
                window_size = int(np.mean(dynamic_cycles[~np.isnan(dynamic_cycles)]))
                window_size = max(self.min_cycle, min(window_size, self.max_cycle))
            else:
                window_size = self.window
            
            # 各アルゴリズムを実行
            entropy = calculate_information_entropy(prices, window_size)
            fractal_dim = calculate_fractal_dimension(prices, window_size)
            spectral_power = calculate_spectral_power(prices, min(32, window_size))
            neural_coeff = calculate_neural_adaptation(prices, window_size, self.learning_rate)
            ml_score = calculate_ml_prediction(prices, window_size)
            efficiency = calculate_efficiency_ratio_simple(prices, window_size)
            vol_regime = calculate_volatility_regime(prices, window_size)
            cycle_strength = calculate_cycle_strength(prices, window_size)
            
            # 最終判定
            trend_state, confidence, composite_score = calculate_supreme_trend_state(
                entropy, fractal_dim, spectral_power, neural_coeff,
                ml_score, efficiency, vol_regime, cycle_strength,
                self.threshold
            )
            
            # 結果を作成
            result = SupremeTrendStateResult(
                trend_state=trend_state,
                confidence=confidence,
                entropy_score=entropy,
                fractal_dimension=fractal_dim,
                spectral_power=spectral_power,
                neural_adaptation=neural_coeff,
                ml_prediction=ml_score,
                efficiency_ratio=efficiency,
                volatility_regime=vol_regime,
                cycle_strength=cycle_strength,
                composite_score=composite_score
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_state
            
            return result
            
        except Exception as e:
            self.logger.error(f"Supreme Trend State計算エラー: {e}")
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            return SupremeTrendStateResult(
                trend_state=np.zeros(n, dtype=np.int32),
                confidence=np.zeros(n),
                entropy_score=np.zeros(n),
                fractal_dimension=np.zeros(n),
                spectral_power=np.zeros(n),
                neural_adaptation=np.zeros(n),
                ml_prediction=np.zeros(n),
                efficiency_ratio=np.zeros(n),
                volatility_regime=np.zeros(n),
                cycle_strength=np.zeros(n),
                composite_score=np.zeros(n)
            )
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態（バイナリ出力）を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.trend_state.copy()
    
    def get_confidence(self) -> Optional[np.ndarray]:
        """信頼度を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.confidence.copy()
    
    def get_composite_score(self) -> Optional[np.ndarray]:
        """複合スコアを取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        result = self._result_cache[self._cache_keys[-1]]
        return result.composite_score.copy()
    
    def is_trending(self) -> bool:
        """現在がトレンド状態かを判定"""
        trend_state = self.get_trend_state()
        if trend_state is None or len(trend_state) == 0:
            return False
        return bool(trend_state[-1])
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if self.cycle_detector is not None:
            self.cycle_detector.reset()