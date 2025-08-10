#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum-Fractal Adaptive Volatility Channel (QFAVC)
量子フラクタル適応ボラティリティチャネル

革新的な動的チャネルインジケーター - 既存の概念を完全に超越した新アルゴリズム

核となる革新要素:
🌌 量子もつれ理論による価格相関解析
🔮 フラクタル次元によるmarket complexity計測
🌊 液体力学による市場フロー状態判定
🎯 適応カルマンフィルターによる動的ノイズモデリング
📊 GARCH-Hurst統合ボラティリティモデル
⚡ 超低遅延・超高追従性センターライン
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange, vectorize, float64, int64, boolean
import math

from .indicator import Indicator
from .price_source import PriceSource
from .cycle.ehlers_unified_dc import EhlersUnifiedDC
from .kalman_filter import KalmanFilter


@dataclass
class QFAVCResult:
    """QFAVC計算結果"""
    # チャネル要素
    centerline: np.ndarray          # 量子適応センターライン
    upper_channel: np.ndarray       # 上部チャネル
    lower_channel: np.ndarray       # 下部チャネル
    dynamic_width: np.ndarray       # 動的チャネル幅
    
    # 量子解析成分
    quantum_entanglement: np.ndarray    # 量子もつれ強度
    coherence_factor: np.ndarray        # 量子コヒーレンス因子
    superposition_state: np.ndarray     # 重ね合わせ状態
    
    # フラクタル解析成分
    fractal_dimension: np.ndarray       # フラクタル次元
    hurst_exponent: np.ndarray          # ハースト指数
    complexity_index: np.ndarray        # 複雑性指数
    
    # 流体力学成分
    flow_regime: np.ndarray             # フロー状態 (層流=1, 乱流=-1)
    reynolds_number: np.ndarray         # レイノルズ数
    turbulence_intensity: np.ndarray    # 乱流強度
    
    # 適応ボラティリティ成分
    garch_volatility: np.ndarray        # GARCH推定ボラティリティ
    adaptive_variance: np.ndarray       # 適応分散
    volatility_regime: np.ndarray       # ボラティリティレジーム
    
    # 統合指標
    market_phase: np.ndarray            # 市場フェーズ判定
    adaptation_factor: np.ndarray       # 適応因子
    confidence_score: np.ndarray        # 信頼度スコア


# === 量子解析関数群 ===

@njit(fastmath=True, parallel=True, cache=True)
def calculate_quantum_entanglement(prices: np.ndarray, window: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    量子もつれ理論による価格相関解析
    
    価格系列間の非線形相関を量子もつれ状態として解釈し、
    市場の隠れた相関構造を検出する革新的アルゴリズム
    """
    n = len(prices)
    entanglement = np.full(n, np.nan)
    coherence = np.full(n, np.nan)
    superposition = np.full(n, np.nan)
    
    for i in prange(window, n):
        # 局所価格ベクトルの抽出
        local_prices = prices[i-window:i]
        
        # 量子状態ベクトルの構成
        returns = np.diff(local_prices)
        if len(returns) == 0:
            continue
            
        # 量子もつれエントロピー計算
        normalized_returns = returns / (np.std(returns) + 1e-10)
        
        # ベル状態類似度の計算
        bell_correlation = 0.0
        coherence_sum = 0.0
        
        for j in range(len(normalized_returns) - 1):
            for k in range(j + 1, len(normalized_returns)):
                # 量子もつれ測度
                correlation = normalized_returns[j] * normalized_returns[k]
                bell_correlation += math.exp(-abs(correlation))
                
                # コヒーレンス測度
                phase_diff = math.atan2(normalized_returns[k], normalized_returns[j])
                coherence_sum += math.cos(phase_diff)
        
        # 正規化
        pairs_count = len(normalized_returns) * (len(normalized_returns) - 1) / 2
        if pairs_count > 0:
            entanglement[i] = bell_correlation / pairs_count
            coherence[i] = abs(coherence_sum) / pairs_count
        
        # 重ね合わせ状態の計算
        price_momentum = (prices[i] - prices[i-window//2]) / (prices[i-window//2] + 1e-10)
        superposition[i] = math.tanh(price_momentum) * coherence[i]
    
    return entanglement, coherence, superposition


@njit(fastmath=True, parallel=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, window: int = 34) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    フラクタル次元とハースト指数による複雑性解析
    
    市場の自己相似性とトレンド持続性を定量化し、
    price actionの本質的な複雑さを測定
    """
    n = len(prices)
    fractal_dim = np.full(n, np.nan)
    hurst_exp = np.full(n, np.nan)
    complexity = np.full(n, np.nan)
    
    for i in prange(window, n):
        local_prices = prices[i-window:i]
        
        # Box-counting法によるフラクタル次元
        price_range = np.max(local_prices) - np.min(local_prices)
        if price_range == 0:
            continue
            
        # スケーリング解析
        scales = np.array([2, 4, 8, 16])
        log_scales = np.log(scales)
        log_variations = np.zeros(len(scales))
        
        for j, scale in enumerate(scales):
            if scale >= len(local_prices):
                continue
                
            # 各スケールでの変動計算
            variations = 0.0
            count = 0
            for k in range(0, len(local_prices) - scale, scale):
                segment = local_prices[k:k+scale]
                variations += np.max(segment) - np.min(segment)
                count += 1
            
            if count > 0:
                log_variations[j] = math.log(variations / count + 1e-10)
        
        # フラクタル次元推定 (線形回帰の傾き)（ゼロ除算対策）
        std_scales = np.std(log_scales)
        std_variations = np.std(log_variations)
        if std_scales > 1e-10 and std_variations > 1e-10:
            correlation = np.corrcoef(log_scales, log_variations)[0, 1]
            if not np.isnan(correlation):
                slope = correlation * std_variations / std_scales
                fractal_dim[i] = 2.0 - slope  # 理論的な調整
            else:
                fractal_dim[i] = 1.5  # デフォルト値
        else:
            fractal_dim[i] = 1.5  # デフォルト値
        
        # ハースト指数の計算 (R/S解析)
        returns = np.diff(local_prices)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            deviations = returns - mean_return
            cumulative_deviations = np.cumsum(deviations)
            
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = max(np.std(returns), 1e-10)
            
            if S > 1e-10 and R > 1e-10 and len(returns) > 1:
                log_rs = math.log(R/S)
                log_n = math.log(len(returns))
                if abs(log_n) > 1e-10:
                    hurst_exp[i] = log_rs / log_n
                else:
                    hurst_exp[i] = 0.5  # デフォルト値
            else:
                hurst_exp[i] = 0.5  # デフォルト値
        
        # 複雑性指数 (フラクタル次元とハースト指数の統合)
        if not np.isnan(fractal_dim[i]) and not np.isnan(hurst_exp[i]):
            complexity[i] = fractal_dim[i] * (1.0 - abs(hurst_exp[i] - 0.5))
    
    return fractal_dim, hurst_exp, complexity


@njit(fastmath=True, parallel=True, cache=True)
def calculate_fluid_dynamics(prices: np.ndarray, volume: np.ndarray, window: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    液体力学による市場フロー解析
    
    価格の動きを流体の動きとして解釈し、
    層流（トレンド）と乱流（レンジ）の状態を判定
    """
    n = len(prices)
    flow_regime = np.full(n, np.nan)
    reynolds_number = np.full(n, np.nan)
    turbulence = np.full(n, np.nan)
    
    for i in prange(window, n):
        local_prices = prices[i-window:i]
        local_volume = volume[i-window:i] if len(volume) > i else np.ones(window)
        
        # 速度場の計算 (価格変化率)
        velocity = np.diff(local_prices)
        if len(velocity) == 0:
            continue
            
        # 密度場の計算 (正規化ボリューム)
        density = local_volume / (np.mean(local_volume) + 1e-10)
        
        # レイノルズ数の計算（ゼロ除算対策強化）
        characteristic_velocity = max(np.std(velocity), 1e-10)
        characteristic_length = window
        mean_density = max(np.mean(density), 1e-10)
        viscosity = 1.0 / mean_density  # 逆粘性
        
        if viscosity > 1e-10:
            reynolds = (characteristic_velocity * characteristic_length) / viscosity
        else:
            reynolds = 1000.0  # デフォルト値
        reynolds_number[i] = reynolds
        
        # フロー状態の判定
        critical_reynolds = 2300  # 臨界レイノルズ数
        if reynolds > critical_reynolds:
            flow_regime[i] = -1  # 乱流 (レンジ相場)
        else:
            flow_regime[i] = 1   # 層流 (トレンド相場)
        
        # 乱流強度の計算（ゼロ除算対策）
        velocity_fluctuations = velocity - np.mean(velocity)
        turbulence_kinetic_energy = np.mean(velocity_fluctuations ** 2)
        velocity_squared = max(characteristic_velocity ** 2, 1e-10)
        turbulence[i] = turbulence_kinetic_energy / velocity_squared
        
        # 正規化
        turbulence[i] = min(max(turbulence[i], 0.0), 1.0)
    
    return flow_regime, reynolds_number, turbulence


@njit(fastmath=True, cache=True)
def garch_volatility_estimation(returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
    """
    GARCH(1,1)モデルによる適応ボラティリティ推定
    
    条件付き分散の動的モデリングにより、
    時変ボラティリティを高精度で推定
    """
    n = len(returns)
    volatility = np.full(n, np.nan)
    
    if n < 2:
        return volatility
    
    # 初期値設定
    long_run_variance = np.var(returns)
    conditional_variance = long_run_variance
    
    for i in range(1, n):
        # GARCH(1,1)更新式
        lagged_return_squared = returns[i-1] ** 2
        conditional_variance = (
            (1 - alpha - beta) * long_run_variance +
            alpha * lagged_return_squared +
            beta * conditional_variance
        )
        
        volatility[i] = math.sqrt(max(conditional_variance, 1e-10))
    
    return volatility


@njit(fastmath=True, parallel=True, cache=True)
def adaptive_kalman_centerline(prices: np.ndarray, quantum_coherence: np.ndarray, 
                              process_noise_factor: float = 0.01) -> np.ndarray:
    """
    適応カルマンフィルターによる量子適応センターライン
    
    量子コヒーレンス因子に基づいてノイズモデルを動的調整し、
    市場状況に応じて最適な平滑化を実現
    """
    n = len(prices)
    filtered_prices = np.full(n, np.nan)
    
    if n < 2:
        return filtered_prices
    
    # カルマンフィルター初期化
    state_estimate = prices[0]
    error_covariance = 1.0
    
    filtered_prices[0] = state_estimate
    
    for i in range(1, n):
        # 量子コヒーレンスに基づく適応的ノイズ調整
        coherence = quantum_coherence[i] if not np.isnan(quantum_coherence[i]) else 0.5
        
        # プロセスノイズとobservationノイズの動的調整
        process_noise = process_noise_factor * (1.0 - coherence)
        observation_noise = 0.1 * (1.0 + coherence)
        
        # 予測ステップ
        state_prediction = state_estimate
        error_prediction = error_covariance + process_noise
        
        # 更新ステップ（ゼロ除算対策）
        denominator = error_prediction + observation_noise
        if denominator > 1e-10:
            kalman_gain = error_prediction / denominator
        else:
            kalman_gain = 0.5  # デフォルト値
            
        state_estimate = state_prediction + kalman_gain * (prices[i] - state_prediction)
        error_covariance = (1 - kalman_gain) * error_prediction
        
        filtered_prices[i] = state_estimate
    
    return filtered_prices


@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_channel_width(
    garch_vol: np.ndarray,
    hurst_exp: np.ndarray,
    flow_regime: np.ndarray,
    complexity: np.ndarray,
    base_multiplier: float = 2.0
) -> np.ndarray:
    """
    統合的動的チャネル幅計算
    
    GARCH-Hurst統合モデル、フロー状態、複雑性指数を統合し、
    市場状況に完全適応する革新的チャネル幅を算出
    """
    n = len(garch_vol)
    dynamic_width = np.full(n, np.nan)
    
    for i in prange(n):
        if (np.isnan(garch_vol[i]) or np.isnan(hurst_exp[i]) or 
            np.isnan(flow_regime[i]) or np.isnan(complexity[i])):
            continue
        
        # ベースボラティリティ (GARCH)
        base_volatility = garch_vol[i]
        
        # ハースト指数による調整
        # H > 0.5: トレンド持続性 → チャネル幅縮小
        # H < 0.5: 平均回帰性 → チャネル幅拡大
        hurst_factor = 1.0 + 2.0 * abs(hurst_exp[i] - 0.5)
        if hurst_exp[i] > 0.5:
            hurst_factor = 1.0 / hurst_factor  # トレンド時は縮小
        
        # フロー状態による調整
        flow_factor = 1.0
        if flow_regime[i] > 0:  # 層流 (トレンド)
            flow_factor = 0.7  # チャネル幅縮小
        else:  # 乱流 (レンジ)
            flow_factor = 1.5  # チャネル幅拡大
        
        # 複雑性による微調整
        complexity_factor = 1.0 + 0.5 * complexity[i]
        
        # 統合的チャネル幅
        dynamic_width[i] = (
            base_multiplier * 
            base_volatility * 
            hurst_factor * 
            flow_factor * 
            complexity_factor
        )
        
        # 安全な範囲に制限
        dynamic_width[i] = max(min(dynamic_width[i], base_multiplier * 3.0), base_multiplier * 0.3)
    
    return dynamic_width


class QuantumFractalAdaptiveChannel(Indicator):
    """
    Quantum-Fractal Adaptive Volatility Channel (QFAVC)
    
    革新的な動的チャネルインジケーター
    - 量子もつれ理論による価格相関解析
    - フラクタル次元による複雑性評価
    - 液体力学による市場フロー状態判定
    - 適応カルマンフィルターセンターライン
    - GARCH-Hurst統合ボラティリティモデル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        quantum_window: int = 21,
        fractal_window: int = 34,
        flow_window: int = 21,
        garch_window: int = 55,
        
        # 動的期間決定用
        use_dynamic_periods: bool = True,
        dc_detector_type: str = 'dudi_e',
        dc_cycle_part: float = 0.4,
        
        # ボラティリティパラメータ
        base_multiplier: float = 2.0,
        garch_alpha: float = 0.1,
        garch_beta: float = 0.85,
        
        # カルマンフィルターパラメータ
        kalman_process_noise: float = 0.01,
        
        # データソース
        src_type: str = 'hlc3',
        volume_src: str = 'volume'
    ):
        """
        Quantum-Fractal Adaptive Channel コンストラクタ
        """
        super().__init__(f"QFAVC(qw={quantum_window},fw={fractal_window})")
        
        # パラメータ保存
        self.quantum_window = quantum_window
        self.fractal_window = fractal_window
        self.flow_window = flow_window
        self.garch_window = garch_window
        
        self.use_dynamic_periods = use_dynamic_periods
        self.base_multiplier = base_multiplier
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta
        self.kalman_process_noise = kalman_process_noise
        
        self.src_type = src_type
        self.volume_src = volume_src
        
        # 依存コンポーネント
        self.price_source = PriceSource()
        
        # 動的期間決定用（オプション）
        if self.use_dynamic_periods:
            self.dominant_cycle = EhlersUnifiedDC(
                detector_type=dc_detector_type,
                cycle_part=dc_cycle_part,
                src_type=src_type
            )
        else:
            self.dominant_cycle = None
        
        # 結果キャッシュ
        self._result_cache = {}
        self._cache_keys = []
        self._max_cache_size = 5
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ生成（高速化）"""
        try:
            if isinstance(data, pd.DataFrame):
                shape_signature = data.shape
                first_close = float(data.iloc[0].get('close', data.iloc[0, -1])) if len(data) > 0 else 0.0
                last_close = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if len(data) > 0 else 0.0
                data_signature = (shape_signature, first_close, last_close)
            else:
                shape_signature = data.shape
                first_val = float(data[0, -1]) if len(data) > 0 and data.ndim > 1 else float(data[0]) if len(data) > 0 else 0.0
                last_val = float(data[-1, -1]) if len(data) > 0 and data.ndim > 1 else float(data[-1]) if len(data) > 0 else 0.0
                data_signature = (shape_signature, first_val, last_val)
            
            params_signature = (
                self.quantum_window, self.fractal_window, self.flow_window,
                self.base_multiplier, self.src_type
            )
            
            return f"{hash(data_signature)}_{hash(params_signature)}"
        except:
            return f"{id(data)}_{self.quantum_window}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QFAVCResult:
        """
        QFAVC計算メイン関数
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                return self._result_cache[data_hash]
            
            # 価格データ抽出
            if isinstance(data, pd.DataFrame):
                price_series = self.price_source.get_source(data, self.src_type)
                if hasattr(price_series, 'values'):
                    prices = price_series.values
                else:
                    prices = price_series
                volume = data.get(self.volume_src, pd.Series(np.ones(len(data)))).values
            else:
                prices = data[:, 3] if data.ndim > 1 else data  # close価格
                volume = np.ones(len(prices))  # ダミーボリューム
            
            n = len(prices)
            
            # 動的期間決定（オプション）
            if self.use_dynamic_periods and self.dominant_cycle:
                try:
                    dc_values = self.dominant_cycle.calculate(data)
                    avg_period = int(np.nanmean(dc_values))
                    
                    # 期間を動的調整
                    self.quantum_window = max(min(avg_period, 55), 8)
                    self.fractal_window = max(min(int(avg_period * 1.6), 89), 13)
                    self.flow_window = max(min(avg_period, 34), 8)
                except:
                    pass  # エラー時はデフォルト値使用
            
            # === 段階1: 量子解析 ===
            quantum_entanglement, coherence_factor, superposition_state = calculate_quantum_entanglement(
                prices, self.quantum_window
            )
            
            # === 段階2: フラクタル解析 ===
            fractal_dimension, hurst_exponent, complexity_index = calculate_fractal_dimension(
                prices, self.fractal_window
            )
            
            # === 段階3: 流体力学解析 ===
            flow_regime, reynolds_number, turbulence_intensity = calculate_fluid_dynamics(
                prices, volume, self.flow_window
            )
            
            # === 段階4: GARCH ボラティリティ推定 ===  
            returns = np.diff(prices)
            garch_volatility = garch_volatility_estimation(returns, self.garch_alpha, self.garch_beta)
            # 長さを価格と合わせる
            garch_vol_padded = np.full(n, np.nan)
            garch_vol_padded[1:] = garch_volatility
            
            # === 段階5: 適応センターライン ===
            centerline = adaptive_kalman_centerline(
                prices, coherence_factor, self.kalman_process_noise
            )
            
            # === 段階6: 動的チャネル幅計算 ===
            dynamic_width = calculate_dynamic_channel_width(
                garch_vol_padded, hurst_exponent, flow_regime, 
                complexity_index, self.base_multiplier
            )
            
            # === 段階7: チャネル構築 ===
            upper_channel = centerline + dynamic_width
            lower_channel = centerline - dynamic_width
            
            # === 段階8: 統合指標計算 ===
            market_phase = np.full(n, np.nan)
            adaptation_factor = np.full(n, np.nan)
            confidence_score = np.full(n, np.nan)
            adaptive_variance = np.full(n, np.nan)
            volatility_regime = np.full(n, np.nan)
            
            for i in range(n):
                if not np.isnan(flow_regime[i]) and not np.isnan(hurst_exponent[i]):
                    # 市場フェーズ判定
                    if flow_regime[i] > 0 and hurst_exponent[i] > 0.5:
                        market_phase[i] = 1  # 強いトレンド
                    elif flow_regime[i] < 0 and hurst_exponent[i] < 0.5:
                        market_phase[i] = -1  # 強いレンジ
                    else:
                        market_phase[i] = 0  # 中間状態
                    
                    # 適応因子
                    adaptation_factor[i] = abs(hurst_exponent[i] - 0.5) * (1.0 - turbulence_intensity[i])
                    
                    # 信頼度スコア
                    confidence_score[i] = (
                        coherence_factor[i] * 0.4 +
                        (1.0 - turbulence_intensity[i]) * 0.3 +
                        abs(flow_regime[i]) * 0.3
                    ) if not np.isnan(coherence_factor[i]) and not np.isnan(turbulence_intensity[i]) else 0.5
                
                # 適応分散
                if not np.isnan(garch_vol_padded[i]):
                    adaptive_variance[i] = garch_vol_padded[i] ** 2
                
                # ボラティリティレジーム
                if not np.isnan(garch_vol_padded[i]) and i > 20:
                    recent_vol = np.nanmean(garch_vol_padded[max(0, i-20):i])
                    if garch_vol_padded[i] > recent_vol * 1.5:
                        volatility_regime[i] = 1  # 高ボラティリティ
                    elif garch_vol_padded[i] < recent_vol * 0.7:
                        volatility_regime[i] = -1  # 低ボラティリティ
                    else:
                        volatility_regime[i] = 0  # 正常ボラティリティ
            
            # 結果構築
            result = QFAVCResult(
                # チャネル要素
                centerline=centerline,
                upper_channel=upper_channel,
                lower_channel=lower_channel,
                dynamic_width=dynamic_width,
                
                # 量子解析成分
                quantum_entanglement=quantum_entanglement,
                coherence_factor=coherence_factor,
                superposition_state=superposition_state,
                
                # フラクタル解析成分
                fractal_dimension=fractal_dimension,
                hurst_exponent=hurst_exponent,
                complexity_index=complexity_index,
                
                # 流体力学成分
                flow_regime=flow_regime,
                reynolds_number=reynolds_number,
                turbulence_intensity=turbulence_intensity,
                
                # 適応ボラティリティ成分
                garch_volatility=garch_vol_padded,
                adaptive_variance=adaptive_variance,
                volatility_regime=volatility_regime,
                
                # 統合指標
                market_phase=market_phase,
                adaptation_factor=adaptation_factor,
                confidence_score=confidence_score
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return result
            
        except Exception as e:
            self.logger.error(f"QFAVC計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            empty_array = np.full(n, np.nan)
            return QFAVCResult(
                centerline=empty_array, upper_channel=empty_array, lower_channel=empty_array,
                dynamic_width=empty_array, quantum_entanglement=empty_array, coherence_factor=empty_array,
                superposition_state=empty_array, fractal_dimension=empty_array, hurst_exponent=empty_array,
                complexity_index=empty_array, flow_regime=empty_array, reynolds_number=empty_array,
                turbulence_intensity=empty_array, garch_volatility=empty_array, adaptive_variance=empty_array,
                volatility_regime=empty_array, market_phase=empty_array, adaptation_factor=empty_array,
                confidence_score=empty_array
            )
    
    def get_channel_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """チャネルバンド取得"""
        if data is not None:
            result = self.calculate(data)
        elif self._result_cache:
            result = list(self._result_cache.values())[-1]
        else:
            empty_array = np.array([])
            return empty_array, empty_array, empty_array
        return result.centerline, result.upper_channel, result.lower_channel
    
    def get_market_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict:
        """市場知能レポート生成"""
        if data is not None:
            result = self.calculate(data)
        elif self._result_cache:
            result = list(self._result_cache.values())[-1]
        else:
            return {"status": "no_data"}
        
        # 最新値の取得
        latest_idx = -1
        while latest_idx >= -len(result.market_phase) and np.isnan(result.market_phase[latest_idx]):
            latest_idx -= 1
        
        if abs(latest_idx) >= len(result.market_phase):
            return {"status": "insufficient_data"}
        
        return {
            "current_market_phase": int(result.market_phase[latest_idx]) if not np.isnan(result.market_phase[latest_idx]) else 0,
            "hurst_exponent": float(result.hurst_exponent[latest_idx]) if not np.isnan(result.hurst_exponent[latest_idx]) else 0.5,
            "flow_regime": "layer_flow" if result.flow_regime[latest_idx] > 0 else "turbulent_flow",
            "quantum_coherence": float(result.coherence_factor[latest_idx]) if not np.isnan(result.coherence_factor[latest_idx]) else 0.5,
            "complexity_index": float(result.complexity_index[latest_idx]) if not np.isnan(result.complexity_index[latest_idx]) else 0.5,
            "confidence_score": float(result.confidence_score[latest_idx]) if not np.isnan(result.confidence_score[latest_idx]) else 0.5,
            "volatility_regime": int(result.volatility_regime[latest_idx]) if not np.isnan(result.volatility_regime[latest_idx]) else 0
        } 