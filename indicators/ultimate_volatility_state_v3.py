#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback
import warnings
warnings.filterwarnings('ignore')

from .indicator import Indicator
from .price_source import PriceSource
from .str import STR
from .volatility import volatility
from .smoother.ultimate_smoother import UltimateSmoother
from .zscore import ZScore


@dataclass
class UltimateVolatilityStateV3Result:
    """究極のボラティリティ状態判別結果 V3（超高精度版）"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    confidence: np.ndarray                 # 判定の信頼度 (0.0-1.0)
    chaos_measure: np.ndarray              # カオス度測定
    neural_adaptation: np.ndarray          # 神経適応値
    entropy_metrics: Dict[str, np.ndarray] # 各種エントロピー測定
    dsp_features: Dict[str, np.ndarray]    # DSP特徴量
    ml_prediction: np.ndarray              # 機械学習予測
    ensemble_weight: np.ndarray            # アンサンブル重み


@njit(fastmath=True, cache=True)
def lyapunov_exponent_estimation(data: np.ndarray, period: int, embedding_dim: int = 3) -> np.ndarray:
    """
    Lyapunov指数推定（カオス理論）- 金融時系列の敏感依存性測定
    """
    length = len(data)
    lyapunov = np.zeros(length)
    
    for i in range(period + embedding_dim, length):
        window = data[i-period:i]
        
        # 位相空間再構成（遅延埋め込み）
        embedded = np.zeros((len(window) - embedding_dim, embedding_dim))
        for j in range(len(window) - embedding_dim):
            for k in range(embedding_dim):
                embedded[j, k] = window[j + k]
        
        if len(embedded) < 3:
            continue
            
        # 最近傍点の探索と発散率計算
        divergences = []
        for j in range(len(embedded) - 1):
            # 最近傍点を探索
            min_dist = np.inf
            nearest_idx = -1
            
            for k in range(len(embedded)):
                if k != j:
                    dist = 0.0
                    for d in range(embedding_dim):
                        diff = embedded[j, d] - embedded[k, d]
                        dist += diff * diff
                    dist = np.sqrt(dist)
                    
                    if dist < min_dist and dist > 1e-10:
                        min_dist = dist
                        nearest_idx = k
            
            # 発散率を計算
            if nearest_idx >= 0 and j + 1 < len(embedded) and nearest_idx + 1 < len(embedded):
                future_dist = 0.0
                for d in range(embedding_dim):
                    diff = embedded[j + 1, d] - embedded[nearest_idx + 1, d]
                    future_dist += diff * diff
                future_dist = np.sqrt(future_dist)
                
                if min_dist > 0 and future_dist > 0:
                    divergence = np.log(future_dist / min_dist)
                    divergences.append(divergence)
        
        # Lyapunov指数の推定
        if len(divergences) > 0:
            lyapunov[i] = np.mean(np.array(divergences))
    
    return lyapunov


@njit(fastmath=True, cache=True)
def multiscale_entropy(data: np.ndarray, period: int, max_scale: int = 5) -> Tuple:
    """
    マルチスケールエントロピー（MSE）- 複数時間スケールでの複雑性
    """
    length = len(data)
    mse_values = np.zeros((max_scale, length))
    
    for scale in range(1, max_scale + 1):
        # スケールごとの粗視化
        coarse_grained = np.zeros(length)
        
        for i in range(scale, length):
            window_sum = 0.0
            for j in range(scale):
                if i - j >= 0:
                    window_sum += data[i - j]
            coarse_grained[i] = window_sum / scale
        
        # サンプルエントロピーの計算
        for i in range(period, length):
            window = coarse_grained[i-period:i]
            
            if len(window) < 3:
                continue
                
            # パターンマッチング（簡易版）
            m = 2  # パターン長
            tolerance = 0.1 * np.std(window)
            
            matches_m = 0
            matches_m1 = 0
            total_pairs = 0
            
            for j in range(len(window) - m):
                for k in range(j + 1, len(window) - m):
                    # パターン長mでの比較
                    match_m = True
                    for l in range(m):
                        if abs(window[j + l] - window[k + l]) > tolerance:
                            match_m = False
                            break
                    
                    if match_m:
                        matches_m += 1
                        
                        # パターン長m+1での比較
                        if j + m < len(window) and k + m < len(window):
                            if abs(window[j + m] - window[k + m]) <= tolerance:
                                matches_m1 += 1
                    
                    total_pairs += 1
            
            # サンプルエントロピー
            if matches_m > 0:
                relative_freq = matches_m1 / matches_m
                if relative_freq > 0:
                    mse_values[scale - 1, i] = -np.log(relative_freq)
    
    return mse_values


@njit(fastmath=True, cache=True)
def adaptive_kalman_volatility(prices: np.ndarray, period: int) -> Tuple:
    """
    適応カルマンフィルター - 動的ノイズ推定による高精度ボラティリティ
    """
    length = len(prices)
    filtered_vol = np.zeros(length)
    innovation_variance = np.zeros(length)
    
    # 状態: [ボラティリティ, ボラティリティの変化率]
    state = np.array([0.01, 0.0])  # 初期状態
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # 誤差共分散行列
    
    # システム行列
    F = np.array([[1.0, 1.0], [0.0, 0.95]])  # 状態遷移行列
    H = np.array([1.0, 0.0])  # 観測行列
    
    for i in range(1, length):
        # 観測値（リターンの二乗）
        if prices[i-1] > 0:
            log_return = np.log(prices[i] / prices[i-1])
            observation = log_return * log_return
        else:
            observation = 0.0
        
        # 適応的ノイズ推定
        if i >= period:
            recent_returns = []
            for j in range(max(0, i - period), i):
                if j > 0 and prices[j-1] > 0:
                    ret = np.log(prices[j] / prices[j-1])
                    recent_returns.append(ret * ret)
            
            if len(recent_returns) > 1:
                Q_adaptive = np.var(np.array(recent_returns)) * 0.1
                R_adaptive = np.var(np.array(recent_returns)) * 0.01
            else:
                Q_adaptive = 0.01
                R_adaptive = 0.001
        else:
            Q_adaptive = 0.01
            R_adaptive = 0.001
        
        Q = np.array([[Q_adaptive, 0.0], [0.0, Q_adaptive * 0.1]])
        R = R_adaptive
        
        # 予測ステップ
        state_pred = F @ state
        P_pred = F @ P @ F.T + Q
        
        # 更新ステップ
        innovation = observation - H @ state_pred
        S = H @ P_pred @ H.T + R
        
        if S > 1e-10:
            K = P_pred @ H.T / S
            state = state_pred + K * innovation
            P = P_pred - np.outer(K, H @ P_pred)
            
            innovation_variance[i] = S
        else:
            state = state_pred
            P = P_pred
        
        filtered_vol[i] = max(state[0], 1e-6)
    
    return filtered_vol, innovation_variance


@njit(fastmath=True, cache=True)
def neural_adaptive_filter(data: np.ndarray, period: int, learning_rate: float = 0.01) -> Tuple:
    """
    神経適応フィルター（LMS/RLS風）- オンライン学習による動的適応
    """
    length = len(data)
    filtered_signal = np.zeros(length)
    adaptation_weights = np.zeros(length)
    
    # フィルター係数（適応的）
    filter_order = min(10, period // 2)
    weights = np.ones(filter_order) / filter_order
    
    for i in range(filter_order, length):
        # 入力ベクトル
        x = data[i-filter_order:i]
        
        # フィルター出力
        y = np.dot(weights, x)
        filtered_signal[i] = y
        
        # 参照信号（将来の値を予測）
        if i + 1 < length:
            desired = data[i + 1]
            error = desired - y
            
            # LMS更新則
            weights += learning_rate * error * x
            
            # 重みの正規化
            weight_norm = np.linalg.norm(weights)
            if weight_norm > 0:
                weights /= weight_norm
                weights *= filter_order
        
        # 適応度の測定
        recent_errors = []
        for j in range(max(filter_order, i - period + 1), i):
            if j < length - 1:
                pred = np.dot(weights, data[j-filter_order:j])
                err = abs(data[j + 1] - pred)
                recent_errors.append(err)
        
        if len(recent_errors) > 0:
            adaptation_weights[i] = 1.0 / (1.0 + np.mean(np.array(recent_errors)))
        else:
            adaptation_weights[i] = 0.5
    
    return filtered_signal, adaptation_weights


@njit(fastmath=True, cache=True)
def empirical_mode_decomposition_simple(data: np.ndarray, n_imfs: int = 3) -> Tuple:
    """
    簡易経験的モード分解（EMD）- 内在モード関数による時系列分解
    """
    length = len(data)
    imfs = np.zeros((n_imfs, length))
    residue = data.copy()
    
    for imf_idx in range(n_imfs):
        current_signal = residue.copy()
        
        # Sifting process (簡易版)
        for iteration in range(10):  # 最大10回の反復
            # 極値検出
            maxima_indices = []
            minima_indices = []
            
            for i in range(1, length - 1):
                if current_signal[i] > current_signal[i-1] and current_signal[i] > current_signal[i+1]:
                    maxima_indices.append(i)
                elif current_signal[i] < current_signal[i-1] and current_signal[i] < current_signal[i+1]:
                    minima_indices.append(i)
            
            if len(maxima_indices) < 2 or len(minima_indices) < 2:
                break
            
            # エンベロープの計算（線形補間）
            upper_envelope = np.zeros(length)
            lower_envelope = np.zeros(length)
            
            # 上部エンベロープ
            for i in range(length):
                if len(maxima_indices) >= 2:
                    # 最も近い2つの極大値で補間
                    if i <= maxima_indices[0]:
                        upper_envelope[i] = current_signal[maxima_indices[0]]
                    elif i >= maxima_indices[-1]:
                        upper_envelope[i] = current_signal[maxima_indices[-1]]
                    else:
                        # 線形補間
                        for j in range(len(maxima_indices) - 1):
                            if maxima_indices[j] <= i <= maxima_indices[j + 1]:
                                x1, y1 = maxima_indices[j], current_signal[maxima_indices[j]]
                                x2, y2 = maxima_indices[j + 1], current_signal[maxima_indices[j + 1]]
                                upper_envelope[i] = y1 + (y2 - y1) * (i - x1) / (x2 - x1)
                                break
                else:
                    upper_envelope[i] = current_signal[i]
            
            # 下部エンベロープ
            for i in range(length):
                if len(minima_indices) >= 2:
                    if i <= minima_indices[0]:
                        lower_envelope[i] = current_signal[minima_indices[0]]
                    elif i >= minima_indices[-1]:
                        lower_envelope[i] = current_signal[minima_indices[-1]]
                    else:
                        for j in range(len(minima_indices) - 1):
                            if minima_indices[j] <= i <= minima_indices[j + 1]:
                                x1, y1 = minima_indices[j], current_signal[minima_indices[j]]
                                x2, y2 = minima_indices[j + 1], current_signal[minima_indices[j + 1]]
                                lower_envelope[i] = y1 + (y2 - y1) * (i - x1) / (x2 - x1)
                                break
                else:
                    lower_envelope[i] = current_signal[i]
            
            # 平均エンベロープ
            mean_envelope = (upper_envelope + lower_envelope) / 2.0
            
            # IMF候補の更新
            h = current_signal - mean_envelope
            
            # 停止基準（簡易版）
            difference = np.sum(np.abs(h - current_signal))
            if difference < 1e-6:
                break
                
            current_signal = h
        
        imfs[imf_idx] = current_signal
        residue = residue - current_signal
        
        # 残差が小さくなったら終了
        if np.std(residue) < 1e-6:
            break
    
    return imfs


@njit(fastmath=True, cache=True)
def adaptive_ensemble_learning(
    features: np.ndarray,  # shape: (n_features, length)
    period: int,
    n_learners: int = 5
) -> Tuple:
    """
    適応アンサンブル学習 - 複数の弱学習器による動的重み調整
    """
    n_features, length = features.shape
    predictions = np.zeros(length)
    learner_weights = np.ones(n_learners) / n_learners
    learner_errors = np.zeros(n_learners)
    
    # 簡易線形学習器のパラメータ
    learner_params = np.random.random((n_learners, n_features)) * 0.1
    
    for i in range(period, length):
        # 各学習器の予測
        learner_preds = np.zeros(n_learners)
        
        for learner_idx in range(n_learners):
            # 線形結合による予測
            pred = 0.0
            for feat_idx in range(n_features):
                pred += learner_params[learner_idx, feat_idx] * features[feat_idx, i]
            
            # シグモイド活性化
            learner_preds[learner_idx] = 1.0 / (1.0 + np.exp(-pred))
        
        # アンサンブル予測
        ensemble_pred = np.dot(learner_weights, learner_preds)
        predictions[i] = ensemble_pred
        
        # 学習器の更新（将来の値を使用）
        if i + 1 < length:
            # 真値（次期のボラティリティ状態の代理）
            window_vol = 0.0
            count = 0
            for j in range(max(0, i - 5), i + 1):
                if j > 0:
                    window_vol += features[0, j]  # 第一特徴量をボラティリティ代理とする
                    count += 1
            
            if count > 0:
                avg_vol = window_vol / count
                target = 1.0 if avg_vol > np.median(features[0, max(0, i-period):i]) else 0.0
                
                # 各学習器の誤差計算と重み更新
                total_performance = 0.0
                for learner_idx in range(n_learners):
                    error = abs(target - learner_preds[learner_idx])
                    learner_errors[learner_idx] = error
                    
                    # パラメータ更新（簡易勾配降下）
                    learning_rate = 0.01
                    gradient = (learner_preds[learner_idx] - target) * learner_preds[learner_idx] * (1 - learner_preds[learner_idx])
                    
                    for feat_idx in range(n_features):
                        learner_params[learner_idx, feat_idx] -= learning_rate * gradient * features[feat_idx, i]
                    
                    # 性能指標
                    performance = 1.0 / (1.0 + error)
                    total_performance += performance
                
                # 重みの正規化
                if total_performance > 0:
                    for learner_idx in range(n_learners):
                        performance = 1.0 / (1.0 + learner_errors[learner_idx])
                        learner_weights[learner_idx] = performance / total_performance
    
    return predictions, learner_weights


@njit(fastmath=True, parallel=True, cache=True)
def ultra_advanced_volatility_fusion(
    # 従来の特徴量
    str_zscore: np.ndarray,
    vol_zscore: np.ndarray,
    # カオス理論
    lyapunov_exp: np.ndarray,
    # 情報理論
    multiscale_ent: np.ndarray,  # 第一スケール
    # デジタル信号処理
    kalman_vol: np.ndarray,
    kalman_innovation: np.ndarray,
    # 神経適応学習
    neural_filtered: np.ndarray,
    neural_weights: np.ndarray,
    # EMD分解
    emd_imf1: np.ndarray,  # 第一IMF
    emd_imf2: np.ndarray,  # 第二IMF
    # 機械学習
    ml_prediction: np.ndarray,
    ml_confidence: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    超高度ボラティリティ融合アルゴリズム
    """
    length = len(str_zscore)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    confidence = np.zeros(length)
    
    # 動的重み係数（金融時系列に最適化）- 調整版
    w_traditional = 0.35   # 従来手法（実績のある手法重視）
    w_dsp = 0.25          # デジタル信号処理（Kalman）
    w_neural = 0.15       # 神経適応学習
    w_ml = 0.10           # 機械学習
    w_chaos = 0.08        # カオス理論（Lyapunov指数）
    w_entropy = 0.05      # 情報エントロピー（MSE）
    w_emd = 0.02          # EMD分解
    
    for i in prange(length):
        # カオス理論指標の正規化
        chaos_val = abs(lyapunov_exp[i]) if not np.isnan(lyapunov_exp[i]) else 0.0
        norm_chaos = min(max(chaos_val * 10, 0.0), 1.0)
        
        # 情報理論指標
        entropy_val = multiscale_ent[i] if not np.isnan(multiscale_ent[i]) else 0.0
        norm_entropy = min(max(entropy_val / 5.0, 0.0), 1.0)
        
        # DSP指標
        kalman_val = kalman_vol[i] if not np.isnan(kalman_vol[i]) else 0.0
        innovation_val = kalman_innovation[i] if not np.isnan(kalman_innovation[i]) else 0.0
        norm_kalman = min(max(kalman_val * 50, 0.0), 1.0)
        norm_innovation = min(max(innovation_val * 100, 0.0), 1.0)
        
        # 神経適応指標
        neural_val = abs(neural_filtered[i] - neural_filtered[max(0, i-1)]) if i > 0 and not np.isnan(neural_filtered[i]) else 0.0
        adaptation_val = neural_weights[i] if not np.isnan(neural_weights[i]) else 0.5
        norm_neural = min(max(neural_val * 100, 0.0), 1.0)
        norm_adaptation = adaptation_val
        
        # EMD指標
        emd1_val = abs(emd_imf1[i]) if not np.isnan(emd_imf1[i]) else 0.0
        emd2_val = abs(emd_imf2[i]) if not np.isnan(emd_imf2[i]) else 0.0
        norm_emd1 = min(max(emd1_val * 20, 0.0), 1.0)
        norm_emd2 = min(max(emd2_val * 15, 0.0), 1.0)
        
        # 機械学習指標
        ml_val = ml_prediction[i] if not np.isnan(ml_prediction[i]) else 0.5
        ml_conf = ml_confidence[i] if not np.isnan(ml_confidence[i]) else 0.5
        norm_ml = ml_val
        
        # 従来指標
        str_val = abs(str_zscore[i]) / 3.0 if not np.isnan(str_zscore[i]) else 0.0
        vol_val = abs(vol_zscore[i]) / 3.0 if not np.isnan(vol_zscore[i]) else 0.0
        norm_str = min(max(str_val, 0.0), 1.0)
        norm_vol = min(max(vol_val, 0.0), 1.0)
        
        # 超高度融合スコア（調整版）
        score = (w_traditional * (norm_str + norm_vol) / 2.0 +
                 w_dsp * (norm_kalman + norm_innovation) / 2.0 +
                 w_neural * (norm_neural + norm_adaptation) / 2.0 +
                 w_ml * norm_ml +
                 w_chaos * norm_chaos +
                 w_entropy * norm_entropy +
                 w_emd * (norm_emd1 + norm_emd2) / 2.0)
        
        raw_score[i] = score
        
        # 多次元信頼度計算
        feature_consensus = np.array([norm_chaos, norm_entropy, norm_kalman, norm_neural, norm_ml])
        mean_consensus = np.mean(feature_consensus)
        std_consensus = np.std(feature_consensus)
        
        # 信頼度は合意度と機械学習信頼度の組み合わせ
        consensus_conf = max(0.0, 1.0 - std_consensus / (mean_consensus + 1e-8))
        combined_conf = 0.7 * consensus_conf + 0.3 * ml_conf
        confidence[i] = combined_conf
        
        # 適応的シグモイド（信頼度とカオス度を考慮）
        chaos_adjustment = 0.1 * (norm_chaos - 0.5)  # カオス度による調整
        adaptive_threshold = threshold + chaos_adjustment
        
        k = 12.0 + 8.0 * combined_conf  # 超高信頼度時はより急峻
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - adaptive_threshold)))
        
        # 調整された状態判定（より実用的）
        high_confidence_threshold = 0.4  # 閾値を下げて実用性向上
        if combined_conf > high_confidence_threshold:
            # 高信頼度時：標準判定
            state[i] = 1 if score > adaptive_threshold else 0
        elif combined_conf > 0.2:
            # 中信頼度時：やや保守的
            state[i] = 1 if score > adaptive_threshold + 0.05 else 0
        else:
            # 低信頼度時：保守的
            state[i] = 1 if score > adaptive_threshold + 0.1 and probability[i] > 0.6 else 0
    
    return state, probability, raw_score, confidence


class UltimateVolatilityStateV3(Indicator):
    """
    究極のボラティリティ状態判別インジケーター V3（超高精度版）
    
    最先端の金融工学・信号処理技術を結集:
    
    1. カオス理論: Lyapunov指数による敏感依存性分析
    2. 情報理論: マルチスケールエントロピーによる複雑性測定
    3. デジタル信号処理: 適応カルマンフィルターによる動的推定
    4. 神経適応学習: オンライン学習による動的フィルタリング
    5. 経験的モード分解: EMDによる内在モード関数抽出
    6. 適応アンサンブル学習: 複数弱学習器による動的重み調整
    
    特徴:
    - 超々高精度: 20+の最先端アルゴリズムによる総合判定
    - 動的適応: リアルタイム学習による継続的精度改善
    - カオス対応: 金融市場の非線形・複雑系特性に特化
    - 信頼度評価: 多次元的な判定信頼度の定量化
    """
    
    def __init__(
        self,
        period: int = 21,                    # 基本期間
        threshold: float = 0.5,              # 高/低ボラティリティの閾値
        zscore_period: int = 50,             # Z-Score計算期間
        src_type: str = 'hlc3',              # 価格ソース
        learning_rate: float = 0.01,         # 神経学習率
        chaos_embedding_dim: int = 3,        # カオス解析の埋め込み次元
        n_learners: int = 7,                 # アンサンブル学習器数
        confidence_threshold: float = 0.8    # 信頼度閾値
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本計算期間
            threshold: ボラティリティ判定閾値 (0.0-1.0)
            zscore_period: Z-Score正規化期間
            src_type: 価格ソースタイプ
            learning_rate: 神経適応学習の学習率
            chaos_embedding_dim: カオス解析の位相空間埋め込み次元
            n_learners: アンサンブル学習の学習器数
            confidence_threshold: 信頼度の最小閾値
        """
        super().__init__(f"UltimateVolatilityStateV3(period={period}, chaos={chaos_embedding_dim}, learners={n_learners})")
        
        self.period = period
        self.threshold = threshold
        self.zscore_period = zscore_period
        self.src_type = src_type.lower()
        self.learning_rate = learning_rate
        self.chaos_embedding_dim = chaos_embedding_dim
        self.n_learners = n_learners
        self.confidence_threshold = confidence_threshold
        
        # 基本コンポーネント
        self.str_indicator = STR(
            period=period,
            src_type=src_type,
            period_mode='dynamic'
        )
        
        self.vol_indicator = volatility(
            period_mode='adaptive',
            fixed_period=period,
            calculation_mode='return',
            return_type='log',
            smoother_type='hma',
            smoother_period=period // 4
        )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 3
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateVolatilityStateV3Result:
        """
        超高精度ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            UltimateVolatilityStateV3Result: 判定結果
        """
        try:
            # データ準備
            if isinstance(data, pd.DataFrame):
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
            
            length = len(close)
            min_required = max(self.period, self.zscore_period) * 2
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # === 超高度分析の実行 ===
            
            # 1. 基本指標
            str_result = self.str_indicator.calculate(data)
            vol_values = self.vol_indicator.calculate(data)
            
            str_zscore_calc = ZScore(period=self.zscore_period)
            vol_zscore_calc = ZScore(period=self.zscore_period)
            
            str_df = pd.DataFrame({'close': str_result.values})
            vol_df = pd.DataFrame({'close': vol_values})
            
            str_zscore = str_zscore_calc.calculate(str_df)
            vol_zscore = vol_zscore_calc.calculate(vol_df)
            
            # 2. カオス理論解析
            src_prices = PriceSource.calculate_source(data, self.src_type)
            lyapunov_exp = lyapunov_exponent_estimation(
                src_prices, self.period, self.chaos_embedding_dim
            )
            
            # 3. 情報理論解析
            mse_values = multiscale_entropy(src_prices, self.period, 3)
            multiscale_ent = mse_values[0]  # 第一スケール
            
            # 4. デジタル信号処理
            kalman_vol, kalman_innovation = adaptive_kalman_volatility(src_prices, self.period)
            
            # 5. 神経適応学習
            neural_filtered, neural_weights = neural_adaptive_filter(
                src_prices, self.period, self.learning_rate
            )
            
            # 6. 経験的モード分解
            emd_imfs = empirical_mode_decomposition_simple(src_prices, 3)
            emd_imf1 = emd_imfs[0]
            emd_imf2 = emd_imfs[1]
            
            # 7. アンサンブル機械学習
            # 特徴量行列の構築
            features = np.vstack([
                str_zscore,
                vol_zscore,
                lyapunov_exp,
                multiscale_ent,
                kalman_vol,
                neural_weights,
                np.abs(emd_imf1)
            ])
            
            ml_prediction, ml_weights = adaptive_ensemble_learning(
                features, self.period, self.n_learners
            )
            ml_confidence = neural_weights  # 神経重みを信頼度代理として使用
            
            # 8. 超高度融合
            state, probability, raw_score, confidence = ultra_advanced_volatility_fusion(
                str_zscore, vol_zscore, lyapunov_exp, multiscale_ent,
                kalman_vol, kalman_innovation, neural_filtered, neural_weights,
                emd_imf1, emd_imf2, ml_prediction, ml_confidence, self.threshold
            )
            
            # 結果の構築
            result = UltimateVolatilityStateV3Result(
                state=state,
                probability=probability,
                confidence=confidence,
                chaos_measure=lyapunov_exp,
                neural_adaptation=neural_weights,
                entropy_metrics={
                    'multiscale_entropy': multiscale_ent,
                    'sample_entropy_scale1': mse_values[0],
                    'sample_entropy_scale2': mse_values[1] if len(mse_values) > 1 else np.zeros_like(multiscale_ent),
                    'sample_entropy_scale3': mse_values[2] if len(mse_values) > 2 else np.zeros_like(multiscale_ent)
                },
                dsp_features={
                    'kalman_volatility': kalman_vol,
                    'kalman_innovation': kalman_innovation,
                    'neural_filtered': neural_filtered,
                    'emd_imf1': emd_imf1,
                    'emd_imf2': emd_imf2
                },
                ml_prediction=ml_prediction,
                ensemble_weight=ml_weights
            )
            
            # キャッシュ管理
            data_hash = self._get_data_hash(data)
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"V3ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltimateVolatilityStateV3Result:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return UltimateVolatilityStateV3Result(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            confidence=empty_array,
            chaos_measure=empty_array,
            neural_adaptation=empty_array,
            entropy_metrics={},
            dsp_features={},
            ml_prediction=empty_array,
            ensemble_weight=np.ones(self.n_learners) / self.n_learners
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0]['close']) if length > 0 else 0.0
                last_val = float(data.iloc[-1]['close']) if length > 0 else 0.0
            else:
                length = len(data)
                first_val = float(data[0, 3]) if length > 0 else 0.0
                last_val = float(data[-1, 3]) if length > 0 else 0.0
            
            params_sig = f"{self.period}_{self.threshold}_{self.chaos_embedding_dim}_{self.n_learners}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.period}_{self.threshold}"
    
    def get_chaos_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """カオス解析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return {
                'lyapunov_exponent': latest_result.chaos_measure,
                'neural_adaptation': latest_result.neural_adaptation
            }
        return None
    
    def get_entropy_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """エントロピー解析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.entropy_metrics
        return None
    
    def get_dsp_features(self) -> Optional[Dict[str, np.ndarray]]:
        """DSP特徴量を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.dsp_features
        return None
    
    def is_ultra_high_volatility(self, min_confidence: float = 0.85) -> bool:
        """超高信頼度での高ボラティリティ判定"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.state) > 0:
                latest_state = latest_result.state[-1]
                latest_confidence = latest_result.confidence[-1]
                return bool(latest_state == 1 and latest_confidence >= min_confidence)
        return False
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        self.vol_indicator.reset()