#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from numba import jit, prange, float64, int32
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@jit(nopython=True)
def robust_entropy(data: np.ndarray, window: int = 20) -> float:
    """
    ロバストエントロピー計算（外れ値に対して安定）
    """
    if len(data) < window:
        return 0.5
    
    recent_data = data[-window:]
    
    # 外れ値除去（IQR方式）
    q1 = np.percentile(recent_data, 25)
    q3 = np.percentile(recent_data, 75)
    iqr = q3 - q1
    
    if iqr > 0:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = recent_data[(recent_data >= lower_bound) & (recent_data <= upper_bound)]
    else:
        filtered_data = recent_data
    
    if len(filtered_data) < 3:
        return 0.5
    
    # 正規化
    data_range = np.max(filtered_data) - np.min(filtered_data)
    if data_range == 0:
        return 0.0
    
    normalized = (filtered_data - np.min(filtered_data)) / data_range
    
    # ヒストグラム計算
    hist, _ = np.histogram(normalized, bins=8, range=(0, 1))
    hist = hist.astype(np.float64)
    total = np.sum(hist)
    
    if total == 0:
        return 0.0
    
    prob_dist = hist / total
    
    # シャノンエントロピー
    entropy = 0.0
    for p in prob_dist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy / np.log2(8)


@jit(nopython=True)
def enhanced_kalman_smoother(
    observations: np.ndarray,
    process_noise: float = 0.001,
    observation_noise: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    強化版Kalmanスムーザー（前方-後方パス）
    """
    n = len(observations)
    
    # 前方パス
    states_forward = np.zeros(n)
    covariances_forward = np.zeros(n)
    
    # 初期状態
    state = observations[0] if n > 0 else 20.0
    covariance = 1.0
    
    for i in range(n):
        # 予測
        state_pred = state
        cov_pred = covariance + process_noise
        
        # 適応的な観測ノイズ
        innovation = observations[i] - state_pred
        adaptive_obs_noise = observation_noise * (1 + 0.5 * abs(innovation) / (abs(state_pred) + 1e-10))
        
        # 更新
        innovation_cov = cov_pred + adaptive_obs_noise
        kalman_gain = cov_pred / innovation_cov if innovation_cov > 0 else 0
        
        state = state_pred + kalman_gain * innovation
        covariance = (1 - kalman_gain) * cov_pred
        
        states_forward[i] = state
        covariances_forward[i] = covariance
    
    # 後方パス（スムージング）
    smoothed_states = np.zeros(n)
    smoothed_states[-1] = states_forward[-1]
    
    for i in range(n-2, -1, -1):
        if i < n-1:
            A = covariances_forward[i] / (covariances_forward[i] + process_noise)
            smoothed_states[i] = states_forward[i] + A * (smoothed_states[i+1] - states_forward[i])
        else:
            smoothed_states[i] = states_forward[i]
    
    # 制約適用
    for i in range(n):
        smoothed_states[i] = max(6.0, min(50.0, smoothed_states[i]))
    
    return smoothed_states, covariances_forward


@jit(nopython=True)
def stable_hilbert_cycle(
    price: np.ndarray,
    period_range: Tuple[int, int] = (6, 50)
) -> np.ndarray:
    """
    安定化Hilbert変換によるサイクル検出
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    period = np.zeros(n)
    
    # 安定性パラメータ
    alpha = 0.07  # より保守的な値
    
    for i in range(n):
        if i < 6:
            smooth[i] = price[i]
            period[i] = 20.0
            continue
        
        # 安定化スムージング
        smooth[i] = (4 * price[i] + 3 * price[i-1] + 2 * price[i-2] + price[i-3]) / 10
        
        # 安定化デトレンド
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)
        
        # Hilbert変換近似
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
        i1[i] = detrender[i-3] if i >= 3 else 0
        
        # 安定化フィルタ
        if i >= 1:
            i1[i] = alpha * i1[i] + (1 - alpha) * i1[i-1]
            q1[i] = alpha * q1[i] + (1 - alpha) * q1[i-1]
        
        # 安定化期間計算
        if i >= 1 and (i1[i] != 0 or q1[i] != 0):
            re = i1[i] * i1[i-1] + q1[i] * q1[i-1]
            im = i1[i] * q1[i-1] - q1[i] * i1[i-1]
            
            if abs(im) > 1e-10 and abs(re) > 1e-10:
                raw_period = 2 * pi / abs(np.arctan(im / re))
                
                # より厳しい制限
                max_change = 0.2  # 20%の変化まで許可
                if raw_period > period[i-1] * (1 + max_change):
                    raw_period = period[i-1] * (1 + max_change)
                elif raw_period < period[i-1] * (1 - max_change):
                    raw_period = period[i-1] * (1 - max_change)
                
                raw_period = max(period_range[0], min(period_range[1], raw_period))
                
                # より強い平滑化
                period[i] = 0.1 * raw_period + 0.9 * period[i-1]
            else:
                period[i] = period[i-1]
        else:
            period[i] = period[i-1] if i >= 1 else 20.0
    
    return period


@jit(nopython=True)
def adaptive_autocorrelation_cycle(
    price: np.ndarray,
    max_period: int = 50,
    min_period: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応的自己相関によるサイクル検出
    """
    n = len(price)
    periods = np.zeros(n)
    confidences = np.zeros(n)
    
    for i in range(max_period, n):
        window_size = min(100, i)  # 長期窓で安定性向上
        recent_data = price[i-window_size:i+1]
        
        max_corr = 0.0
        best_period = 20.0
        
        # 自己相関計算
        for period in range(min_period, min(max_period, window_size//3)):
            if len(recent_data) >= 2 * period:
                # より長い区間での相関計算
                data1 = recent_data[:-period]
                data2 = recent_data[period:]
                
                if len(data1) > 10 and len(data2) > 10:
                    # 相関係数計算
                    mean1 = np.mean(data1)
                    mean2 = np.mean(data2)
                    
                    num = np.sum((data1 - mean1) * (data2 - mean2))
                    den1 = np.sqrt(np.sum((data1 - mean1)**2))
                    den2 = np.sqrt(np.sum((data2 - mean2)**2))
                    
                    if den1 > 0 and den2 > 0:
                        corr = num / (den1 * den2)
                        
                        if abs(corr) > max_corr:
                            max_corr = abs(corr)
                            best_period = float(period)
        
        periods[i] = best_period
        confidences[i] = max_corr
    
    # 初期値設定
    for i in range(max_period):
        periods[i] = 20.0
        confidences[i] = 0.5
    
    return periods, confidences


@jit(nopython=True)
def multi_method_consensus(
    hilbert_periods: np.ndarray,
    autocorr_periods: np.ndarray,
    autocorr_confidences: np.ndarray,
    noise_levels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    複数手法の合意形成による最終期間決定
    """
    n = len(hilbert_periods)
    final_periods = np.zeros(n)
    consensus_scores = np.zeros(n)
    
    for i in range(n):
        # 手法間の一致度計算
        period_diff = abs(hilbert_periods[i] - autocorr_periods[i])
        max_period = max(hilbert_periods[i], autocorr_periods[i])
        
        if max_period > 0:
            agreement = 1.0 - (period_diff / max_period)
        else:
            agreement = 0.0
        
        # ノイズレベルによる調整
        noise_factor = 1.0 - noise_levels[i]
        
        # 信頼度による重み計算
        hilbert_weight = 0.7 * noise_factor  # Hilbertは一般的に安定
        autocorr_weight = autocorr_confidences[i] * (1 - noise_factor)
        
        total_weight = hilbert_weight + autocorr_weight
        if total_weight > 0:
            hilbert_weight /= total_weight
            autocorr_weight /= total_weight
        else:
            hilbert_weight = 0.7
            autocorr_weight = 0.3
        
        # 重み付き平均
        final_periods[i] = (hilbert_weight * hilbert_periods[i] + 
                           autocorr_weight * autocorr_periods[i])
        
        # 合意スコア
        consensus_scores[i] = agreement * min(1.0, hilbert_weight + autocorr_weight)
    
    return final_periods, consensus_scores


@jit(nopython=True)
def calculate_ultimate_cycle_numba(
    price: np.ndarray,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1,
    entropy_window: int = 30,
    period_range: Tuple[int, int] = (6, 50)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    究極のサイクル検出アルゴリズム
    """
    n = len(price)
    
    # 1. ノイズレベル評価
    noise_levels = np.zeros(n)
    for i in range(entropy_window, n):
        noise_levels[i] = robust_entropy(price[max(0, i-entropy_window):i+1], entropy_window)
    
    # 初期値設定
    for i in range(entropy_window):
        noise_levels[i] = 0.5
    
    # 2. 安定化Hilbert変換
    hilbert_periods = stable_hilbert_cycle(price, period_range)
    
    # 3. 適応的自己相関
    autocorr_periods, autocorr_confidences = adaptive_autocorrelation_cycle(
        price, period_range[1], period_range[0]
    )
    
    # 4. 複数手法の合意形成
    consensus_periods, consensus_scores = multi_method_consensus(
        hilbert_periods, autocorr_periods, autocorr_confidences, noise_levels
    )
    
    # 5. 強化版Kalmanスムージング
    final_periods, uncertainties = enhanced_kalman_smoother(
        consensus_periods, process_noise=0.001, observation_noise=0.01
    )
    
    # 6. 最終サイクル値計算
    dom_cycle = np.zeros(n)
    for i in range(n):
        cycle_value = np.ceil(final_periods[i] * cycle_part)
        dom_cycle[i] = max(min_output, min(max_output, cycle_value))
    
    return dom_cycle, final_periods, consensus_scores, noise_levels


class EhlersUltimateCycle(EhlersDominantCycle):
    """
    究極のサイクル検出器 - 最高の安定性を目指した統合アプローチ
    
    🎯 **設計思想:**
    従来手法の安定性を超えることを最優先に、複数の革新的技術を慎重に統合
    
    🏆 **主要な特徴:**
    1. **ロバストエントロピー**: 外れ値に強いノイズ評価
    2. **安定化Hilbert変換**: 急激な変化を抑制した期間検出
    3. **適応的自己相関**: 長期窓による信頼性の高い周期検出
    4. **多手法合意形成**: 複数アルゴリズムの慎重な統合
    5. **強化版Kalmanスムージング**: 前方-後方パスによる最適平滑化
    
    🎨 **革新ポイント:**
    - 安定性を最優先とした保守的なパラメータ設定
    - 外れ値に対するロバスト処理
    - 段階的な信頼度評価システム
    - 長期的な一貫性を重視した設計
    
    💪 **期待される効果:**
    - 従来手法を超える安定性
    - 市場ショックに対する高い耐性
    - 一貫したサイクル検出
    - 極めて低いノイズレベル
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        entropy_window: int = 30,
        period_range: Tuple[int, int] = (6, 50),
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            entropy_window: エントロピー計算のウィンドウサイズ（デフォルト: 30）
            period_range: サイクル期間の範囲（デフォルト: (6, 50)）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        super().__init__(
            f"EhlersUltimate({cycle_part}, {period_range})",
            cycle_part,
            period_range[1],  # max_cycle
            period_range[0],  # min_cycle
            max_output,
            min_output
        )
        
        self.entropy_window = entropy_window
        self.period_range = period_range
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 追加の結果保存用
        self._consensus_scores = None
        self._noise_levels = None
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
        """
        指定されたソースタイプに基づいて価格データを計算する
        """
        if isinstance(data, pd.DataFrame):
            if src_type == 'close':
                if 'close' in data.columns:
                    return data['close'].values
                elif 'Close' in data.columns:
                    return data['Close'].values
                else:
                    raise ValueError("DataFrameには'close'または'Close'カラムが必要です")
            
            elif src_type == 'hlc3':
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    return (data['high'] + data['low'] + data['close']).values / 3
                elif all(col in data.columns for col in ['High', 'Low', 'Close']):
                    return (data['High'] + data['Low'] + data['Close']).values / 3
                else:
                    raise ValueError("hlc3の計算には'high', 'low', 'close'カラムが必要です")
            
            elif src_type == 'hl2':
                if all(col in data.columns for col in ['high', 'low']):
                    return (data['high'] + data['low']).values / 2
                elif all(col in data.columns for col in ['High', 'Low']):
                    return (data['High'] + data['Low']).values / 2
                else:
                    raise ValueError("hl2の計算には'high', 'low'カラムが必要です")
            
            elif src_type == 'ohlc4':
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    return (data['open'] + data['high'] + data['low'] + data['close']).values / 4
                elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    return (data['Open'] + data['High'] + data['Low'] + data['Close']).values / 4
                else:
                    raise ValueError("ohlc4の計算には'open', 'high', 'low', 'close'カラムが必要です")
        
        else:  # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                if src_type == 'close':
                    return data[:, 3]  # close
                elif src_type == 'hlc3':
                    return (data[:, 1] + data[:, 2] + data[:, 3]) / 3  # high, low, close
                elif src_type == 'hl2':
                    return (data[:, 1] + data[:, 2]) / 2  # high, low
                elif src_type == 'ohlc4':
                    return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4  # open, high, low, close
            else:
                return data  # 1次元配列として扱う
        
        return data
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        究極のアルゴリズムを使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash
            
            # ソースタイプに基づいて価格データを取得
            price = self.calculate_source_values(data, self.src_type)
            
            # Numba関数を使用してドミナントサイクルを計算
            dom_cycle, raw_period, consensus_scores, noise_levels = calculate_ultimate_cycle_numba(
                price,
                self.cycle_part,
                self.max_output,
                self.min_output,
                self.entropy_window,
                self.period_range
            )
            
            # 結果を保存
            self._result = DominantCycleResult(
                values=dom_cycle,
                raw_period=raw_period,
                smooth_period=raw_period  # この実装では同じ
            )
            
            # 追加メタデータを保存
            self._consensus_scores = consensus_scores
            self._noise_levels = noise_levels
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersUltimateCycle計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    @property
    def consensus_scores(self) -> Optional[np.ndarray]:
        """合意スコアを取得"""
        return self._consensus_scores
    
    @property
    def noise_levels(self) -> Optional[np.ndarray]:
        """ノイズレベルを取得"""
        return self._noise_levels
    
    def get_analysis_summary(self) -> Dict:
        """
        分析サマリーを取得
        """
        if self._result is None:
            return {}
        
        summary = {
            'algorithm': 'Ultimate Cycle Detector',
            'methods_used': [
                'Robust Entropy Assessment',
                'Stabilized Hilbert Transform',
                'Adaptive Autocorrelation',
                'Multi-Method Consensus',
                'Enhanced Kalman Smoother'
            ],
            'cycle_range': self.period_range,
            'avg_consensus': np.mean(self._consensus_scores) if self._consensus_scores is not None else None,
            'avg_noise_level': np.mean(self._noise_levels) if self._noise_levels is not None else None,
            'dominant_cycle_stats': {
                'mean': np.mean(self._result.values),
                'std': np.std(self._result.values),
                'min': np.min(self._result.values),
                'max': np.max(self._result.values)
            },
            'stability_features': [
                'Conservative parameter settings for maximum stability',
                'Robust outlier handling in entropy calculation',
                'Gradual change constraints in Hilbert transform',
                'Long-term window autocorrelation analysis',
                'Forward-backward Kalman smoothing'
            ]
        }
        
        return summary 