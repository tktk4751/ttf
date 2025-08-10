#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from numba import jit
try:
    import scipy.signal as signal
except ImportError:
    signal = None
import warnings
warnings.filterwarnings('ignore')

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
try:
    from indicators.kalman.unified_kalman import UnifiedKalman
except ImportError:
    try:
        from ..kalman.unified_kalman import UnifiedKalman
    except ImportError:
        UnifiedKalman = None


@jit(nopython=True)
def calculate_entropy(data: np.ndarray, window: int = 20) -> float:
    """
    情報エントロピーを計算してノイズレベルを評価
    """
    if len(data) < window:
        return 0.5
    
    # データを正規化
    recent_data = data[-window:]
    data_range = np.max(recent_data) - np.min(recent_data)
    if data_range == 0:
        return 0.0
    
    normalized = (recent_data - np.min(recent_data)) / data_range
    
    # ヒストグラムベースでエントロピー計算
    hist, _ = np.histogram(normalized, bins=10, range=(0, 1))
    hist = hist.astype(np.float64)
    hist = hist / np.sum(hist)
    
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy / np.log2(10)  # 正規化


@jit(nopython=True)
def adaptive_homodyne_discriminator(
    price: np.ndarray,
    period_range: Tuple[int, int] = (6, 50),
    noise_factor: float = 1.0
) -> np.ndarray:
    """
    適応型ホモダイン判別器
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    period = np.zeros(n)
    
    # ノイズ適応パラメータ
    smooth_alpha = 0.2 * noise_factor
    trend_alpha = 0.075 + 0.025 * (1 - noise_factor)
    
    for i in range(n):
        if i < 6:
            smooth[i] = price[i]
            period[i] = 20.0
            continue
        
        # 適応型スムージング
        weights = np.array([4.0, 3.0, 2.0, 1.0]) * noise_factor
        smooth[i] = (weights[0] * price[i] + 
                    weights[1] * price[i-1] + 
                    weights[2] * price[i-2] + 
                    weights[3] * price[i-3]) / np.sum(weights)
        
        # 適応型デトレンド
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (trend_alpha * period[i-1] + 0.54)
        
        # 直交コンポーネント
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (trend_alpha * period[i-1] + 0.54)
        i1[i] = detrender[i-3] if i >= 3 else 0
        
        # 適応型スムージング
        if i >= 1:
            i1[i] = smooth_alpha * i1[i] + (1 - smooth_alpha) * i1[i-1]
            q1[i] = smooth_alpha * q1[i] + (1 - smooth_alpha) * q1[i-1]
        
        # ホモダイン判別
        if i >= 1 and (i1[i] != 0 or q1[i] != 0):
            re = i1[i] * i1[i-1] + q1[i] * q1[i-1]
            im = i1[i] * q1[i-1] - q1[i] * i1[i-1]
            
            if im != 0 and re != 0:
                raw_period = 2 * pi / abs(np.arctan(im / re))
                
                # 適応型制限
                if raw_period > 1.5 * period[i-1]:
                    raw_period = 1.5 * period[i-1]
                elif raw_period < 0.67 * period[i-1]:
                    raw_period = 0.67 * period[i-1]
                
                raw_period = max(period_range[0], min(period_range[1], raw_period))
                period[i] = smooth_alpha * raw_period + (1 - smooth_alpha) * period[i-1]
            else:
                period[i] = period[i-1]
        else:
            period[i] = period[i-1] if i >= 1 else 20.0
    
    return period


@jit(nopython=True)
def adaptive_phase_accumulator(
    price: np.ndarray,
    period_range: Tuple[int, int] = (6, 50),
    noise_factor: float = 1.0
) -> np.ndarray:
    """
    適応型位相累積アルゴリズム
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    phase = np.zeros(n)
    period = np.zeros(n)
    
    # ノイズ適応パラメータ
    smooth_alpha = 0.15 * (1 + noise_factor * 0.5)
    
    for i in range(n):
        if i < 6:
            smooth[i] = price[i]
            period[i] = 20.0
            continue
        
        # スムージング
        smooth[i] = (4 * price[i] + 3 * price[i-1] + 2 * price[i-2] + price[i-3]) / 10
        
        # デトレンド
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)
        
        # 直交コンポーネント
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
        i1[i] = detrender[i-3] if i >= 3 else 0
        
        # スムージング
        if i >= 1:
            i1[i] = smooth_alpha * i1[i] + (1 - smooth_alpha) * i1[i-1]
            q1[i] = smooth_alpha * q1[i] + (1 - smooth_alpha) * q1[i-1]
        
        # 位相計算
        if i1[i] != 0:
            phase[i] = np.arctan(abs(q1[i] / i1[i]))
        else:
            phase[i] = phase[i-1] if i >= 1 else 0
        
        # 象限調整
        if i1[i] < 0 and q1[i] > 0:
            phase[i] = pi - phase[i]
        elif i1[i] < 0 and q1[i] < 0:
            phase[i] = pi + phase[i]
        elif i1[i] > 0 and q1[i] < 0:
            phase[i] = 2 * pi - phase[i]
        
        # 位相差分
        if i >= 1:
            delta_phase = phase[i-1] - phase[i]
            
            # ラップアラウンド処理
            if phase[i-1] < pi/2 and phase[i] > 3*pi/2:
                delta_phase = 2*pi + phase[i-1] - phase[i]
            
            # 適応型制限
            min_delta = period_range[0] * pi / 180 * noise_factor
            max_delta = period_range[1] * pi / 180 / noise_factor
            delta_phase = max(min_delta, min(max_delta, delta_phase))
            
            # 位相累積
            phase_sum = 0.0
            inst_period = 0.0
            for count in range(min(40, i)):
                if i - count >= 1:
                    phase_sum += delta_phase
                    if phase_sum > 2*pi and inst_period == 0:
                        inst_period = count + 1
                        break
            
            if inst_period > 0:
                inst_period = max(period_range[0], min(period_range[1], inst_period))
                period[i] = 0.25 * inst_period + 0.75 * period[i-1]
            else:
                period[i] = period[i-1]
        else:
            period[i] = 20.0
    
    return period


@jit(nopython=True)
def adaptive_dual_differential(
    price: np.ndarray,
    period_range: Tuple[int, int] = (6, 50),
    noise_factor: float = 1.0
) -> np.ndarray:
    """
    適応型デュアル微分アルゴリズム
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    period = np.zeros(n)
    
    # ノイズ適応パラメータ
    smooth_alpha = 0.15 * (1 + noise_factor * 0.3)
    
    for i in range(n):
        if i < 6:
            smooth[i] = price[i]
            period[i] = 20.0
            continue
        
        # スムージング
        smooth[i] = (4 * price[i] + 3 * price[i-1] + 2 * price[i-2] + price[i-3]) / 10
        
        # デトレンド
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)
        
        # 直交コンポーネント
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
        i1[i] = detrender[i-3] if i >= 3 else 0
        
        # 90度位相進み
        if i >= 6:
            ji = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 0.5769 * i1[i-4] - 
                  0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
            jq = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 0.5769 * q1[i-4] - 
                  0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)
            
            i2[i] = i1[i] - jq
            q2[i] = q1[i] + ji
        else:
            i2[i] = i1[i]
            q2[i] = q1[i]
        
        # スムージング
        if i >= 1:
            i2[i] = smooth_alpha * i2[i] + (1 - smooth_alpha) * i2[i-1]
            q2[i] = smooth_alpha * q2[i] + (1 - smooth_alpha) * q2[i-1]
        
        # デュアル微分判別器
        if i >= 1:
            value1 = q2[i] * (i2[i] - i2[i-1]) - i2[i] * (q2[i] - q2[i-1])
            
            if abs(value1) > 0.01:
                raw_period = 2 * pi * (i2[i]**2 + q2[i]**2) / value1
                
                # 適応型制限
                if raw_period > 1.5 * period[i-1]:
                    raw_period = 1.5 * period[i-1]
                elif raw_period < 0.67 * period[i-1]:
                    raw_period = 0.67 * period[i-1]
                
                raw_period = max(period_range[0], min(period_range[1], raw_period))
                period[i] = smooth_alpha * raw_period + (1 - smooth_alpha) * period[i-1]
            else:
                period[i] = period[i-1]
        else:
            period[i] = 20.0
    
    return period


@jit(nopython=True)
def calculate_confidence_weights(
    periods: np.ndarray,
    consistencies: np.ndarray,
    noise_factor: float
) -> np.ndarray:
    """
    各手法の信頼度を計算
    """
    n_methods = len(periods)
    if n_methods == 0:
        return np.ones(1)
    
    weights = np.zeros(n_methods)
    
    # 一貫性ベースの重み
    for i in range(n_methods):
        consistency_weight = consistencies[i]
        
        # ノイズファクターによる調整
        noise_adjustment = 1.0 - 0.3 * noise_factor
        
        weights[i] = consistency_weight * noise_adjustment
    
    # 正規化
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        weights = np.ones(n_methods) / n_methods
    
    return weights


@jit(nopython=True)
def calculate_consistency(period_series: np.ndarray, window: int = 10) -> float:
    """
    期間の一貫性を計算
    """
    if len(period_series) < window:
        return 0.5
    
    recent_periods = period_series[-window:]
    mean_period = np.mean(recent_periods)
    
    if mean_period == 0:
        return 0.0
    
    variance = np.var(recent_periods)
    cv = np.sqrt(variance) / mean_period  # 変動係数
    
    # 一貫性スコア（低い変動係数ほど高いスコア）
    consistency = np.exp(-cv * 2.0)
    return min(1.0, consistency)


@jit(nopython=True)
def ensemble_integration(
    homodyne_period: np.ndarray,
    phase_period: np.ndarray,
    dual_diff_period: np.ndarray,
    noise_factors: np.ndarray,
    window: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アンサンブル統合
    """
    n = len(homodyne_period)
    final_period = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    for i in range(n):
        if i < window:
            # 初期期間は平均を使用
            periods = np.array([homodyne_period[i], phase_period[i], dual_diff_period[i]])
            final_period[i] = np.mean(periods)
            confidence_scores[i] = 0.5
            continue
        
        # 各手法の一貫性を計算
        homodyne_consistency = calculate_consistency(homodyne_period[max(0, i-window):i+1])
        phase_consistency = calculate_consistency(phase_period[max(0, i-window):i+1])
        dual_consistency = calculate_consistency(dual_diff_period[max(0, i-window):i+1])
        
        consistencies = np.array([homodyne_consistency, phase_consistency, dual_consistency])
        periods = np.array([homodyne_period[i], phase_period[i], dual_diff_period[i]])
        
        # 重みを計算
        weights = calculate_confidence_weights(periods, consistencies, noise_factors[i])
        
        # 加重平均
        final_period[i] = np.sum(periods * weights)
        confidence_scores[i] = np.max(consistencies)
    
    return final_period, confidence_scores


@jit(nopython=True)
def calculate_adaptive_ensemble_numba(
    price: np.ndarray,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1,
    entropy_window: int = 20,
    period_range: Tuple[int, int] = (6, 50)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    適応型アンサンブル・サイクル検出のメイン関数
    """
    n = len(price)
    
    # ノイズファクターを計算
    noise_factors = np.zeros(n)
    for i in range(entropy_window, n):
        noise_factors[i] = calculate_entropy(price[max(0, i-entropy_window):i+1], entropy_window)
    
    # 各手法でサイクルピリオドを計算
    homodyne_periods = adaptive_homodyne_discriminator(price, period_range, np.mean(noise_factors))
    phase_periods = adaptive_phase_accumulator(price, period_range, np.mean(noise_factors))
    dual_diff_periods = adaptive_dual_differential(price, period_range, np.mean(noise_factors))
    
    # アンサンブル統合
    final_periods, confidence_scores = ensemble_integration(
        homodyne_periods, phase_periods, dual_diff_periods, noise_factors
    )
    
    # 最終サイクル値を計算
    dom_cycle = np.zeros(n)
    for i in range(n):
        cycle_value = np.ceil(final_periods[i] * cycle_part)
        dom_cycle[i] = max(min_output, min(max_output, cycle_value))
    
    return dom_cycle, final_periods, confidence_scores, noise_factors


class EhlersAdaptiveEnsembleCycle(EhlersDominantCycle):
    """
    革新的な適応型アンサンブル・サイクル検出器
    
    このアルゴリズムは複数の異なるサイクル検出手法を統合し、市場の状況に応じて
    動的に適応する革新的なアプローチを採用しています。
    
    主な特徴:
    1. **マルチ手法アンサンブル**: ホモダイン判別器、位相累積、デュアル微分を統合
    2. **適応型ノイズ処理**: 情報エントロピーベースのノイズレベル評価
    3. **動的重み付け**: 各手法の信頼度に基づく動的重み調整
    4. **一貫性評価**: 期間の一貫性による品質管理
    5. **超低遅延設計**: 最新の市場状況に即座に適応
    
    革新的な要素:
    - 情報理論ベースのノイズ評価
    - 適応型パラメータ調整
    - マルチスケール時間解析
    - 信頼度ベースのアンサンブル
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        entropy_window: int = 20,
        period_range: Tuple[int, int] = (6, 50),
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            entropy_window: エントロピー計算のウィンドウサイズ（デフォルト: 20）
            period_range: サイクル期間の範囲（デフォルト: (6, 50)）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            use_kalman_filter: Kalmanフィルターを使用するか（デフォルト: False）
            kalman_filter_type: Kalmanフィルターのタイプ（デフォルト: 'adaptive'）
        """
        super().__init__(
            f"EhlersAdaptiveEnsemble({cycle_part}, {period_range})",
            cycle_part,
            period_range[1],  # max_cycle
            period_range[0],  # min_cycle
            max_output,
            min_output
        )
        
        self.entropy_window = entropy_window
        self.period_range = period_range
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # Kalmanフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = UnifiedKalman(
                filter_type=kalman_filter_type,
                src_type='close'  # 内部的にはcloseで使用
            )
        
        # 追加の結果保存用
        self._confidence_scores = None
        self._noise_factors = None
    
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
        適応型アンサンブル・アルゴリズムを使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, confidence_scores, noise_factors = calculate_adaptive_ensemble_numba(
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
            self._confidence_scores = confidence_scores
            self._noise_factors = noise_factors
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersAdaptiveEnsembleCycle計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    @property
    def confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        return self._confidence_scores
    
    @property
    def noise_factors(self) -> Optional[np.ndarray]:
        """ノイズファクターを取得"""
        return self._noise_factors
    
    def get_analysis_summary(self) -> Dict:
        """
        分析サマリーを取得
        """
        if self._result is None:
            return {}
        
        summary = {
            'algorithm': 'Adaptive Ensemble Cycle Detector',
            'methods_used': ['Homodyne Discriminator', 'Phase Accumulator', 'Dual Differential'],
            'cycle_range': self.period_range,
            'avg_confidence': np.mean(self._confidence_scores) if self._confidence_scores is not None else None,
            'avg_noise_level': np.mean(self._noise_factors) if self._noise_factors is not None else None,
            'dominant_cycle_stats': {
                'mean': np.mean(self._result.values),
                'std': np.std(self._result.values),
                'min': np.min(self._result.values),
                'max': np.max(self._result.values)
            }
        }
        
        return summary 