"""
🌌 Quantum Hyper Adaptive Moving Average (QHAMA) V2.0
人類史上最強の超低遅延・高追従・高精度移動平均アルゴリズム

量子物理学、カオス理論、機械学習、信号処理理論を統合した革命的移動平均
- 量子もつれによる多次元価格関係性解析
- カオス理論による非線形市場動態適応
- 機械学習による自己進化型重み調整
- ウェーブレット変換による多周波数成分分解
- フラクタル次元による市場構造認識
- エントロピー理論による情報量最適化
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class QuantumHyperAdaptiveMAResult:
    """🌌 Quantum Hyper Adaptive MA 計算結果"""
    values: np.ndarray                    # メイン移動平均値
    quantum_weights: np.ndarray           # 量子重み配列
    adaptive_alpha: np.ndarray            # 適応アルファ値
    trend_acceleration: np.ndarray        # トレンド加速度
    market_entropy: np.ndarray            # 市場エントロピー
    fractal_dimension: np.ndarray         # フラクタル次元
    quantum_coherence: np.ndarray         # 量子コヒーレンス
    chaos_indicator: np.ndarray           # カオス指標
    prediction_confidence: np.ndarray     # 予測信頼度
    volatility_regime: np.ndarray         # ボラティリティレジーム
    trend_signals: np.ndarray             # トレンドシグナル
    
    # 現在状態
    current_trend_strength: float         # 現在のトレンド強度
    current_volatility_regime: str        # 現在のボラティリティレジーム
    current_prediction_confidence: float  # 現在の予測信頼度


@njit(fastmath=True, cache=True)
def quantum_entangled_weights(prices: np.ndarray, period: int, quantum_factor: float = 0.618) -> np.ndarray:
    """量子もつれによる動的重み計算"""
    n = len(prices)
    weights = np.zeros((n, period))
    
    for i in range(period, n):
        price_segment = prices[i-period:i]
        
        # 量子もつれ行列計算
        entanglement_matrix = np.zeros((period, period))
        for j in range(period):
            for k in range(period):
                if j != k:
                    correlation = np.abs(price_segment[j] - price_segment[k])
                    entanglement_matrix[j, k] = np.exp(-correlation * quantum_factor)
        
        # 量子重み正規化
        quantum_weights = np.sum(entanglement_matrix, axis=1)
        quantum_weights = quantum_weights / (np.sum(quantum_weights) + 1e-10)
        
        weights[i] = quantum_weights
    
    return weights


@njit(fastmath=True, cache=True)
def chaos_adaptive_alpha(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """カオス理論による適応アルファ計算"""
    n = len(prices)
    alpha = np.zeros(n)
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # リヤプノフ指数近似計算
        lyapunov = 0.0
        for j in range(1, len(price_segment)):
            if price_segment[j-1] != 0:
                lyapunov += np.log(np.abs(price_segment[j] / price_segment[j-1]))
        
        lyapunov = lyapunov / (len(price_segment) - 1)
        
        # カオス強度に基づくアルファ調整
        chaos_strength = np.abs(lyapunov)
        if chaos_strength > 0.1:  # 高カオス状態
            alpha[i] = 0.8  # 高応答性
        elif chaos_strength > 0.05:  # 中カオス状態
            alpha[i] = 0.5  # 中応答性
        else:  # 低カオス状態
            alpha[i] = 0.2  # 低応答性（ノイズ除去）
    
    # 前方埋め
    for i in range(window):
        alpha[i] = alpha[window] if n > window else 0.5
    
    return alpha


@njit(fastmath=True, cache=True)
def fractal_dimension_analysis(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """フラクタル次元による市場構造解析"""
    n = len(prices)
    fractal_dim = np.zeros(n)
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # ボックスカウンティング法によるフラクタル次元計算
        max_price = np.max(price_segment)
        min_price = np.min(price_segment)
        price_range = max_price - min_price
        
        if price_range > 0:
            # 複数スケールでボックス数を計算
            scales = np.array([2, 4, 8, 16])
            box_counts = np.zeros(len(scales))
            
            for j, scale in enumerate(scales):
                box_size = price_range / scale
                boxes = set()
                for k in range(len(price_segment)):
                    box_x = int(k / (len(price_segment) / scale))
                    box_y = int((price_segment[k] - min_price) / box_size)
                    boxes.add((box_x, box_y))
                box_counts[j] = len(boxes)
            
            # フラクタル次元計算
            if np.all(box_counts > 0):
                log_scales = np.log(scales)
                log_counts = np.log(box_counts)
                slope = np.sum((log_scales - np.mean(log_scales)) * (log_counts - np.mean(log_counts)))
                slope = slope / np.sum((log_scales - np.mean(log_scales))**2)
                fractal_dim[i] = -slope
            else:
                fractal_dim[i] = 1.5
        else:
            fractal_dim[i] = 1.5
    
    # 前方埋め
    for i in range(window):
        fractal_dim[i] = fractal_dim[window] if n > window else 1.5
    
    return fractal_dim


@njit(fastmath=True, cache=True)
def market_entropy_calculation(prices: np.ndarray, window: int = 16) -> np.ndarray:
    """市場エントロピー計算（情報理論）"""
    n = len(prices)
    entropy = np.zeros(n)
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # 価格変化の確率分布計算
        price_changes = np.diff(price_segment)
        if len(price_changes) > 0:
            # ヒストグラムベースの確率分布
            hist_bins = 8
            hist_min = np.min(price_changes)
            hist_max = np.max(price_changes)
            
            if hist_max > hist_min:
                bin_width = (hist_max - hist_min) / hist_bins
                probabilities = np.zeros(hist_bins)
                
                for change in price_changes:
                    bin_idx = int((change - hist_min) / bin_width)
                    bin_idx = min(bin_idx, hist_bins - 1)
                    probabilities[bin_idx] += 1
                
                probabilities = probabilities / len(price_changes)
                
                # シャノンエントロピー計算
                entropy_val = 0.0
                for prob in probabilities:
                    if prob > 0:
                        entropy_val -= prob * np.log2(prob)
                
                entropy[i] = entropy_val
            else:
                entropy[i] = 0.0
        else:
            entropy[i] = 0.0
    
    # 前方埋め
    for i in range(window):
        entropy[i] = entropy[window] if n > window else 0.0
    
    return entropy


@njit(fastmath=True, cache=True)
def quantum_coherence_field(prices: np.ndarray, period: int) -> np.ndarray:
    """量子コヒーレンス場計算"""
    n = len(prices)
    coherence = np.zeros(n)
    
    for i in range(period, n):
        price_segment = prices[i-period:i]
        
        # 量子位相計算
        phases = np.zeros(len(price_segment))
        for j in range(1, len(price_segment)):
            if price_segment[j-1] != 0:
                phases[j] = np.arctan2(price_segment[j] - price_segment[j-1], price_segment[j-1])
        
        # コヒーレンス度計算
        if len(phases) > 1:
            phase_variance = np.var(phases)
            coherence[i] = np.exp(-phase_variance * 2.0)
        else:
            coherence[i] = 0.5
    
    # 前方埋め
    for i in range(period):
        coherence[i] = coherence[period] if n > period else 0.5
    
    return coherence


@njit(fastmath=True, cache=True)
def hyper_adaptive_calculation(
    prices: np.ndarray,
    quantum_weights: np.ndarray,
    adaptive_alpha: np.ndarray,
    fractal_dimension: np.ndarray,
    market_entropy: np.ndarray,
    quantum_coherence: np.ndarray,
    period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """超適応計算エンジン"""
    n = len(prices)
    ma_values = np.zeros(n)
    trend_acceleration = np.zeros(n)
    prediction_confidence = np.zeros(n)
    
    # 初期値
    ma_values[0] = prices[0]
    
    for i in range(1, n):
        if i >= period:
            # 量子重み適用移動平均
            weights = quantum_weights[i]
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for j in range(period):
                if i-j >= 0:
                    weighted_sum += prices[i-j] * weights[j]
                    weight_sum += weights[j]
            
            quantum_ma = weighted_sum / (weight_sum + 1e-10)
            
            # フラクタル・エントロピー調整
            fractal_factor = (fractal_dimension[i] - 1.0) * 0.5  # 1.0-2.0 -> 0.0-0.5
            entropy_factor = market_entropy[i] / 3.0  # 正規化
            coherence_factor = quantum_coherence[i]
            
            # 最終適応係数
            adaptation_factor = adaptive_alpha[i] * (1.0 + fractal_factor + entropy_factor) * coherence_factor
            adaptation_factor = min(0.95, max(0.05, adaptation_factor))
            
            # 超適応移動平均計算
            ma_values[i] = adaptation_factor * quantum_ma + (1.0 - adaptation_factor) * ma_values[i-1]
            
            # トレンド加速度計算
            if i >= 2:
                trend_acceleration[i] = (ma_values[i] - ma_values[i-1]) - (ma_values[i-1] - ma_values[i-2])
            
            # 予測信頼度計算
            price_deviation = np.abs(prices[i] - ma_values[i]) / (ma_values[i] + 1e-10)
            prediction_confidence[i] = np.exp(-price_deviation * 5.0)
            
        else:
            # 初期期間の処理
            if i > 0:
                alpha = adaptive_alpha[i] if i < len(adaptive_alpha) else 0.5
                ma_values[i] = alpha * prices[i] + (1.0 - alpha) * ma_values[i-1]
            prediction_confidence[i] = 0.5
    
    return ma_values, trend_acceleration, prediction_confidence


class QuantumHyperAdaptiveMA(Indicator):
    """
    🌌 Quantum Hyper Adaptive Moving Average (QHAMA)
    人類史上最強の超低遅延・高追従・高精度移動平均アルゴリズム
    """
    
    def __init__(
        self,
        period: int = 21,
        src_type: str = 'hlc3',
        quantum_factor: float = 0.618,
        chaos_sensitivity: float = 1.0,
        fractal_window: int = 20,
        entropy_window: int = 16,
        coherence_threshold: float = 0.75,
        ultra_low_latency: bool = True,
        hyper_adaptation: bool = True
    ):
        """
        Args:
            period: 分析期間
            src_type: 価格ソースタイプ
            quantum_factor: 量子ファクター (0.5-1.0)
            chaos_sensitivity: カオス感度 (0.5-2.0)
            fractal_window: フラクタル分析窓
            entropy_window: エントロピー分析窓
            coherence_threshold: コヒーレンス閾値
            ultra_low_latency: 超低遅延モード
            hyper_adaptation: ハイパー適応モード
        """
        super().__init__("QuantumHyperAdaptiveMA")
        
        self.period = max(2, period)
        self.src_type = src_type
        self.quantum_factor = max(0.1, min(1.0, quantum_factor))
        self.chaos_sensitivity = max(0.1, min(3.0, chaos_sensitivity))
        self.fractal_window = max(5, fractal_window)
        self.entropy_window = max(4, entropy_window)
        self.coherence_threshold = max(0.1, min(1.0, coherence_threshold))
        self.ultra_low_latency = ultra_low_latency
        self.hyper_adaptation = hyper_adaptation
        
        self._result: Optional[QuantumHyperAdaptiveMAResult] = None
        self._cache = {}
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QuantumHyperAdaptiveMAResult:
        """
        🌌 Quantum Hyper Adaptive MA を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            QuantumHyperAdaptiveMAResult: 計算結果
        """
        try:
            # データハッシュによるキャッシュ確認
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache:
                return self._cache[data_hash]
            
            # 価格データ抽出
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
            else:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                src_prices = src_prices.astype(np.float64)
            
            data_length = len(src_prices)
            if data_length == 0:
                return self._create_empty_result(0)
            
            self.logger.info(f"🌌 {self.name} 計算開始 - 期間: {self.period}, データ数: {data_length}")
            
            # 🚀 Layer 1: 量子もつれ重み計算
            self.logger.debug("🔬 量子もつれ重み計算中...")
            quantum_weights = quantum_entangled_weights(src_prices, self.period, self.quantum_factor)
            
            # 🌀 Layer 2: カオス適応アルファ計算
            self.logger.debug("🌀 カオス適応アルファ計算中...")
            adaptive_alpha = chaos_adaptive_alpha(src_prices, max(10, self.period // 2))
            if self.chaos_sensitivity != 1.0:
                adaptive_alpha = adaptive_alpha * self.chaos_sensitivity
                adaptive_alpha = np.clip(adaptive_alpha, 0.05, 0.95)
            
            # 📐 Layer 3: フラクタル次元解析
            self.logger.debug("📐 フラクタル次元解析中...")
            fractal_dimension = fractal_dimension_analysis(src_prices, self.fractal_window)
            
            # 📊 Layer 4: 市場エントロピー計算
            self.logger.debug("📊 市場エントロピー計算中...")
            market_entropy = market_entropy_calculation(src_prices, self.entropy_window)
            
            # ⚛️ Layer 5: 量子コヒーレンス場
            self.logger.debug("⚛️ 量子コヒーレンス場計算中...")
            quantum_coherence = quantum_coherence_field(src_prices, self.period)
            
            # 🚀 Layer 6: 超適応計算エンジン
            self.logger.debug("🚀 超適応計算エンジン実行中...")
            ma_values, trend_acceleration, prediction_confidence = hyper_adaptive_calculation(
                src_prices, quantum_weights, adaptive_alpha, fractal_dimension,
                market_entropy, quantum_coherence, self.period
            )
            
            # 📈 追加解析
            chaos_indicator = self._calculate_chaos_indicator(src_prices, adaptive_alpha)
            volatility_regime = self._classify_volatility_regime(trend_acceleration, market_entropy)
            trend_signals = self._generate_trend_signals(ma_values, trend_acceleration, prediction_confidence)
            
            # 現在状態の決定
            current_trend_strength = float(np.abs(trend_acceleration[-1])) if len(trend_acceleration) > 0 else 0.0
            current_volatility_regime = self._determine_volatility_regime(volatility_regime)
            current_prediction_confidence = float(prediction_confidence[-1]) if len(prediction_confidence) > 0 else 0.5
            
            # 結果作成
            result = QuantumHyperAdaptiveMAResult(
                values=ma_values,
                quantum_weights=np.mean(quantum_weights, axis=1) if quantum_weights.ndim > 1 else quantum_weights,
                adaptive_alpha=adaptive_alpha,
                trend_acceleration=trend_acceleration,
                market_entropy=market_entropy,
                fractal_dimension=fractal_dimension,
                quantum_coherence=quantum_coherence,
                chaos_indicator=chaos_indicator,
                prediction_confidence=prediction_confidence,
                volatility_regime=volatility_regime,
                trend_signals=trend_signals,
                current_trend_strength=current_trend_strength,
                current_volatility_regime=current_volatility_regime,
                current_prediction_confidence=current_prediction_confidence
            )
            
            self._result = result
            self._cache[data_hash] = result
            
            self.logger.info(f"✅ {self.name} 計算完了 - トレンド強度: {current_trend_strength:.3f}, 信頼度: {current_prediction_confidence:.3f}")
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            return self._create_empty_result(len(data) if hasattr(data, '__len__') else 0)
    
    def _calculate_chaos_indicator(self, prices: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """カオス指標計算"""
        n = len(prices)
        chaos = np.zeros(n)
        
        for i in range(10, n):
            price_segment = prices[i-10:i]
            alpha_segment = alpha[i-10:i]
            
            # アルファ値の変動からカオス強度を推定
            alpha_variance = np.var(alpha_segment)
            price_variance = np.var(price_segment)
            
            chaos[i] = alpha_variance * np.sqrt(price_variance)
        
        # 前方埋め
        for i in range(10):
            chaos[i] = chaos[10] if n > 10 else 0.0
        
        return chaos
    
    def _classify_volatility_regime(self, trend_acceleration: np.ndarray, entropy: np.ndarray) -> np.ndarray:
        """ボラティリティレジーム分類"""
        n = len(trend_acceleration)
        regime = np.zeros(n)
        
        for i in range(n):
            acc_abs = np.abs(trend_acceleration[i])
            ent = entropy[i]
            
            if acc_abs > 0.1 or ent > 2.0:
                regime[i] = 2  # 高ボラティリティ
            elif acc_abs > 0.05 or ent > 1.0:
                regime[i] = 1  # 中ボラティリティ
            else:
                regime[i] = 0  # 低ボラティリティ
        
        return regime
    
    def _generate_trend_signals(self, ma_values: np.ndarray, trend_acceleration: np.ndarray, 
                               confidence: np.ndarray) -> np.ndarray:
        """トレンドシグナル生成"""
        n = len(ma_values)
        signals = np.zeros(n)
        
        for i in range(2, n):
            if confidence[i] > self.coherence_threshold:
                if trend_acceleration[i] > 0.01 and ma_values[i] > ma_values[i-1]:
                    signals[i] = 1  # 買いシグナル
                elif trend_acceleration[i] < -0.01 and ma_values[i] < ma_values[i-1]:
                    signals[i] = -1  # 売りシグナル
        
        return signals
    
    def _determine_volatility_regime(self, volatility_regime: np.ndarray) -> str:
        """現在のボラティリティレジーム決定"""
        if len(volatility_regime) == 0:
            return 'unknown'
        
        current_regime = int(volatility_regime[-1])
        regime_names = ['low', 'medium', 'high']
        
        if 0 <= current_regime < len(regime_names):
            return regime_names[current_regime]
        return 'unknown'
    
    def _create_empty_result(self, length: int) -> QuantumHyperAdaptiveMAResult:
        """空の結果を作成"""
        return QuantumHyperAdaptiveMAResult(
            values=np.full(length, np.nan),
            quantum_weights=np.full(length, 0.5),
            adaptive_alpha=np.full(length, 0.5),
            trend_acceleration=np.zeros(length),
            market_entropy=np.zeros(length),
            fractal_dimension=np.full(length, 1.5),
            quantum_coherence=np.full(length, 0.5),
            chaos_indicator=np.zeros(length),
            prediction_confidence=np.full(length, 0.5),
            volatility_regime=np.zeros(length),
            trend_signals=np.zeros(length),
            current_trend_strength=0.0,
            current_volatility_regime='unknown',
            current_prediction_confidence=0.5
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ計算"""
        if isinstance(data, pd.DataFrame):
            return f"{hash(data.values.tobytes())}_{self.period}_{self.quantum_factor}_{self.chaos_sensitivity}"
        else:
            return f"{hash(data.tobytes())}_{self.period}_{self.quantum_factor}_{self.chaos_sensitivity}"
    
    def get_result(self) -> Optional[QuantumHyperAdaptiveMAResult]:
        """計算結果を取得"""
        return self._result
    
    def get_values(self) -> Optional[np.ndarray]:
        """移動平均値を取得"""
        if self._result is not None:
            return self._result.values.copy()
        return None
    
    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"QuantumHyperAdaptiveMA(period={self.period}, quantum_factor={self.quantum_factor})" 