#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
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
class UltimateVolatilityStateV2Result:
    """究極のボラティリティ状態判別結果 V2"""
    state: np.ndarray                    # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray              # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray               # 生のボラティリティスコア
    confidence: np.ndarray               # 判定の信頼度
    components: Dict[str, np.ndarray]    # 各コンポーネントの寄与度
    timeframe_analysis: Dict[str, np.ndarray]  # 時間軸別分析結果


@njit(fastmath=True, cache=True)
def discrete_wavelet_transform_simple(data: np.ndarray, levels: int = 3) -> Tuple:
    """
    シンプルなDiscrete Wavelet Transform（Daubechiesライク）
    """
    length = len(data)
    if length < 4 or levels <= 0:
        return data, np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)
    
    # Daubechiesライクなウェーブレット係数（シンプル版）
    h0, h1, h2, h3 = 0.6830127, 1.1830127, 0.3169873, -0.1830127
    g0, g1, g2, g3 = h3, -h2, h1, -h0
    
    # レベル1分解
    approx1 = np.zeros(length)
    detail1 = np.zeros(length)
    
    for i in range(2, length):
        # ローパスフィルター（近似）
        approx1[i] = (h0 * data[i] + h1 * data[i-1] + 
                      h2 * data[i-2] if i >= 2 else 0)
        
        # ハイパスフィルター（詳細）
        detail1[i] = (g0 * data[i] + g1 * data[i-1] + 
                      g2 * data[i-2] if i >= 2 else 0)
    
    # レベル2分解
    approx2 = np.zeros(length)
    detail2 = np.zeros(length)
    
    for i in range(2, length):
        approx2[i] = (h0 * approx1[i] + h1 * approx1[i-1] + 
                      h2 * approx1[i-2] if i >= 2 else 0)
        detail2[i] = (g0 * approx1[i] + g1 * approx1[i-1] + 
                      g2 * approx1[i-2] if i >= 2 else 0)
    
    # レベル3分解
    approx3 = np.zeros(length)
    detail3 = np.zeros(length)
    
    for i in range(2, length):
        approx3[i] = (h0 * approx2[i] + h1 * approx2[i-1] + 
                      h2 * approx2[i-2] if i >= 2 else 0)
        detail3[i] = (g0 * approx2[i] + g1 * approx2[i-1] + 
                      g2 * approx2[i-2] if i >= 2 else 0)
    
    return approx3, detail1, detail2, detail3


@njit(fastmath=True, cache=True)
def calculate_wavelet_volatility(prices: np.ndarray, period: int) -> Tuple:
    """
    ウェーブレット分解によるマルチタイムフレーム・ボラティリティ分析
    """
    length = len(prices)
    short_vol = np.zeros(length)      # 短期ボラティリティ（detail1）
    medium_vol = np.zeros(length)     # 中期ボラティリティ（detail2）
    long_vol = np.zeros(length)       # 長期ボラティリティ（detail3）
    trend_vol = np.zeros(length)      # トレンドボラティリティ（approx3）
    
    # ウェーブレット分解
    approx, detail1, detail2, detail3 = discrete_wavelet_transform_simple(prices)
    
    # 各成分のボラティリティを計算
    for i in range(period, length):
        window_start = max(0, i - period + 1)
        
        # 短期ボラティリティ（高周波成分）
        d1_window = detail1[window_start:i+1]
        short_vol[i] = np.std(d1_window) if len(d1_window) > 1 else 0.0
        
        # 中期ボラティリティ（中周波成分）
        d2_window = detail2[window_start:i+1]
        medium_vol[i] = np.std(d2_window) if len(d2_window) > 1 else 0.0
        
        # 長期ボラティリティ（低周波成分）
        d3_window = detail3[window_start:i+1]
        long_vol[i] = np.std(d3_window) if len(d3_window) > 1 else 0.0
        
        # トレンドボラティリティ（近似成分の変化率）
        approx_window = approx[window_start:i+1]
        if len(approx_window) > 1:
            approx_returns = np.diff(approx_window)
            trend_vol[i] = np.std(approx_returns) if len(approx_returns) > 0 else 0.0
    
    return short_vol, medium_vol, long_vol, trend_vol


@njit(fastmath=True, cache=True)
def calculate_spectral_entropy(data: np.ndarray, period: int) -> np.ndarray:
    """
    スペクトルエントロピーベースの複雑性測定
    """
    length = len(data)
    entropy = np.zeros(length)
    
    for i in range(period, length):
        window = data[i-period+1:i+1]
        
        # パワースペクトラムの簡易推定
        # FFTの代わりに、異なる周期の正弦波との相関を計算
        frequencies = np.array([2, 3, 5, 8, 13, 21])  # フィボナッチ周期
        power = np.zeros(len(frequencies))
        
        for j, freq in enumerate(frequencies):
            if freq < period:
                # 正弦波との相関
                sin_corr = 0.0
                cos_corr = 0.0
                for k in range(len(window)):
                    angle = 2.0 * np.pi * freq * k / period
                    sin_corr += window[k] * np.sin(angle)
                    cos_corr += window[k] * np.cos(angle)
                power[j] = sin_corr * sin_corr + cos_corr * cos_corr
        
        # エントロピー計算
        total_power = np.sum(power)
        if total_power > 0:
            normalized_power = power / total_power
            ent = 0.0
            for p in normalized_power:
                if p > 1e-10:
                    ent -= p * np.log(p)
            entropy[i] = ent / np.log(len(frequencies))  # 正規化
    
    return entropy


@njit(fastmath=True, cache=True)
def calculate_hurst_exponent_rolling(data: np.ndarray, period: int) -> np.ndarray:
    """
    ローリングHurst指数（フラクタル特性）
    """
    length = len(data)
    hurst = np.zeros(length)
    
    for i in range(period, length):
        window = data[i-period+1:i+1]
        
        # R/S分析の簡易版
        lags = np.array([2, 4, 8, 16])
        rs_values = np.zeros(len(lags))
        
        for j, lag in enumerate(lags):
            if lag < len(window):
                # 平均を中心とした累積偏差
                mean_val = np.mean(window)
                cumsum = 0.0
                max_cumsum = -np.inf
                min_cumsum = np.inf
                
                for k in range(len(window)):
                    cumsum += window[k] - mean_val
                    max_cumsum = max(max_cumsum, cumsum)
                    min_cumsum = min(min_cumsum, cumsum)
                
                # Range
                range_val = max_cumsum - min_cumsum
                
                # Standard deviation
                std_val = np.std(window)
                
                # R/S ratio
                if std_val > 0:
                    rs_values[j] = range_val / std_val
                else:
                    rs_values[j] = 1.0
        
        # Hurst指数の推定（簡易版）
        # H = log(R/S) / log(n) の傾き
        if len(rs_values) > 0:
            log_rs = np.log(rs_values + 1e-10)
            log_lags = np.log(lags.astype(np.float64))
            
            # 線形回帰の簡易版
            n = len(log_lags)
            sum_x = np.sum(log_lags)
            sum_y = np.sum(log_rs)
            sum_xy = np.sum(log_lags * log_rs)
            sum_x2 = np.sum(log_lags * log_lags)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                # NumbaでのClip処理
                if slope < 0.0:
                    hurst[i] = 0.0
                elif slope > 1.0:
                    hurst[i] = 1.0
                else:
                    hurst[i] = slope
            else:
                hurst[i] = 0.5
    
    return hurst


@njit(fastmath=True, cache=True)
def calculate_garch_volatility(returns: np.ndarray, period: int) -> np.ndarray:
    """
    GARCH(1,1)風のボラティリティモデル
    """
    length = len(returns)
    garch_vol = np.zeros(length)
    
    alpha = 0.1  # ARCH項の係数
    beta = 0.85  # GARCH項の係数
    omega = 0.01  # 定数項
    
    # 初期値
    if length > 0:
        garch_vol[0] = np.std(returns[:min(period, length)]) if period > 0 else 0.01
    
    for i in range(1, length):
        if i >= period:
            # GARCH(1,1): σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
            prev_return_sq = returns[i-1] * returns[i-1]
            prev_vol_sq = garch_vol[i-1] * garch_vol[i-1]
            
            vol_sq = omega + alpha * prev_return_sq + beta * prev_vol_sq
            garch_vol[i] = np.sqrt(vol_sq)
        else:
            garch_vol[i] = garch_vol[0]
    
    return garch_vol


@njit(fastmath=True, parallel=True, cache=True)
def advanced_volatility_fusion(
    # 既存の指標
    str_zscore: np.ndarray,
    vol_zscore: np.ndarray,
    acceleration: np.ndarray,
    range_vol: np.ndarray,
    entropy: np.ndarray,
    fractal: np.ndarray,
    # 新しい高度な指標
    short_vol: np.ndarray,
    medium_vol: np.ndarray,
    long_vol: np.ndarray,
    trend_vol: np.ndarray,
    spectral_entropy: np.ndarray,
    hurst_exp: np.ndarray,
    garch_vol: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    高度なボラティリティシグナル融合（多角的分析）
    """
    length = len(str_zscore)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    confidence = np.zeros(length)
    
    # 新しい重み係数（精度重視）
    w_str = 0.15          # STR（価格レンジ）
    w_vol = 0.15          # 統計的ボラティリティ
    w_acc = 0.08          # 価格加速度
    w_range = 0.08        # レンジボラティリティ
    w_entropy = 0.05      # 情報エントロピー
    w_fractal = 0.05      # フラクタル次元
    
    # ウェーブレット系（重要度高）
    w_short = 0.18        # 短期ボラティリティ
    w_medium = 0.15       # 中期ボラティリティ
    w_long = 0.12         # 長期ボラティリティ
    w_trend = 0.10        # トレンドボラティリティ
    
    # 高度な分析
    w_spectral = 0.08     # スペクトルエントロピー
    w_hurst = 0.05        # Hurst指数
    w_garch = 0.15        # GARCHボラティリティ
    
    for i in prange(length):
        # 基本指標の正規化
        norm_str = min(max(abs(str_zscore[i]) / 3.0, 0.0), 1.0)
        norm_vol = min(max(abs(vol_zscore[i]) / 3.0, 0.0), 1.0)
        norm_acc = min(max(acceleration[i] * 100, 0.0), 1.0)
        norm_range = min(max(range_vol[i] * 10, 0.0), 1.0)
        norm_entropy = entropy[i]
        norm_fractal = min(max((fractal[i] - 1.0) / 0.5, 0.0), 1.0)
        
        # ウェーブレット系の正規化
        # ウェーブレット系の正規化（NaN対策）
        short_val = short_vol[i] * 50 if not np.isnan(short_vol[i]) else 0.0
        medium_val = medium_vol[i] * 30 if not np.isnan(medium_vol[i]) else 0.0
        long_val = long_vol[i] * 20 if not np.isnan(long_vol[i]) else 0.0
        trend_val = trend_vol[i] * 100 if not np.isnan(trend_vol[i]) else 0.0
        
        norm_short = min(max(short_val, 0.0), 1.0)
        norm_medium = min(max(medium_val, 0.0), 1.0)
        norm_long = min(max(long_val, 0.0), 1.0)
        norm_trend = min(max(trend_val, 0.0), 1.0)
        
        # 高度な分析の正規化（NaN対策）
        spectral_val = spectral_entropy[i] if not np.isnan(spectral_entropy[i]) else 0.0
        hurst_val = hurst_exp[i] if not np.isnan(hurst_exp[i]) else 0.5
        garch_val = garch_vol[i] * 20 if not np.isnan(garch_vol[i]) else 0.0
        
        norm_spectral = min(max(spectral_val, 0.0), 1.0)
        norm_hurst = abs(hurst_val - 0.5) * 2.0  # 0.5からの偏差を強調
        norm_garch = min(max(garch_val, 0.0), 1.0)
        
        # 重み付き融合
        score = (w_str * norm_str + 
                 w_vol * norm_vol + 
                 w_acc * norm_acc + 
                 w_range * norm_range + 
                 w_entropy * norm_entropy + 
                 w_fractal * norm_fractal +
                 w_short * norm_short +
                 w_medium * norm_medium +
                 w_long * norm_long +
                 w_trend * norm_trend +
                 w_spectral * norm_spectral +
                 w_hurst * norm_hurst +
                 w_garch * norm_garch)
        
        raw_score[i] = score
        
        # 信頼度の計算（複数の指標が一致する度合い）
        indicators = np.array([norm_str, norm_vol, norm_short, norm_medium, norm_long, norm_garch])
        mean_indicator = np.mean(indicators)
        std_indicator = np.std(indicators)
        confidence[i] = max(0.0, 1.0 - std_indicator / (mean_indicator + 1e-8))
        
        # 適応的シグモイド関数（信頼度を考慮）
        k = 8.0 + 4.0 * confidence[i]  # 高信頼度ほど急峻
        adjusted_threshold = threshold + 0.1 * (0.5 - confidence[i])  # 低信頼度時は閾値を調整
        
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - adjusted_threshold)))
        
        # 状態判定（信頼度が低い場合は保守的に）
        if confidence[i] > 0.6:
            state[i] = 1 if score > threshold else 0
        else:
            # 信頼度が低い場合はより高い閾値を要求
            state[i] = 1 if score > threshold + 0.15 else 0
    
    return state, probability, raw_score, confidence


class UltimateVolatilityStateV2(Indicator):
    """
    究極のボラティリティ状態判別インジケーター V2（超高精度版）
    
    V2の改良点:
    1. ウェーブレット分解による多時間軸ボラティリティ分析
    2. スペクトルエントロピーによる周波数領域分析
    3. Hurst指数によるフラクタル特性分析
    4. GARCH風ボラティリティモデリング
    5. 信頼度ベースの適応的判定
    6. より精密な重み配分と融合アルゴリズム
    
    特徴:
    - 超高精度: 14の多角的指標による総合判定
    - マルチタイムフレーム: 短期・中期・長期の同時分析
    - 適応性: 信頼度ベースの動的閾値調整
    - 頑健性: 複数の数学的手法の融合
    """
    
    def __init__(
        self,
        period: int = 21,                    # 基本期間
        threshold: float = 0.5,              # 高/低ボラティリティの閾値
        zscore_period: int = 50,             # Z-Score計算期間
        src_type: str = 'hlc3',              # 価格ソース
        smoother_period: int = 3,            # スムージング期間（短縮）
        adaptive_threshold: bool = True,     # 適応的閾値調整
        confidence_threshold: float = 0.7    # 信頼度閾値
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本計算期間
            threshold: ボラティリティ判定閾値 (0.0-1.0)
            zscore_period: Z-Score正規化期間
            src_type: 価格ソースタイプ
            smoother_period: 最終出力のスムージング期間
            adaptive_threshold: 閾値の動的調整を有効化
            confidence_threshold: 信頼度の最小閾値
        """
        super().__init__(f"UltimateVolatilityStateV2(period={period}, threshold={threshold})")
        
        self.period = period
        self.threshold = threshold
        self.zscore_period = zscore_period
        self.src_type = src_type.lower()
        self.smoother_period = smoother_period
        self.adaptive_threshold = adaptive_threshold
        self.confidence_threshold = confidence_threshold
        
        # コンポーネントインジケーターの初期化
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
            smoother_period=period // 3
        )
        
        self.smoother = UltimateSmoother(
            period=smoother_period,
            src_type='close'
        )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
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
            
            params_sig = f"{self.period}_{self.threshold}_{self.zscore_period}_{self.confidence_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.period}_{self.threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateVolatilityStateV2Result:
        """
        高精度ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            UltimateVolatilityStateV2Result: 判定結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                return self._result_cache[data_hash]
            
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
            min_required = max(self.period, self.zscore_period) + 50
            
            if length < min_required:
                # データ不足時の空結果
                empty_result = UltimateVolatilityStateV2Result(
                    state=np.zeros(length, dtype=np.int8),
                    probability=np.zeros(length),
                    raw_score=np.zeros(length),
                    confidence=np.zeros(length),
                    components={},
                    timeframe_analysis={}
                )
                return empty_result
            
            # === 既存の分析 ===
            
            # 1. STRベースのボラティリティ
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            str_zscore_calc = ZScore(period=self.zscore_period)
            str_df = pd.DataFrame({'close': str_values})
            str_zscore = str_zscore_calc.calculate(str_df)
            
            # 2. 統計的ボラティリティ
            vol_values = self.vol_indicator.calculate(data)
            
            vol_zscore_calc = ZScore(period=self.zscore_period)
            vol_df = pd.DataFrame({'close': vol_values})
            vol_zscore = vol_zscore_calc.calculate(vol_df)
            
            # 3. 価格加速度
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            # === 新しい高度な分析 ===
            
            # 4. ウェーブレット分解による多時間軸分析
            short_vol, medium_vol, long_vol, trend_vol = calculate_wavelet_volatility(
                src_prices, self.period
            )
            
            # 5. スペクトルエントロピー
            spectral_entropy = calculate_spectral_entropy(src_prices, self.period)
            
            # 6. Hurst指数（フラクタル特性）
            hurst_exp = calculate_hurst_exponent_rolling(src_prices, self.period)
            
            # 7. GARCH風ボラティリティ
            returns = np.zeros(length)
            for i in range(1, length):
                if close[i-1] > 0:
                    returns[i] = np.log(close[i] / close[i-1])
            garch_vol = calculate_garch_volatility(returns, self.period)
            
            # 8. その他の既存指標（簡略化）
            acceleration = np.zeros(length)
            for i in range(self.period, length):
                if src_prices[i-self.period] > 0:
                    vel1 = (src_prices[i-self.period//2] - src_prices[i-self.period]) / src_prices[i-self.period]
                    vel2 = (src_prices[i] - src_prices[i-self.period//2]) / src_prices[i-self.period//2] if src_prices[i-self.period//2] > 0 else 0
                    acceleration[i] = abs(vel2 - vel1)
            
            range_vol = np.zeros(length)
            for i in range(self.period, length):
                window_high = high[i-self.period+1:i+1]
                window_low = low[i-self.period+1:i+1]
                window_close = close[i-self.period+1:i+1]
                
                avg_range = np.mean(window_high - window_low)
                avg_close = np.mean(window_close)
                range_vol[i] = avg_range / avg_close if avg_close > 0 else 0
            
            entropy = np.zeros(length)  # 簡略化
            fractal = np.ones(length) * 1.5  # デフォルト値
            
            # 適応的閾値調整
            effective_threshold = self.threshold
            if self.adaptive_threshold:
                lookback = min(200, length // 5)
                if length > lookback:
                    recent_scores = []
                    for i in range(length - lookback, length):
                        if i >= min_required:
                            # 複数指標の統合スコア
                            score = (abs(str_zscore[i]) + abs(vol_zscore[i]) + 
                                   short_vol[i] * 50 + medium_vol[i] * 30 + 
                                   garch_vol[i] * 20) / 5
                            recent_scores.append(score)
                    
                    if recent_scores:
                        median_score = np.median(recent_scores)
                        effective_threshold = 0.25 + 0.5 * min(median_score / 2.0, 1.0)
                        effective_threshold = np.clip(effective_threshold, 0.25, 0.75)
            
            # 高度なシグナル融合
            state, probability, raw_score, confidence = advanced_volatility_fusion(
                str_zscore, vol_zscore, acceleration, range_vol,
                entropy, fractal, short_vol, medium_vol, long_vol, trend_vol,
                spectral_entropy, hurst_exp, garch_vol, effective_threshold
            )
            
            # 最終的なスムージング（信頼度重み付き）
            # 高信頼度の判定ほどスムージングを控えめに
            smooth_weights = np.maximum(confidence, 0.3)  # 最小重み0.3
            
            state_df = pd.DataFrame({'close': state.astype(np.float64) * smooth_weights})
            smoothed_state_result = self.smoother.calculate(state_df)
            smoothed_state = (smoothed_state_result.values > 0.5).astype(np.int8)
            
            prob_df = pd.DataFrame({'close': probability * smooth_weights})
            smoothed_prob_result = self.smoother.calculate(prob_df)
            smoothed_probability = smoothed_prob_result.values
            
            # コンポーネントとタイムフレーム分析の保存
            components = {
                'str_zscore': str_zscore.copy(),
                'vol_zscore': vol_zscore.copy(),
                'acceleration': acceleration.copy(),
                'range_volatility': range_vol.copy(),
                'spectral_entropy': spectral_entropy.copy(),
                'hurst_exponent': hurst_exp.copy(),
                'garch_volatility': garch_vol.copy()
            }
            
            timeframe_analysis = {
                'short_term_volatility': short_vol.copy(),
                'medium_term_volatility': medium_vol.copy(),
                'long_term_volatility': long_vol.copy(),
                'trend_volatility': trend_vol.copy()
            }
            
            # 結果作成
            result = UltimateVolatilityStateV2Result(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                confidence=confidence,
                components=components,
                timeframe_analysis=timeframe_analysis
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"V2ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            # エラー時は空の結果を返す
            return UltimateVolatilityStateV2Result(
                state=np.zeros(len(data), dtype=np.int8),
                probability=np.zeros(len(data)),
                raw_score=np.zeros(len(data)),
                confidence=np.zeros(len(data)),
                components={},
                timeframe_analysis={}
            )
    
    def get_state(self) -> Optional[np.ndarray]:
        """現在のボラティリティ状態を取得 (1: 高, 0: 低)"""
        if self._values is not None:
            return self._values.astype(np.int8)
        return None
    
    def get_confidence(self) -> Optional[np.ndarray]:
        """判定の信頼度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.confidence
        return None
    
    def get_timeframe_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """タイムフレーム別分析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.timeframe_analysis
        return None
    
    def is_high_volatility_confident(self, min_confidence: float = None) -> bool:
        """高信頼度での高ボラティリティ判定"""
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        state = self.get_state()
        confidence = self.get_confidence()
        
        if state is not None and confidence is not None and len(state) > 0:
            latest_state = state[-1]
            latest_confidence = confidence[-1]
            return bool(latest_state == 1 and latest_confidence >= min_confidence)
        return False
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        self.vol_indicator.reset()
        self.smoother.reset()