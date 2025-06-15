#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇宙最強4状態市場分類器 - Cosmos Supreme 4-State Market Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌌 Cosmic 4-State Classification:
1. 🔥 High Volatility Range Market  - 高ボラレンジ相場
2. ❄️ Low Volatility Range Market   - 低ボラレンジ相場
3. 🚀 High Volatility Trend Market  - 高ボラトレンド相場
4. 🐌 Low Volatility Trend Market   - 低ボラトレンド相場

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pywt
import warnings
from numba import jit, prange
import logging

warnings.filterwarnings('ignore')

@dataclass
class Cosmic4StateResult:
    """宇宙4状態分類結果"""
    market_state: str  # "HighVolRange", "LowVolRange", "HighVolTrend", "LowVolTrend"
    trend_strength: float  # トレンド強度 (0.0-1.0)
    volatility_level: float  # ボラティリティレベル (0.0-1.0)
    confidence: float  # 分類信頼度
    stability: float  # 安定性スコア
    method_consensus: Dict[str, Dict[str, str]]  # 各手法の判定結果
    volatility_metrics: Dict[str, float]  # ボラティリティ指標
    trend_metrics: Dict[str, float]  # トレンド指標


class CosmosSupreme4StateClassifier:
    """
    宇宙最強4状態市場分類器
    
    🌟 Revolutionary 4-State Classification:
    - 高精度トレンド/レンジ判定
    - 多次元ボラティリティ解析
    - 動的閾値による適応的分類
    - 純粋統計的根拠による判定
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [10, 21, 50, 100],
                 volatility_lookback: int = 252):
        """
        Args:
            lookback_periods: 分析期間のリスト
            volatility_lookback: ボラティリティ評価期間
        """
        self.lookback_periods = lookback_periods
        self.volatility_lookback = volatility_lookback
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🌌 Cosmos Supreme 4-State Market Classifier v2.0 初期化完了")
        print("🎯 目標: 4つの市場状態の超高精度分類")

    def classify(self, data: pd.DataFrame) -> Cosmic4StateResult:
        """
        メイン分類メソッド
        
        Args:
            data: OHLCV データ
            
        Returns:
            Cosmic4StateResult: 分類結果
        """
        if len(data) < max(self.lookback_periods):
            return Cosmic4StateResult(
                market_state="LowVolRange",  # デフォルト
                trend_strength=0.0,
                volatility_level=0.0,
                confidence=0.5,
                stability=0.0,
                method_consensus={},
                volatility_metrics={},
                trend_metrics={}
            )
        
        # 1. 複数手法による独立分析
        method_results = self._multi_method_analysis(data)
        
        # 2. 統合ボラティリティ解析
        volatility_analysis = self._comprehensive_volatility_analysis(data)
        
        # 3. 統合トレンド解析
        trend_analysis = self._comprehensive_trend_analysis(data)
        
        # 4. 4状態分類
        market_state = self._classify_4_states(trend_analysis, volatility_analysis)
        
        # 5. 分類信頼度計算
        confidence = self._calculate_classification_confidence(method_results, trend_analysis, volatility_analysis)
        
        # 6. 安定性評価
        stability = self._calculate_stability_score(data)
        
        return Cosmic4StateResult(
            market_state=market_state,
            trend_strength=trend_analysis['strength'],
            volatility_level=volatility_analysis['level'],
            confidence=confidence,
            stability=stability,
            method_consensus={method: {'trend': result.get('trend_pattern', 'Range'), 
                                    'volatility': result.get('vol_pattern', 'Low')} 
                           for method, result in method_results.items()},
            volatility_metrics=volatility_analysis['metrics'],
            trend_metrics=trend_analysis['metrics']
        )

    def _multi_method_analysis(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """複数手法による独立分析"""
        methods = {}
        
        # 1. Wavelet-Fourier Hybrid Analysis
        methods['wavelet_fourier'] = self._wavelet_fourier_4state_analysis(data)
        
        # 2. Fractal Dimension Analysis
        methods['fractal'] = self._fractal_4state_analysis(data)
        
        # 3. Entropy-based Analysis
        methods['entropy'] = self._entropy_4state_analysis(data)
        
        # 4. Advanced Momentum Analysis
        methods['momentum'] = self._momentum_4state_analysis(data)
        
        # 5. Support/Resistance Analysis
        methods['support_resistance'] = self._sr_4state_analysis(data)
        
        # 6. Change Point Detection
        methods['change_point'] = self._changepoint_4state_analysis(data)
        
        # 7. Kalman Filter Analysis
        methods['kalman'] = self._kalman_4state_analysis(data)
        
        return methods

    def _comprehensive_volatility_analysis(self, data: pd.DataFrame) -> Dict:
        """統合ボラティリティ解析"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        vol_metrics = {}
        
        # 1. True Range Volatility
        tr = self._calculate_true_range(high_prices, low_prices, close_prices)
        current_atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        historical_atr = np.mean(tr[-self.volatility_lookback:]) if len(tr) >= self.volatility_lookback else np.mean(tr)
        vol_metrics['atr_ratio'] = current_atr / historical_atr if historical_atr > 0 else 1.0
        
        # 2. Close-to-Close Volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        current_vol = np.std(returns[-21:]) if len(returns) >= 21 else np.std(returns)
        historical_vol = np.std(returns[-self.volatility_lookback:]) if len(returns) >= self.volatility_lookback else np.std(returns)
        vol_metrics['return_vol_ratio'] = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 3. Intraday Range Volatility
        daily_ranges = (high_prices - low_prices) / close_prices
        current_range_vol = np.mean(daily_ranges[-21:]) if len(daily_ranges) >= 21 else np.mean(daily_ranges)
        historical_range_vol = np.mean(daily_ranges[-self.volatility_lookback:]) if len(daily_ranges) >= self.volatility_lookback else np.mean(daily_ranges)
        vol_metrics['range_vol_ratio'] = current_range_vol / historical_range_vol if historical_range_vol > 0 else 1.0
        
        # 4. GARCH-like Volatility Persistence
        if len(returns) >= 50:
            vol_persistence = self._calculate_volatility_persistence(returns)
            vol_metrics['persistence'] = vol_persistence
        else:
            vol_metrics['persistence'] = 0.5
        
        # 5. VIX-like Volatility Index
        if len(returns) >= 30:
            vix_like = self._calculate_vix_like_index(returns)
            vol_metrics['vix_like'] = vix_like
        else:
            vol_metrics['vix_like'] = 0.5
        
        # 統合ボラティリティレベル計算
        vol_components = [
            vol_metrics['atr_ratio'],
            vol_metrics['return_vol_ratio'], 
            vol_metrics['range_vol_ratio']
        ]
        integrated_vol = np.mean(vol_components)
        
        # 動的閾値（過去データに基づく）
        vol_threshold = self._calculate_dynamic_volatility_threshold(data)
        
        return {
            'level': min(1.0, integrated_vol),
            'is_high': integrated_vol > vol_threshold,
            'threshold': vol_threshold,
            'metrics': vol_metrics
        }

    def _comprehensive_trend_analysis(self, data: pd.DataFrame) -> Dict:
        """統合トレンド解析"""
        close_prices = data['close'].values
        
        trend_metrics = {}
        
        # 1. Multi-timeframe Linear Regression
        regression_strengths = []
        for period in self.lookback_periods:
            if len(close_prices) >= period:
                x = np.arange(period)
                y = close_prices[-period:]
                _, _, r_value, _, _ = stats.linregress(x, y)
                regression_strengths.append(abs(r_value))
        
        trend_metrics['avg_regression_strength'] = np.mean(regression_strengths) if regression_strengths else 0
        
        # 2. Directional Movement Index (DMI)
        if len(data) >= 14:
            dmi_trend = self._calculate_dmi_trend(data)
            trend_metrics['dmi_trend'] = dmi_trend
        else:
            trend_metrics['dmi_trend'] = 0
        
        # 3. Moving Average Convergence
        if len(close_prices) >= 50:
            ma_trend = self._calculate_ma_trend_strength(close_prices)
            trend_metrics['ma_trend'] = ma_trend
        else:
            trend_metrics['ma_trend'] = 0
        
        # 4. Price Channel Breakout Analysis
        channel_trend = self._calculate_channel_trend(data)
        trend_metrics['channel_trend'] = channel_trend
        
        # 5. Momentum Consistency
        momentum_trend = self._calculate_momentum_consistency(close_prices)
        trend_metrics['momentum_consistency'] = momentum_trend
        
        # 統合トレンド強度計算
        trend_components = [
            trend_metrics['avg_regression_strength'],
            trend_metrics['dmi_trend'],
            trend_metrics['ma_trend'],
            trend_metrics['channel_trend'],
            trend_metrics['momentum_consistency']
        ]
        integrated_trend = np.mean(trend_components)
        
        # 動的閾値（過去データに基づく）
        trend_threshold = self._calculate_dynamic_trend_threshold(data)
        
        return {
            'strength': min(1.0, integrated_trend),
            'is_trending': integrated_trend > trend_threshold,
            'threshold': trend_threshold,
            'metrics': trend_metrics
        }

    def _classify_4_states(self, trend_analysis: Dict, volatility_analysis: Dict) -> str:
        """4状態分類"""
        is_trending = trend_analysis['is_trending']
        is_high_vol = volatility_analysis['is_high']
        
        if is_trending and is_high_vol:
            return "HighVolTrend"
        elif is_trending and not is_high_vol:
            return "LowVolTrend"
        elif not is_trending and is_high_vol:
            return "HighVolRange"
        else:
            return "LowVolRange"

    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range計算"""
        if len(high) == 0:
            return np.array([])
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr2[0] = tr1[0]  # 最初の値は前日終値がないのでtr1を使用
        tr3[0] = tr1[0]
        
        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _calculate_volatility_persistence(self, returns: np.ndarray) -> float:
        """ボラティリティ持続性計算"""
        if len(returns) < 10:
            return 0.5
        
        # 簡易GARCH効果の測定
        squared_returns = returns**2
        volatility_series = pd.Series(squared_returns).rolling(window=5).std().dropna()
        
        if len(volatility_series) < 5:
            return 0.5
        
        # ボラティリティクラスタリングの測定
        vol_changes = np.diff(volatility_series)
        persistence = np.corrcoef(vol_changes[:-1], vol_changes[1:])[0, 1]
        
        return max(0, min(1, (persistence + 1) / 2))  # -1,1 を 0,1 に変換

    def _calculate_vix_like_index(self, returns: np.ndarray) -> float:
        """VIX様指数計算"""
        if len(returns) < 20:
            return 0.5
        
        # 近似的なVIX計算
        recent_vol = np.std(returns[-20:]) * np.sqrt(252)  # 年率化
        long_term_vol = np.std(returns) * np.sqrt(252)
        
        vix_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1
        
        return min(1.0, vix_ratio)

    def _calculate_dynamic_volatility_threshold(self, data: pd.DataFrame) -> float:
        """動的ボラティリティ閾値計算"""
        # 過去データからパーセンタイルベースの閾値を計算
        close_prices = data['close'].values
        
        if len(close_prices) < 50:
            return 1.2  # デフォルト閾値
        
        # ローリングボラティリティ計算
        window = 21
        rolling_vols = []
        
        for i in range(window, len(close_prices)):
            period_returns = np.diff(close_prices[i-window:i]) / close_prices[i-window:i-1]
            vol = np.std(period_returns)
            rolling_vols.append(vol)
        
        if rolling_vols:
            # 70パーセンタイルを高ボラティリティの閾値とする
            threshold = np.percentile(rolling_vols, 70)
            current_vol = rolling_vols[-1] if rolling_vols else 0
            return (threshold / current_vol) if current_vol > 0 else 1.2
        
        return 1.2

    def _calculate_dynamic_trend_threshold(self, data: pd.DataFrame) -> float:
        """動的トレンド閾値計算"""
        close_prices = data['close'].values
        
        if len(close_prices) < 50:
            return 0.6  # デフォルト閾値
        
        # ローリング線形回帰強度計算
        window = 21
        rolling_trend_strengths = []
        
        for i in range(window, len(close_prices)):
            x = np.arange(window)
            y = close_prices[i-window:i]
            _, _, r_value, _, _ = stats.linregress(x, y)
            rolling_trend_strengths.append(abs(r_value))
        
        if rolling_trend_strengths:
            # 60パーセンタイルを強トレンドの閾値とする
            return np.percentile(rolling_trend_strengths, 60)
        
        return 0.6

    def _calculate_dmi_trend(self, data: pd.DataFrame) -> float:
        """DMI-based trend strength"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(high) < 14:
            return 0
        
        # True Range
        tr = self._calculate_true_range(high, low, close)
        
        # Directional Movement
        dm_plus = np.maximum(high[1:] - high[:-1], 0)
        dm_minus = np.maximum(low[:-1] - low[1:], 0)
        
        # 条件に基づく調整
        for i in range(len(dm_plus)):
            if dm_plus[i] <= dm_minus[i]:
                dm_plus[i] = 0
            if dm_minus[i] <= dm_plus[i]:
                dm_minus[i] = 0
        
        # Smoothed values (14-period)
        if len(tr) >= 14:
            tr_smooth = np.mean(tr[-14:])
            dm_plus_smooth = np.mean(dm_plus[-14:])
            dm_minus_smooth = np.mean(dm_minus[-14:])
            
            di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
            return dx
        
        return 0

    def _calculate_ma_trend_strength(self, prices: np.ndarray) -> float:
        """Moving Average trend strength"""
        if len(prices) < 50:
            return 0
        
        # 複数期間の移動平均
        ma_periods = [5, 10, 20, 50]
        mas = []
        
        for period in ma_periods:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                mas.append(ma)
        
        if len(mas) < 2:
            return 0
        
        # 移動平均の順序チェック
        ascending = all(mas[i] <= mas[i+1] for i in range(len(mas)-1))
        descending = all(mas[i] >= mas[i+1] for i in range(len(mas)-1))
        
        if ascending or descending:
            # 現在価格と移動平均の乖離度
            current_price = prices[-1]
            ma_spread = abs(current_price - mas[0]) / current_price if current_price > 0 else 0
            return min(1.0, ma_spread * 10)
        
        return 0

    def _calculate_channel_trend(self, data: pd.DataFrame) -> float:
        """Price channel breakout trend strength"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        if len(close) < 20:
            return 0
        
        # 20期間チャネル
        period = 20
        recent_high = np.max(high[-period:])
        recent_low = np.min(low[-period:])
        current_price = close[-1]
        
        channel_range = recent_high - recent_low
        if channel_range == 0:
            return 0
        
        # チャネル内での位置
        position_in_channel = (current_price - recent_low) / channel_range
        
        # 上下ブレイクアウトの強度
        if current_price > recent_high:
            breakout_strength = (current_price - recent_high) / channel_range
            return min(1.0, breakout_strength * 2)
        elif current_price < recent_low:
            breakout_strength = (recent_low - current_price) / channel_range
            return min(1.0, breakout_strength * 2)
        
        # チャネル内では位置に基づく弱いトレンド
        return abs(position_in_channel - 0.5) * 0.5

    def _calculate_momentum_consistency(self, prices: np.ndarray) -> float:
        """Momentum consistency across timeframes"""
        if len(prices) < 21:
            return 0
        
        # 複数期間のROC
        periods = [3, 5, 10, 15, 20]
        rocs = []
        
        for period in periods:
            if len(prices) >= period + 1:
                roc = (prices[-1] - prices[-period-1]) / prices[-period-1]
                rocs.append(roc)
        
        if not rocs:
            return 0
        
        # 方向の一貫性
        positive_count = sum(1 for roc in rocs if roc > 0)
        consistency = abs(positive_count / len(rocs) - 0.5) * 2
        
        # 強度の平均
        avg_strength = np.mean([abs(roc) for roc in rocs])
        
        return min(1.0, consistency * avg_strength * 100)

    # 簡略化された手法分析メソッド群
    def _wavelet_fourier_4state_analysis(self, data: pd.DataFrame) -> Dict:
        close_prices = data['close'].values
        coeffs = pywt.wavedec(close_prices, 'db4', level=4)
        trend_energy = np.sum(np.abs(coeffs[0])**2)
        noise_energy = sum(np.sum(np.abs(c)**2) for c in coeffs[1:3])
        trend_ratio = trend_energy / (trend_energy + noise_energy) if (trend_energy + noise_energy) > 0 else 0
        
        # ボラティリティ分析
        detail_variance = np.var(coeffs[1]) if len(coeffs[1]) > 0 else 0
        vol_level = min(1.0, detail_variance * 1000)
        
        return {
            'trend_pattern': 'Trend' if trend_ratio > 0.6 else 'Range',
            'vol_pattern': 'High' if vol_level > 0.5 else 'Low',
            'trend_strength': trend_ratio,
            'vol_strength': vol_level
        }

    def _fractal_4state_analysis(self, data: pd.DataFrame) -> Dict:
        # 簡略化されたフラクタル分析
        close_prices = data['close'].values[-50:] if len(data) >= 50 else data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else np.array([0])
        
        trend_strength = abs(np.mean(returns)) * 100 if len(returns) > 0 else 0
        vol_strength = np.std(returns) * 10 if len(returns) > 0 else 0
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': min(1.0, trend_strength),
            'vol_strength': min(1.0, vol_strength)
        }

    def _entropy_4state_analysis(self, data: pd.DataFrame) -> Dict:
        returns = data['close'].pct_change().dropna().values[-30:] if len(data) >= 30 else data['close'].pct_change().dropna().values
        
        if len(returns) == 0:
            return {'trend_pattern': 'Range', 'vol_pattern': 'Low', 'trend_strength': 0, 'vol_strength': 0}
        
        # エントロピー計算
        bins = max(5, len(returns) // 5)
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) if len(hist) > 0 else 0
        
        # 正規化
        max_entropy = np.log2(len(hist)) if len(hist) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        trend_strength = 1 - normalized_entropy  # 低エントロピー = 高トレンド
        vol_strength = np.std(returns) * 20
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': trend_strength,
            'vol_strength': min(1.0, vol_strength)
        }

    def _momentum_4state_analysis(self, data: pd.DataFrame) -> Dict:
        close_prices = data['close'].values
        
        # モメンタム計算
        momentum_scores = []
        for period in [5, 10, 20]:
            if len(close_prices) >= period + 1:
                momentum = (close_prices[-1] - close_prices[-period-1]) / close_prices[-period-1]
                momentum_scores.append(abs(momentum))
        
        trend_strength = np.mean(momentum_scores) * 20 if momentum_scores else 0
        
        # ボラティリティ（価格変動）
        if len(close_prices) >= 10:
            recent_vol = np.std(close_prices[-10:]) / np.mean(close_prices[-10:])
            vol_strength = recent_vol * 50
        else:
            vol_strength = 0
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': min(1.0, trend_strength),
            'vol_strength': min(1.0, vol_strength)
        }

    def _sr_4state_analysis(self, data: pd.DataFrame) -> Dict:
        # サポート/レジスタンス分析
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        if len(close_prices) < 10:
            return {'trend_pattern': 'Range', 'vol_pattern': 'Low', 'trend_strength': 0, 'vol_strength': 0}
        
        # レンジ幅
        recent_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:]) if len(high_prices) >= 20 else np.max(high_prices) - np.min(low_prices)
        avg_price = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.mean(close_prices)
        range_ratio = recent_range / avg_price if avg_price > 0 else 0
        
        trend_strength = min(1.0, range_ratio * 5)  # 大きなレンジ = トレンド可能性
        vol_strength = min(1.0, range_ratio * 10)   # レンジ幅 = ボラティリティ
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': trend_strength,
            'vol_strength': vol_strength
        }

    def _changepoint_4state_analysis(self, data: pd.DataFrame) -> Dict:
        # 変化点分析
        close_prices = data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else np.array([0])
        
        if len(returns) < 10:
            return {'trend_pattern': 'Range', 'vol_pattern': 'Low', 'trend_strength': 0, 'vol_strength': 0}
        
        # 変化点密度（簡易版）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            outliers = np.abs(returns - mean_return) > 2 * std_return
            change_density = np.sum(outliers) / len(returns)
        else:
            change_density = 0
        
        trend_strength = 1 - change_density  # 変化点が少ない = 安定トレンド
        vol_strength = change_density         # 変化点が多い = 高ボラティリティ
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': trend_strength,
            'vol_strength': vol_strength
        }

    def _kalman_4state_analysis(self, data: pd.DataFrame) -> Dict:
        # カルマンフィルター分析（簡易版）
        close_prices = data['close'].values
        
        if len(close_prices) < 10:
            return {'trend_pattern': 'Range', 'vol_pattern': 'Low', 'trend_strength': 0, 'vol_strength': 0}
        
        # 線形回帰
        x = np.arange(len(close_prices))
        slope, _, r_value, _, _ = stats.linregress(x, close_prices)
        
        trend_strength = abs(r_value)
        
        # ノイズレベル
        fitted = slope * x + close_prices[0]
        residuals = close_prices - fitted
        noise_level = np.std(residuals) / np.mean(close_prices) if np.mean(close_prices) > 0 else 0
        vol_strength = min(1.0, noise_level * 50)
        
        return {
            'trend_pattern': 'Trend' if trend_strength > 0.5 else 'Range',
            'vol_pattern': 'High' if vol_strength > 0.5 else 'Low',
            'trend_strength': trend_strength,
            'vol_strength': vol_strength
        }

    def _calculate_classification_confidence(self, method_results: Dict, 
                                           trend_analysis: Dict, 
                                           volatility_analysis: Dict) -> float:
        """分類信頼度計算"""
        # 手法間の一致度
        trend_votes = sum(1 for result in method_results.values() if result['trend_pattern'] == ('Trend' if trend_analysis['is_trending'] else 'Range'))
        vol_votes = sum(1 for result in method_results.values() if result['vol_pattern'] == ('High' if volatility_analysis['is_high'] else 'Low'))
        
        trend_consensus = trend_votes / len(method_results) if method_results else 0
        vol_consensus = vol_votes / len(method_results) if method_results else 0
        
        # 統合信頼度
        confidence = (trend_consensus + vol_consensus) / 2
        
        return min(0.95, max(0.05, confidence))

    def _calculate_stability_score(self, data: pd.DataFrame) -> float:
        """安定性スコア計算"""
        close_prices = data['close'].values
        
        if len(close_prices) < 20:
            return 0.5
        
        # 複数期間での変動係数
        stability_scores = []
        for window in [10, 20, 50]:
            if len(close_prices) >= window:
                recent_prices = close_prices[-window:]
                cv = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 1
                stability = 1 / (1 + cv * 5)
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.5


# デモンストレーション
if __name__ == "__main__":
    print("🌌 Cosmos Supreme 4-State Market Classifier デモンストレーション")
    
    # サンプルデータ生成
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    
    # 高ボラトレンドのサンプルデータ
    trend = np.cumsum(np.random.normal(0.002, 0.03, 300))  # 強いトレンド + 高ボラ
    noise = np.random.normal(0, 0.02, 300)
    close_prices = 100 + trend + noise
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': close_prices + np.random.normal(0, 0.5, 300),
        'high': close_prices + np.abs(np.random.normal(0, 1.5, 300)),
        'low': close_prices - np.abs(np.random.normal(0, 1.5, 300)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 300)
    })
    
    # 分類器の初期化と実行
    classifier = CosmosSupreme4StateClassifier()
    result = classifier.classify(sample_data)
    
    print(f"\n🎯 分類結果:")
    print(f"市場状態: {result.market_state}")
    print(f"トレンド強度: {result.trend_strength:.3f}")
    print(f"ボラティリティレベル: {result.volatility_level:.3f}")
    print(f"分類信頼度: {result.confidence:.3f} ({result.confidence*100:.1f}%)")
    print(f"安定性: {result.stability:.3f}")
    
    print(f"\n🔬 手法別判定:")
    for method, patterns in result.method_consensus.items():
        print(f"  {method}: Trend={patterns['trend']}, Vol={patterns['volatility']}")
    
    print(f"\n📊 ボラティリティ指標:")
    for metric, value in result.volatility_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n📈 トレンド指標:")
    for metric, value in result.trend_metrics.items():
        print(f"  {metric}: {value:.4f}") 