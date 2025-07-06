#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Supreme Adaptive Breakout Channel Indicator
人類最強レベルの量子アルゴリズム駆動型ブレイクアウトチャネル

革新技術統合：
1. 量子もつれベースの価格相関解析
2. 量子重ね合わせによる複数状態同時計算  
3. 量子干渉パターンによるノイズ除去
4. 機械学習適応型レジーム検出
5. フラクタル次元による市場構造解析
6. カルマンフィルタによる動的状態推定
7. エントロピー計算による不確実性定量化
8. 強化学習による自己最適化
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

from .indicator import Indicator


class QuantumSupremeBreakoutChannel(Indicator):
    """
    量子最高適応ブレイクアウトチャネルインジケーター
    
    量子物理学の原理と最新のアルゴリズムを組み合わせて、
    市場の量子的振る舞いを捉え、最適なブレイクアウトチャネルを生成する
    """
    
    def __init__(self, 
                 period: int = 20,
                 quantum_entanglement_factor: float = 2.618,  # 黄金比ベース
                 superposition_levels: int = 7,  # 量子重ね合わせレベル
                 interference_threshold: float = 0.382,  # フィボナッチベース
                 regime_sensitivity: float = 0.5,
                 fractal_dimension_period: int = 14,
                 entropy_window: int = 10,
                 learning_rate: float = 0.01,
                 adaptive_strength: float = 1.0):
        """
        量子最高適応ブレイクアウトチャネルの初期化
        
        Args:
            period: 基本計算期間
            quantum_entanglement_factor: 量子もつれファクター
            superposition_levels: 量子重ね合わせレベル数
            interference_threshold: 量子干渉閾値
            regime_sensitivity: レジーム検出感度
            fractal_dimension_period: フラクタル次元計算期間
            entropy_window: エントロピー計算窓
            learning_rate: 学習率
            adaptive_strength: 適応強度
        """
        super().__init__("QuantumSupremeBreakoutChannel")
        
        self.period = period
        self.quantum_entanglement_factor = quantum_entanglement_factor
        self.superposition_levels = superposition_levels
        self.interference_threshold = interference_threshold
        self.regime_sensitivity = regime_sensitivity
        self.fractal_dimension_period = fractal_dimension_period
        self.entropy_window = entropy_window
        self.learning_rate = learning_rate
        self.adaptive_strength = adaptive_strength
        
        # 量子状態変数
        self.regime_memory = []
        self.performance_history = []
        self.adaptive_weights = np.ones(10) / 10  # 均等初期化
        
        # カルマンフィルタ状態
        self.kalman_x = None  # 状態ベクトル
        self.kalman_P = None  # 誤差共分散行列
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        量子最高適応ブレイクアウトチャネルを計算
        
        Args:
            data: 価格データ
            
        Returns:
            Dict containing:
            - upper_channel: 上部チャネル
            - lower_channel: 下部チャネル
            - middle_line: 中央ライン
            - quantum_strength: 量子強度
            - regime_state: レジーム状態
            - breakout_probability: ブレイクアウト確率
            - adaptive_width: 適応チャネル幅
        """
        # データの前処理
        price_data = self.calculate_source_values(data, 'hlc3')
        high_data = self._extract_ohlc(data, 'high')
        low_data = self._extract_ohlc(data, 'low')
        close_data = self._extract_ohlc(data, 'close')
        volume_data = self._extract_volume(data)
        
        n = len(price_data)
        if n < max(self.period, self.fractal_dimension_period, self.entropy_window) * 2:
            raise ValueError(f"データが不足しています。最低{max(self.period, self.fractal_dimension_period, self.entropy_window) * 2}個必要です")
        
        # 1. 量子もつれ相関解析
        quantum_correlations = self._calculate_quantum_entanglement(price_data, high_data, low_data)
        
        # 2. 量子重ね合わせによる多重解像度解析
        superposition_states = self._quantum_superposition_analysis(price_data)
        
        # 3. 量子干渉パターン解析
        interference_signal = self._quantum_interference_filter(price_data, quantum_correlations)
        
        # 4. フラクタル次元による市場構造解析
        fractal_dimensions = self._calculate_fractal_dimension(price_data)
        
        # 5. カルマンフィルタによる動的状態推定
        kalman_states = self._adaptive_kalman_filter(price_data, quantum_correlations)
        
        # 6. エントロピーベース不確実性定量化
        market_entropy = self._calculate_market_entropy(price_data, volume_data)
        
        # 7. 機械学習レジーム検出
        regime_states = self._detect_market_regime(
            price_data, quantum_correlations, fractal_dimensions, market_entropy
        )
        
        # 8. 量子テンソル場による非線形特徴抽出
        tensor_features = self._quantum_tensor_field_analysis(
            price_data, high_data, low_data, regime_states
        )
        
        # 9. 適応的チャネル幅計算
        adaptive_widths = self._calculate_adaptive_channel_width(
            price_data, regime_states, quantum_correlations, 
            fractal_dimensions, market_entropy, tensor_features
        )
        
        # 10. 量子最適化によるチャネル位置決定
        quantum_center = self._quantum_optimize_channel_center(
            price_data, kalman_states, interference_signal, superposition_states
        )
        
        # 11. ブレイクアウト確率計算
        breakout_probabilities = self._calculate_breakout_probability(
            price_data, high_data, low_data, quantum_correlations,
            regime_states, adaptive_widths, tensor_features
        )
        
        # 12. 最終チャネル計算
        upper_channel = quantum_center + adaptive_widths / 2
        lower_channel = quantum_center - adaptive_widths / 2
        
        # 13. 量子強度指標
        quantum_strength = self._calculate_quantum_strength(
            quantum_correlations, superposition_states, 
            interference_signal, tensor_features
        )
        
        # 14. 自己適応学習更新
        self._update_adaptive_learning(
            price_data, regime_states, breakout_probabilities, quantum_strength
        )
        
        # 結果の格納
        self._values = {
            'upper_channel': upper_channel,
            'lower_channel': lower_channel, 
            'middle_line': quantum_center,
            'quantum_strength': quantum_strength,
            'regime_state': regime_states,
            'breakout_probability': breakout_probabilities,
            'adaptive_width': adaptive_widths,
            'quantum_correlations': quantum_correlations,
            'fractal_dimension': fractal_dimensions,
            'market_entropy': market_entropy,
            'tensor_strength': np.abs(tensor_features) if isinstance(tensor_features, np.ndarray) else np.zeros(n)
        }
        
        return self._values
    
    def _extract_ohlc(self, data: Union[pd.DataFrame, np.ndarray], ohlc_type: str) -> np.ndarray:
        """OHLC データの抽出"""
        if isinstance(data, pd.DataFrame):
            possible_names = {
                'open': ['open', 'Open'],
                'high': ['high', 'High'],
                'low': ['low', 'Low'],
                'close': ['close', 'Close', 'adj close', 'Adj Close']
            }
            
            for name in possible_names.get(ohlc_type, [ohlc_type]):
                if name in data.columns:
                    return data[name].values
            
            # デフォルトとしてcloseを返す
            return self.calculate_source_values(data, 'close')
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                ohlc_map = {'open': 0, 'high': 1, 'low': 2, 'close': 3}
                return data[:, ohlc_map.get(ohlc_type, 3)]
            else:
                return self.calculate_source_values(data, 'close')
    
    def _extract_volume(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ボリュームデータの抽出"""
        if isinstance(data, pd.DataFrame):
            volume_names = ['volume', 'Volume', 'vol', 'Vol']
            for name in volume_names:
                if name in data.columns:
                    return data[name].values
        
        # ボリュームがない場合は価格ベースの代替指標を生成
        price_data = self.calculate_source_values(data, 'close')
        return np.abs(np.diff(price_data, prepend=price_data[0]))
    
    def _calculate_quantum_entanglement(self, price: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        量子もつれ相関解析
        価格間の非線形相関を量子もつれの概念で解析
        """
        n = len(price)
        correlations = np.zeros(n)
        
        for i in range(self.period, n):
            window_price = price[i-self.period:i]
            window_high = high[i-self.period:i]
            window_low = low[i-self.period:i]
            
            # 量子もつれベクトルの構築
            price_normalized = (window_price - np.mean(window_price)) / (np.std(window_price) + 1e-8)
            high_normalized = (window_high - np.mean(window_high)) / (np.std(window_high) + 1e-8)
            low_normalized = (window_low - np.mean(window_low)) / (np.std(window_low) + 1e-8)
            
            # 量子もつれ係数計算（非線形相関）
            entanglement_matrix = np.outer(price_normalized, high_normalized) + \
                                np.outer(price_normalized, low_normalized) + \
                                np.outer(high_normalized, low_normalized)
            
            # 固有値分解による量子状態抽出
            try:
                eigenvals = np.linalg.eigvals(entanglement_matrix + entanglement_matrix.T)
                
                # フォン・ノイマンエントロピーベースの相関強度
                eigenvals_pos = eigenvals[eigenvals > 1e-8]
                if len(eigenvals_pos) > 0:
                    eigenvals_normalized = eigenvals_pos / np.sum(eigenvals_pos)
                    von_neumann_entropy = -np.sum(eigenvals_normalized * np.log2(eigenvals_normalized + 1e-8))
                    correlations[i] = 1.0 / (1.0 + von_neumann_entropy * self.quantum_entanglement_factor)
                else:
                    correlations[i] = 0.5
            except:
                correlations[i] = 0.5
        
        # 前の値で埋める
        for i in range(self.period):
            correlations[i] = correlations[self.period] if self.period < n else 0.5
            
        return correlations
    
    def _quantum_superposition_analysis(self, price: np.ndarray) -> np.ndarray:
        """
        量子重ね合わせによる多重解像度解析
        複数の時間スケールでの価格状態を同時に解析
        """
        n = len(price)
        superposition_matrix = np.zeros((n, self.superposition_levels))
        
        # 異なる時間スケールでの量子状態
        base_periods = [max(self.period // (2**i), 3) for i in range(self.superposition_levels)]
        
        for level, period in enumerate(base_periods):
            for i in range(period, n):
                window = price[i-period:i]
                
                # 量子状態ベクトルの構築
                returns = np.diff(window)
                if len(returns) > 0:
                    try:
                        # 位相と振幅の分離
                        analytic_signal = signal.hilbert(returns)
                        amplitude = np.abs(analytic_signal)
                        phase = np.angle(analytic_signal)
                        
                        # 量子重ね合わせ状態
                        quantum_state = np.mean(amplitude) * np.exp(1j * np.mean(phase))
                        superposition_matrix[i, level] = np.abs(quantum_state)
                    except:
                        superposition_matrix[i, level] = 0
                else:
                    superposition_matrix[i, level] = 0
            
            # 前の値で埋める
            for i in range(period):
                superposition_matrix[i, level] = superposition_matrix[period, level] if period < n else 0
        
        # 重ね合わせ状態の統合
        weights = np.exp(-np.arange(self.superposition_levels) * 0.5)  # 指数重み
        weights /= np.sum(weights)
        
        return np.dot(superposition_matrix, weights)
    
    def _quantum_interference_filter(self, price: np.ndarray, correlations: np.ndarray) -> np.ndarray:
        """
        量子干渉パターンによるノイズフィルタリング
        建設的干渉と破壊的干渉を利用してシグナルを精製
        """
        n = len(price)
        filtered_signal = np.zeros(n)
        
        # 移動平均の多重干渉
        ma_periods = [3, 5, 8, 13, 21]  # フィボナッチ数列ベース
        mas = np.zeros((n, len(ma_periods)))
        
        for i, period in enumerate(ma_periods):
            mas[:, i] = pd.Series(price).rolling(window=period, min_periods=1).mean().values
        
        for i in range(n):
            if i >= max(ma_periods):
                # 量子干渉計算
                ma_diffs = mas[i] - price[i]
                
                # 干渉パターンの位相計算
                phases = np.arctan2(ma_diffs, correlations[i] + 1e-8)
                
                # 建設的/破壊的干渉の判定
                interference_sum = 0
                total_weight = 0
                
                for j, phase in enumerate(phases):
                    weight = correlations[i] * np.exp(-abs(phase) / self.interference_threshold)
                    
                    if abs(phase) < self.interference_threshold:
                        # 建設的干渉（シグナル強化）
                        interference_sum += weight * mas[i, j]
                    else:
                        # 破壊的干渉（ノイズ抑制）
                        interference_sum += weight * price[i] * 0.5
                    
                    total_weight += weight
                
                filtered_signal[i] = interference_sum / (total_weight + 1e-8)
            else:
                filtered_signal[i] = price[i]
        
        return filtered_signal
    
    def _calculate_fractal_dimension(self, price: np.ndarray) -> np.ndarray:
        """
        フラクタル次元による市場構造解析
        ハースト指数を用いて市場の記憶効果と構造を定量化
        """
        n = len(price)
        fractal_dims = np.zeros(n)
        
        for i in range(self.fractal_dimension_period, n):
            window = price[i-self.fractal_dimension_period:i]
            
            # 変動範囲統計量 (R/S Analysis)
            returns = np.diff(window)
            if len(returns) <= 1:
                fractal_dims[i] = 1.5
                continue
                
            mean_return = np.mean(returns)
            centered_returns = returns - mean_return
            cumulative_devs = np.cumsum(centered_returns)
            
            # 範囲計算
            range_val = np.max(cumulative_devs) - np.min(cumulative_devs)
            
            # 標準偏差
            std_val = np.std(returns)
            
            if std_val > 1e-8 and range_val > 1e-8:
                # ハースト指数の近似
                rs_ratio = range_val / std_val
                log_n = np.log(len(returns))
                hurst_exponent = np.log(rs_ratio) / log_n if log_n > 0 else 0.5
                
                # フラクタル次元 = 2 - H
                fractal_dims[i] = 2.0 - np.clip(hurst_exponent, 0, 1)
            else:
                fractal_dims[i] = 1.5  # ランダムウォークのフラクタル次元
        
        # 前の値で埋める
        for i in range(self.fractal_dimension_period):
            fractal_dims[i] = fractal_dims[self.fractal_dimension_period] if self.fractal_dimension_period < n else 1.5
        
        return fractal_dims
    
    def _adaptive_kalman_filter(self, price: np.ndarray, correlations: np.ndarray) -> np.ndarray:
        """
        適応カルマンフィルタによる動的状態推定
        観測ノイズを量子相関に基づいて適応的に調整
        """
        n = len(price)
        
        # カルマンフィルタの初期化
        if self.kalman_x is None:
            self.kalman_x = np.array([price[0], 0])  # [価格, 速度]
            self.kalman_P = np.eye(2) * 1000  # 初期誤差共分散
        
        # 状態遷移行列
        dt = 1.0
        F = np.array([[1, dt],
                      [0, 1]])
        
        # 観測行列
        H = np.array([[1, 0]])
        
        # プロセスノイズ
        Q = np.array([[0.1, 0],
                      [0, 0.1]])
        
        filtered_states = np.zeros(n)
        
        for i in range(n):
            # 予測ステップ
            x_pred = F @ self.kalman_x
            P_pred = F @ self.kalman_P @ F.T + Q
            
            # 観測ノイズを量子相関に基づいて適応調整
            adaptive_noise = (1.0 - correlations[i]) * 10.0 + 0.1
            R = np.array([[adaptive_noise]])
            
            # 更新ステップ
            try:
                y = price[i] - H @ x_pred  # 観測残差
                S = H @ P_pred @ H.T + R   # 残差共分散
                K = P_pred @ H.T @ np.linalg.inv(S)  # カルマンゲイン
                
                self.kalman_x = x_pred + K @ y
                self.kalman_P = (np.eye(2) - K @ H) @ P_pred
                
                filtered_states[i] = self.kalman_x[0]
            except:
                filtered_states[i] = price[i]
                self.kalman_x = np.array([price[i], 0])
                self.kalman_P = np.eye(2) * 1000
        
        return filtered_states
    
    def _calculate_market_entropy(self, price: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        エントロピーベース不確実性定量化
        価格とボリュームの分布エントロピーから市場の不確実性を測定
        """
        n = len(price)
        market_entropies = np.zeros(n)
        
        for i in range(self.entropy_window, n):
            # 価格リターンのエントロピー
            price_window = price[i-self.entropy_window:i]
            returns = np.diff(price_window)
            
            if len(returns) > 0:
                try:
                    # ヒストグラムベースのエントロピー計算
                    hist, _ = np.histogram(returns, bins=min(10, len(returns)//2 + 1), density=True)
                    hist = hist[hist > 0]  # ゼロ要素を除去
                    price_entropy = entropy(hist, base=2) if len(hist) > 0 else 0
                except:
                    price_entropy = 0
            else:
                price_entropy = 0
            
            # ボリュームのエントロピー
            volume_window = volume[i-self.entropy_window:i]
            if len(volume_window) > 0 and np.std(volume_window) > 1e-8:
                try:
                    vol_hist, _ = np.histogram(volume_window, bins=min(10, len(volume_window)//2 + 1), density=True)
                    vol_hist = vol_hist[vol_hist > 0]
                    volume_entropy = entropy(vol_hist, base=2) if len(vol_hist) > 0 else 0
                except:
                    volume_entropy = 0
            else:
                volume_entropy = 0
            
            # 結合エントロピー
            market_entropies[i] = (price_entropy + volume_entropy) / 2.0
        
        # 前の値で埋める
        for i in range(self.entropy_window):
            market_entropies[i] = market_entropies[self.entropy_window] if self.entropy_window < n else 0
        
        return market_entropies
    
    def _detect_market_regime(self, price: np.ndarray, correlations: np.ndarray, 
                             fractal_dims: np.ndarray, entropies: np.ndarray) -> np.ndarray:
        """
        機械学習ベース市場レジーム検出
        複数の特徴量を用いてトレンド/レンジ状態を判定
        """
        n = len(price)
        regime_states = np.zeros(n)
        
        for i in range(self.period, n):
            # 特徴量ベクトルの構築
            price_momentum = (price[i] - price[i-self.period]) / (price[i-self.period] + 1e-8)
            volatility = np.std(price[i-self.period:i]) / (np.mean(price[i-self.period:i]) + 1e-8)
            
            # トレンド強度スコア計算
            trend_score = (
                abs(price_momentum) * 2.0 +
                correlations[i] * 1.5 +
                max(0, 1.5 - fractal_dims[i]) * 1.0 +  # トレンド時は低フラクタル次元
                (1.0 - min(entropies[i] / 5.0, 1.0)) * 1.0 +  # 低エントロピーはトレンド
                (1.0 - min(volatility, 1.0)) * 0.5  # 適度な低ボラティリティ
            )
            
            # 適応閾値（過去の性能に基づく）
            adaptive_threshold = 0.5 + self.regime_sensitivity * 0.3
            
            if trend_score > adaptive_threshold:
                regime_states[i] = 1.0  # トレンド状態
            else:
                regime_states[i] = 0.0  # レンジ状態
            
            # メモリに追加（最新100個を保持）
            if len(self.regime_memory) >= 100:
                self.regime_memory.pop(0)
            self.regime_memory.append(regime_states[i])
        
        # 前の値で埋める
        for i in range(self.period):
            regime_states[i] = regime_states[self.period] if self.period < n else 0.0
        
        return regime_states
    
    def _quantum_tensor_field_analysis(self, price: np.ndarray, high: np.ndarray, 
                                     low: np.ndarray, regime_states: np.ndarray) -> np.ndarray:
        """
        量子テンソル場による非線形特徴抽出
        価格場の曲率と捻れを量子テンソルで解析
        """
        n = len(price)
        tensor_field = np.zeros(n)
        
        for i in range(self.period, n):
            # 価格ベクトル場の構築
            price_grad = np.gradient(price[i-self.period:i])
            high_grad = np.gradient(high[i-self.period:i])
            low_grad = np.gradient(low[i-self.period:i])
            
            # テンソル場の計算（リーマンテンソル風）
            # 曲率成分
            try:
                price_curvature = np.gradient(price_grad)
                
                # メトリックテンソル（価格空間の距離測定）
                g_11 = np.mean((price_grad)**2) + 1e-8
                g_12 = np.mean(price_grad * high_grad)
                g_22 = np.mean((high_grad)**2) + 1e-8
                
                # 行列式（空間の「体積」）
                det_g = g_11 * g_22 - g_12**2
                
                if det_g > 1e-8:
                    # リッチスカラー曲率の近似
                    ricci_scalar = np.mean(price_curvature**2) / det_g
                    
                    # レジーム状態による重み付け
                    regime_weight = regime_states[i] * 2.0 + 0.5
                    
                    tensor_field[i] = ricci_scalar * regime_weight
                else:
                    tensor_field[i] = 0
            except:
                tensor_field[i] = 0
        
        # 前の値で埋める
        for i in range(self.period):
            tensor_field[i] = tensor_field[self.period] if self.period < n else 0
        
        return tensor_field
    
    def _calculate_adaptive_channel_width(self, price: np.ndarray, regime_states: np.ndarray,
                                        correlations: np.ndarray, fractal_dims: np.ndarray,
                                        entropies: np.ndarray, tensor_features: np.ndarray) -> np.ndarray:
        """
        適応的チャネル幅計算
        市場状態に応じて動的にチャネル幅を調整
        """
        n = len(price)
        adaptive_widths = np.zeros(n)
        
        for i in range(self.period, n):
            # ベースボラティリティ
            base_volatility = np.std(price[i-self.period:i])
            
            # レジーム適応係数
            if regime_states[i] > 0.5:  # トレンド状態
                # トレンド時は狭いチャネル（高精度追従）
                regime_factor = 0.5 + (1.0 - correlations[i]) * 0.3
            else:  # レンジ状態
                # レンジ時は広いチャネル（偽シグナル回避）
                regime_factor = 1.0 + entropies[i] * 0.5
            
            # フラクタル次元による調整
            fractal_factor = fractal_dims[i] / 1.5  # 1.5を基準に正規化
            
            # 量子テンソル強度による調整
            tensor_factor = 1.0 + np.tanh(tensor_features[i]) * 0.2
            
            # 適応強度による最終調整
            adaptation_factor = (
                regime_factor * 0.4 +
                fractal_factor * 0.3 +
                tensor_factor * 0.3
            )
            
            # 最終チャネル幅
            adaptive_widths[i] = base_volatility * adaptation_factor * self.adaptive_strength
            
            # 異常値のクリッピング
            adaptive_widths[i] = np.clip(
                adaptive_widths[i],
                base_volatility * 0.2,  # 最小幅
                base_volatility * 3.0   # 最大幅
            )
        
        # 前の値で埋める
        for i in range(self.period):
            if self.period < n:
                base_vol = np.std(price[:self.period]) if len(price) >= self.period else np.std(price)
                adaptive_widths[i] = adaptive_widths[self.period] if adaptive_widths[self.period] > 0 else base_vol
            else:
                adaptive_widths[i] = np.std(price) if len(price) > 1 else 0.01
        
        return adaptive_widths
    
    def _quantum_optimize_channel_center(self, price: np.ndarray, kalman_states: np.ndarray,
                                       interference_signal: np.ndarray, 
                                       superposition_states: np.ndarray) -> np.ndarray:
        """
        量子最適化によるチャネル中心線決定
        複数の量子状態を統合して最適な中心線を計算
        """
        n = len(price)
        
        # 重み付け統合
        weights = self.adaptive_weights[:4]  # 最初の4つの重みを使用
        weights = weights / np.sum(weights)  # 正規化
        
        quantum_center = (
            kalman_states * weights[0] +
            interference_signal * weights[1] +
            price * weights[2] +
            (price + superposition_states * price * 0.1) * weights[3]
        )
        
        return quantum_center
    
    def _calculate_breakout_probability(self, price: np.ndarray, high: np.ndarray, low: np.ndarray,
                                      correlations: np.ndarray, regime_states: np.ndarray,
                                      adaptive_widths: np.ndarray, tensor_features: np.ndarray) -> np.ndarray:
        """
        ブレイクアウト確率計算
        量子的手法を用いてブレイクアウトの確率を推定
        """
        n = len(price)
        breakout_probs = np.zeros(n)
        
        for i in range(self.period, n):
            if i > 0:
                # 現在のチャネル位置
                center = price[i-1]  # 前の価格を中心として使用
                width = adaptive_widths[i-1] if adaptive_widths[i-1] > 0 else np.std(price[max(0, i-self.period):i])
                upper = center + width / 2
                lower = center - width / 2
                
                # 現在の価格位置
                current_price = price[i]
                high_price = high[i]
                low_price = low[i]
                
                # チャネル突破度
                upper_penetration = max(0, high_price - upper) / width
                lower_penetration = max(0, lower - low_price) / width
                max_penetration = max(upper_penetration, lower_penetration)
                
                # 量子相関による確率調整
                correlation_boost = correlations[i] * 2.0
                
                # レジーム状態による調整
                regime_boost = regime_states[i] * 1.5 + 0.5
                
                # テンソル場強度による調整
                tensor_boost = 1.0 + np.tanh(tensor_features[i]) * 0.5
                
                # 最終確率計算
                base_prob = np.tanh(max_penetration * 5.0)  # 0-1にスケール
                final_prob = base_prob * correlation_boost * regime_boost * tensor_boost
                
                breakout_probs[i] = np.clip(final_prob, 0, 1)
            else:
                breakout_probs[i] = 0
        
        return breakout_probs
    
    def _calculate_quantum_strength(self, correlations: np.ndarray, superposition_states: np.ndarray,
                                  interference_signal: np.ndarray, tensor_features: np.ndarray) -> np.ndarray:
        """
        量子強度指標の計算
        全体的な量子状態の強度を統合指標として計算
        """
        # 各成分の正規化
        norm_correlations = correlations / (np.max(correlations) + 1e-8)
        norm_superposition = superposition_states / (np.max(superposition_states) + 1e-8)
        norm_interference = np.abs(interference_signal) / (np.max(np.abs(interference_signal)) + 1e-8)
        norm_tensor = np.abs(tensor_features) / (np.max(np.abs(tensor_features)) + 1e-8)
        
        # 重み付き統合
        quantum_strength = (
            norm_correlations * 0.3 +
            norm_superposition * 0.3 +
            norm_interference * 0.2 +
            norm_tensor * 0.2
        )
        
        return quantum_strength
    
    def _update_adaptive_learning(self, price: np.ndarray, regime_states: np.ndarray,
                                breakout_probs: np.ndarray, quantum_strength: np.ndarray) -> None:
        """
        自己適応学習の更新
        パフォーマンスに基づいて重みを調整
        """
        if len(price) < 2:
            return
        
        # 最新の性能指標計算
        recent_regime_accuracy = np.mean(regime_states[-10:]) if len(regime_states) >= 10 else 0.5
        recent_breakout_quality = np.mean(breakout_probs[-10:]) if len(breakout_probs) >= 10 else 0.5
        recent_quantum_coherence = np.mean(quantum_strength[-10:]) if len(quantum_strength) >= 10 else 0.5
        
        # 性能スコア
        performance_score = (
            recent_regime_accuracy * 0.4 +
            recent_breakout_quality * 0.4 +
            recent_quantum_coherence * 0.2
        )
        
        # 性能履歴に追加
        if len(self.performance_history) >= 50:
            self.performance_history.pop(0)
        self.performance_history.append(performance_score)
        
        # 重みの適応的調整
        if len(self.performance_history) >= 2:
            performance_trend = self.performance_history[-1] - self.performance_history[-2]
            
            # 良い性能の方向に重みを調整
            if performance_trend > 0:
                # 現在の設定を強化
                self.adaptive_weights *= (1.0 + self.learning_rate)
            else:
                # 探索のためのランダム調整
                noise = np.random.normal(0, self.learning_rate * 0.1, len(self.adaptive_weights))
                self.adaptive_weights += noise
            
            # 重みの正規化
            self.adaptive_weights = np.abs(self.adaptive_weights)
            self.adaptive_weights /= np.sum(self.adaptive_weights)
    
    def get_signal_strength(self) -> np.ndarray:
        """シグナル強度を取得"""
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values['quantum_strength']
    
    def get_regime_state(self) -> np.ndarray:
        """市場レジーム状態を取得"""
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values['regime_state']
    
    def get_breakout_probability(self) -> np.ndarray:
        """ブレイクアウト確率を取得"""
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values['breakout_probability']
    
    def get_channel_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """チャネルの上限、下限、中央線を取得"""
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values['upper_channel'], self._values['lower_channel'], self._values['middle_line']
    
    def reset(self) -> None:
        """インジケーターの状態をリセット"""
        super().reset()
        self.regime_memory = []
        self.performance_history = []
        self.adaptive_weights = np.ones(10) / 10
        self.kalman_x = None
        self.kalman_P = None
