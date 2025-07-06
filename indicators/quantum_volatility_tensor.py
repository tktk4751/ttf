#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union, Optional, NamedTuple, Tuple
from numba import jit
import traceback
import math
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource:
        @staticmethod
        def calculate_source(data, src_type):
            if isinstance(data, pd.DataFrame):
                if src_type == 'close': return data['close'].values
                elif src_type == 'open': return data['open'].values
                elif src_type == 'high': return data['high'].values
                elif src_type == 'low': return data['low'].values
                elif src_type == 'hl2': return ((data['high'] + data['low']) / 2).values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                elif src_type == 'ohlc4': return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
                else: return data['close'].values
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data


class QuantumVolatilityResult(NamedTuple):
    """量子ボラティリティテンソル計算結果"""
    quantum_volatility: np.ndarray      # 統合量子ボラティリティ
    directional_entropy: np.ndarray     # 方向性エントロピー
    fractal_dimension: np.ndarray       # フラクタル次元
    wavelet_energy: np.ndarray          # ウェーブレット エネルギー
    acceleration_tensor: np.ndarray     # 加速度テンソル
    uncertainty_principle: np.ndarray   # 不確定性原理値
    market_regime: np.ndarray           # 市場レジーム (0-6)
    volatility_coherence: np.ndarray    # ボラティリティコヒーレンス
    trend_signals: np.ndarray           # トレンド信号
    current_regime: str                 # 現在の市場レジーム
    current_volatility: float           # 現在のボラティリティ値


@jit(nopython=True, cache=True)
def calculate_directional_entropy(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    方向性エントロピーを計算（価格変動の方向性の不確実性を測定）
    """
    length = len(prices)
    entropy_values = np.full(length, np.nan)
    
    for i in range(window, length):
        price_window = prices[i-window+1:i+1]
        
        # 価格変動の方向性を計算
        directions = np.zeros(window-1)
        for j in range(1, window):
            if price_window[j] > price_window[j-1]:
                directions[j-1] = 1  # 上昇
            elif price_window[j] < price_window[j-1]:
                directions[j-1] = -1  # 下落
            else:
                directions[j-1] = 0  # 横這い
        
        # 方向性の分布を計算
        up_count = np.sum(directions == 1)
        down_count = np.sum(directions == -1)
        flat_count = np.sum(directions == 0)
        
        total = up_count + down_count + flat_count
        if total > 0:
            p_up = up_count / total
            p_down = down_count / total
            p_flat = flat_count / total
            
            # シャノンエントロピーを計算
            entropy_val = 0.0
            if p_up > 0:
                entropy_val -= p_up * np.log2(p_up)
            if p_down > 0:
                entropy_val -= p_down * np.log2(p_down)
            if p_flat > 0:
                entropy_val -= p_flat * np.log2(p_flat)
            
            entropy_values[i] = entropy_val
    
    return entropy_values


@jit(nopython=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    ハースト指数に基づくフラクタル次元を計算
    """
    length = len(prices)
    fractal_dims = np.full(length, np.nan)
    
    for i in range(window, length):
        price_window = prices[i-window+1:i+1]
        
        # ログリターンを計算
        log_returns = np.zeros(window-1)
        for j in range(1, window):
            if price_window[j] > 0 and price_window[j-1] > 0:
                log_returns[j-1] = np.log(price_window[j] / price_window[j-1])
        
        # R/S統計を計算（簡略版）
        mean_return = np.mean(log_returns)
        deviations = log_returns - mean_return
        
        # 累積偏差
        cumulative_deviations = np.zeros(len(deviations))
        for j in range(len(deviations)):
            cumulative_deviations[j] = np.sum(deviations[:j+1])
        
        # レンジを計算
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # 標準偏差を計算
        S = np.std(log_returns)
        
        if S > 1e-10 and R > 1e-10:
            # ハースト指数の近似
            hurst = np.log(R/S) / np.log(window/2.0)
            # フラクタル次元 = 2 - ハースト指数
            fractal_dim = 2.0 - max(0.0, min(1.0, hurst))
            fractal_dims[i] = fractal_dim
    
    return fractal_dims


@jit(nopython=True, cache=True)
def calculate_acceleration_tensor(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    価格変動の加速度テンソルを計算
    """
    length = len(prices)
    acceleration = np.full(length, np.nan)
    
    for i in range(window, length):
        price_window = prices[i-window+1:i+1]
        
        # 一次差分（速度）
        velocity = np.zeros(window-1)
        for j in range(1, window):
            velocity[j-1] = price_window[j] - price_window[j-1]
        
        # 二次差分（加速度）
        if len(velocity) > 1:
            accel_values = np.zeros(len(velocity)-1)
            for j in range(1, len(velocity)):
                accel_values[j-1] = velocity[j] - velocity[j-1]
            
            # 加速度の分散（加速度テンソルの代理指標）
            if len(accel_values) > 0:
                acceleration[i] = np.var(accel_values)
    
    return acceleration


@jit(nopython=True, cache=True)
def calculate_uncertainty_principle(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    ハイゼンベルクの不確定性原理を価格分析に適用
    ΔP × Δt ≥ ℏ/2 の概念を価格変動に応用
    """
    length = len(prices)
    uncertainty = np.full(length, np.nan)
    
    for i in range(window, length):
        price_window = prices[i-window+1:i+1]
        
        # 価格の不確定性（標準偏差）
        price_uncertainty = np.std(price_window)
        
        # 時間の不確定性（変動の時間的一貫性の逆数）
        time_variations = np.zeros(window-1)
        for j in range(1, window):
            time_variations[j-1] = abs(price_window[j] - price_window[j-1])
        
        # 時間的一貫性の計算
        if len(time_variations) > 1:
            time_consistency = 1.0 / (np.std(time_variations) + 1e-10)
            
            # 不確定性原理値
            uncertainty[i] = price_uncertainty * time_consistency
    
    return uncertainty


def calculate_wavelet_energy(prices: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
    """
    ウェーブレット変換による多重解像度エネルギー解析
    """
    try:
        # ドビーシーズウェーブレットの近似実装
        # 実際のウェーブレット変換の代わりに、多重スケール分析を使用
        length = len(prices)
        energy_values = np.full(length, np.nan)
        
        # 複数のスケールでエネルギーを計算
        scales = [4, 8, 16, 32]
        
        for i in range(32, length):
            total_energy = 0.0
            
            for scale in scales:
                if i >= scale:
                    window = prices[i-scale+1:i+1]
                    
                    # 各スケールでのエネルギー（分散）を計算
                    if len(window) > 1:
                        energy = np.var(window)
                        total_energy += energy / scale  # スケール正規化
            
            energy_values[i] = total_energy
        
        return energy_values
    
    except Exception:
        # エラーの場合のフォールバック
        return calculate_acceleration_tensor(prices, 21)


@jit(nopython=True, cache=True)
def calculate_volatility_coherence(
    entropy: np.ndarray,
    fractal: np.ndarray,
    acceleration: np.ndarray,
    uncertainty: np.ndarray
) -> np.ndarray:
    """
    複数のボラティリティ指標の一貫性（コヒーレンス）を計算
    0-1の範囲で正規化されたコヒーレンススコア
    """
    length = len(entropy)
    coherence = np.full(length, np.nan)
    
    for i in range(21, length):
        # 各指標を0-1の範囲に正規化
        ent = entropy[i] if not np.isnan(entropy[i]) else 1.0
        frac = fractal[i] if not np.isnan(fractal[i]) else 1.5
        acc = acceleration[i] if not np.isnan(acceleration[i]) else 0.0
        unc = uncertainty[i] if not np.isnan(uncertainty[i]) else 0.0
        
        # 対数正規化（大きな値を抑制）
        acc_norm = min(np.log10(acc + 1.0) / 10.0, 1.0) if acc > 0 else 0.0
        unc_norm = min(unc / 10.0, 1.0)  # 不確定性は通常0-10程度
        ent_norm = min(ent / 2.0, 1.0)   # エントロピーは最大1.585（log2(3)）
        frac_norm = max(0.0, min((frac - 1.0) / 1.0, 1.0))  # フラクタル次元は1-2の範囲
        
        # 正規化された指標配列
        indicators = np.array([ent_norm, frac_norm, acc_norm, unc_norm])
        
        # コヒーレンス計算：指標の一致度を測定
        mean_indicator = np.mean(indicators)
        variance = 0.0
        for j in range(4):
            variance += (indicators[j] - mean_indicator) ** 2
        variance /= 4.0
        
        # コヒーレンス = 1 - 正規化された分散
        coherence[i] = max(0.0, min(1.0, 1.0 - variance))
    
    return coherence


@jit(nopython=True, cache=True)
def calculate_market_regime(
    entropy: np.ndarray,
    fractal: np.ndarray,
    coherence: np.ndarray
) -> np.ndarray:
    """
    市場レジームを7段階で分類
    0: Ultra Low Volatility, 1: Low Volatility, 2: Normal, 3: Elevated, 
    4: High Volatility, 5: Extreme Volatility, 6: Crisis Mode
    """
    length = len(entropy)
    regime = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(21, length):
        ent = entropy[i] if not np.isnan(entropy[i]) else 1.0
        frac = fractal[i] if not np.isnan(fractal[i]) else 1.5
        coh = coherence[i] if not np.isnan(coherence[i]) else 0.5
        
        # 正規化された成分
        ent_norm = min(ent / 2.0, 1.0)  # 0-1
        frac_norm = max(0.0, min((2.0 - frac), 1.0))  # 0-1（逆転）
        coh_weight = 0.5 + coh * 0.5  # 0.5-1.0
        
        # 複合ボラティリティスコア（0-1の範囲）
        volatility_score = (ent_norm * 0.6 + frac_norm * 0.4) * coh_weight
        
        # レジーム分類（より適切な閾値）
        if volatility_score < 0.15:
            regime[i] = 0  # Ultra Low
        elif volatility_score < 0.30:
            regime[i] = 1  # Low
        elif volatility_score < 0.50:
            regime[i] = 2  # Normal
        elif volatility_score < 0.65:
            regime[i] = 3  # Elevated
        elif volatility_score < 0.80:
            regime[i] = 4  # High
        elif volatility_score < 0.90:
            regime[i] = 5  # Extreme
        else:
            regime[i] = 6  # Crisis
    
    return regime


@jit(nopython=True, cache=True)
def calculate_quantum_volatility_core(
    entropy: np.ndarray,
    fractal: np.ndarray,
    wavelet: np.ndarray,
    acceleration: np.ndarray,
    uncertainty: np.ndarray,
    coherence: np.ndarray
) -> np.ndarray:
    """
    量子ボラティリティの統合計算
    0-10の範囲で正規化された統合ボラティリティスコア
    """
    length = len(entropy)
    quantum_vol = np.full(length, np.nan)
    
    for i in range(21, length):
        # 各成分を取得（NaN処理）
        ent = entropy[i] if not np.isnan(entropy[i]) else 1.0
        frac = fractal[i] if not np.isnan(fractal[i]) else 1.5
        wav = wavelet[i] if not np.isnan(wavelet[i]) else 0.0
        acc = acceleration[i] if not np.isnan(acceleration[i]) else 0.0
        unc = uncertainty[i] if not np.isnan(uncertainty[i]) else 0.0
        coh = coherence[i] if not np.isnan(coherence[i]) else 0.5
        
        # 各成分を0-1の範囲に正規化
        ent_norm = min(ent / 2.0, 1.0)  # エントロピー最大値は約1.585
        frac_norm = max(0.0, min((2.0 - frac), 1.0))  # フラクタル逆転（高い値=低ボラ）
        wav_norm = min(np.log10(wav + 1.0) / 10.0, 1.0) if wav > 0 else 0.0
        acc_norm = min(np.log10(acc + 1.0) / 10.0, 1.0) if acc > 0 else 0.0
        unc_norm = min(unc / 10.0, 1.0)  # 不確定性の正規化
        
        # 基本ボラティリティスコア（0-1）
        base_volatility = (
            ent_norm * 0.30 +     # エントロピー 30%
            frac_norm * 0.25 +    # フラクタル次元 25%
            wav_norm * 0.20 +     # ウェーブレット 20%
            acc_norm * 0.15 +     # 加速度 15%
            unc_norm * 0.10       # 不確定性 10%
        )
        
        # コヒーレンス調整（信頼性による重み付け）
        # コヒーレンスが高い場合、基本スコアを強調
        # コヒーレンスが低い場合、ノイズとして減衰
        confidence_factor = 0.5 + coh * 0.5  # 0.5-1.0の範囲
        
        # 最終的な量子ボラティリティ（0-10スケール）
        quantum_vol[i] = base_volatility * confidence_factor * 10.0
    
    return quantum_vol


class QuantumVolatilityTensor(Indicator):
    """
    量子ボラティリティテンソル（QVT）インジケーター
    
    革新的な多次元ボラティリティ分析システム：
    
    1. 方向性エントロピー - 価格変動方向の不確実性
    2. フラクタル次元 - 市場の自己相似性とメモリー効果
    3. ウェーブレット エネルギー - 多重時間スケールの変動解析
    4. 加速度テンソル - 価格変動の加速度（二次微分）
    5. 不確定性原理 - ハイゼンベルク原理の価格分析への応用
    6. ボラティリティコヒーレンス - 指標間の一貫性
    7. 市場レジーム分類 - 7段階のボラティリティ状態
    
    ATRの問題を解決：
    - 方向性を考慮した変動測定
    - 多次元的なボラティリティ評価
    - 適応的な市場状態認識
    """
    
    def __init__(self,
                 window: int = 21,
                 fractal_window: int = 21,
                 entropy_window: int = 21,
                 uncertainty_window: int = 21,
                 wavelet_type: str = 'db4',
                 regime_sensitivity: float = 1.0):
        """
        Args:
            window: 基本計算ウィンドウ
            fractal_window: フラクタル次元計算ウィンドウ
            entropy_window: エントロピー計算ウィンドウ
            uncertainty_window: 不確定性計算ウィンドウ
            wavelet_type: ウェーブレット種類
            regime_sensitivity: レジーム分類の感度
        """
        super().__init__(f"QVT(w={window},frac={fractal_window},ent={entropy_window},unc={uncertainty_window})")
        
        self.window = window
        self.fractal_window = fractal_window
        self.entropy_window = entropy_window
        self.uncertainty_window = uncertainty_window
        self.wavelet_type = wavelet_type
        self.regime_sensitivity = regime_sensitivity
        
        # レジーム名の定義
        self.regime_names = [
            'Ultra Low Volatility',
            'Low Volatility', 
            'Normal Volatility',
            'Elevated Volatility',
            'High Volatility',
            'Extreme Volatility',
            'Crisis Mode'
        ]
        
        self._cache = {}
        self._result: Optional[QuantumVolatilityResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(data, pd.DataFrame):
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                data_hash_val = hash(data.tobytes())
            else:
                data_hash_val = hash(str(data))

        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}")
            data_hash_val = hash(str(data))

        param_str = f"w={self.window}_frac={self.fractal_window}_ent={self.entropy_window}_unc={self.uncertainty_window}_wav={self.wavelet_type}"
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QuantumVolatilityResult:
        """
        量子ボラティリティテンソルを計算する
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。")
            return self._empty_result()
            
        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                if len(self._result.quantum_volatility) == current_data_len:
                    return self._copy_result(self._result)
                else:
                    del self._cache[data_hash]
                    self._result = None

            # データの準備
            if isinstance(data, pd.DataFrame):
                prices = data['close'].values.astype(np.float64)
            else:
                prices = data[:, 3].astype(np.float64) if data.ndim > 1 else data.astype(np.float64)

            # 各成分の計算
            self.logger.info("量子ボラティリティテンソルの各成分を計算中...")
            
            # 1. 方向性エントロピー
            directional_entropy = calculate_directional_entropy(prices, self.entropy_window)
            
            # 2. フラクタル次元
            fractal_dimension = calculate_fractal_dimension(prices, self.fractal_window)
            
            # 3. ウェーブレット エネルギー
            wavelet_energy = calculate_wavelet_energy(prices, self.wavelet_type)
            
            # 4. 加速度テンソル
            acceleration_tensor = calculate_acceleration_tensor(prices, self.window)
            
            # 5. 不確定性原理
            uncertainty_principle = calculate_uncertainty_principle(prices, self.uncertainty_window)
            
            # 6. ボラティリティコヒーレンス
            volatility_coherence = calculate_volatility_coherence(
                directional_entropy, fractal_dimension, acceleration_tensor, uncertainty_principle
            )
            
            # 7. 市場レジーム
            market_regime = calculate_market_regime(
                directional_entropy, fractal_dimension, volatility_coherence
            )
            
            # 8. 統合量子ボラティリティ
            quantum_volatility = calculate_quantum_volatility_core(
                directional_entropy, fractal_dimension, wavelet_energy,
                acceleration_tensor, uncertainty_principle, volatility_coherence
            )
            
            # 9. トレンド信号（量子ボラティリティの変化率から）
            trend_signals = self._calculate_trend_signals(quantum_volatility)
            
            # 現在の状態
            current_regime_idx = int(market_regime[-1]) if not np.isnan(market_regime[-1]) else 2
            current_regime_idx = max(0, min(6, current_regime_idx))
            current_regime = self.regime_names[current_regime_idx]
            current_volatility = quantum_volatility[-1] if not np.isnan(quantum_volatility[-1]) else 0.0

            result = QuantumVolatilityResult(
                quantum_volatility=quantum_volatility,
                directional_entropy=directional_entropy,
                fractal_dimension=fractal_dimension,
                wavelet_energy=wavelet_energy,
                acceleration_tensor=acceleration_tensor,
                uncertainty_principle=uncertainty_principle,
                market_regime=market_regime,
                volatility_coherence=volatility_coherence,
                trend_signals=trend_signals,
                current_regime=current_regime,
                current_volatility=current_volatility
            )

            # 結果を保存
            self._result = result
            self._cache[data_hash] = self._result
            return self._copy_result(result)
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"QVT '{self.name}' 計算中にエラー: {error_msg}\n{stack_trace}")
            return self._empty_result(current_data_len)

    def _calculate_trend_signals(self, quantum_volatility: np.ndarray) -> np.ndarray:
        """量子ボラティリティからトレンド信号を計算"""
        length = len(quantum_volatility)
        trend_signals = np.zeros(length, dtype=np.int8)
        
        for i in range(5, length):
            if not np.isnan(quantum_volatility[i]) and not np.isnan(quantum_volatility[i-5]):
                change = quantum_volatility[i] - quantum_volatility[i-5]
                if abs(change) > 0.1:  # 閾値
                    trend_signals[i] = 1 if change > 0 else -1
        
        return trend_signals

    def _empty_result(self, length: int = 0) -> QuantumVolatilityResult:
        """空の結果を返す"""
        return QuantumVolatilityResult(
            quantum_volatility=np.full(length, np.nan),
            directional_entropy=np.full(length, np.nan),
            fractal_dimension=np.full(length, np.nan),
            wavelet_energy=np.full(length, np.nan),
            acceleration_tensor=np.full(length, np.nan),
            uncertainty_principle=np.full(length, np.nan),
            market_regime=np.full(length, np.nan),
            volatility_coherence=np.full(length, np.nan),
            trend_signals=np.zeros(length, dtype=np.int8),
            current_regime='Normal Volatility',
            current_volatility=0.0
        )

    def _copy_result(self, result: QuantumVolatilityResult) -> QuantumVolatilityResult:
        """結果のコピーを作成"""
        return QuantumVolatilityResult(
            quantum_volatility=result.quantum_volatility.copy(),
            directional_entropy=result.directional_entropy.copy(),
            fractal_dimension=result.fractal_dimension.copy(),
            wavelet_energy=result.wavelet_energy.copy(),
            acceleration_tensor=result.acceleration_tensor.copy(),
            uncertainty_principle=result.uncertainty_principle.copy(),
            market_regime=result.market_regime.copy(),
            volatility_coherence=result.volatility_coherence.copy(),
            trend_signals=result.trend_signals.copy(),
            current_regime=result.current_regime,
            current_volatility=result.current_volatility
        )

    # ゲッター メソッド
    def get_quantum_volatility(self) -> Optional[np.ndarray]:
        """統合量子ボラティリティを取得"""
        return self._result.quantum_volatility.copy() if self._result else None

    def get_directional_entropy(self) -> Optional[np.ndarray]:
        """方向性エントロピーを取得"""
        return self._result.directional_entropy.copy() if self._result else None

    def get_fractal_dimension(self) -> Optional[np.ndarray]:
        """フラクタル次元を取得"""
        return self._result.fractal_dimension.copy() if self._result else None

    def get_market_regime(self) -> Optional[np.ndarray]:
        """市場レジームを取得"""
        return self._result.market_regime.copy() if self._result else None

    def get_current_regime(self) -> str:
        """現在の市場レジームを取得"""
        return self._result.current_regime if self._result else 'Normal Volatility'

    def get_volatility_coherence(self) -> Optional[np.ndarray]:
        """ボラティリティコヒーレンスを取得"""
        return self._result.volatility_coherence.copy() if self._result else None

    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        self.logger.debug(f"QVT '{self.name}' がリセットされました。") 