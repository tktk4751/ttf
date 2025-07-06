#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌀 **Quantum Supreme Hilbert Transform V1.0 - 量子Supreme版ヒルベルト変換** 🌀

9点高精度ヒルベルト変換による瞬時振幅・位相・周波数解析
トレンドモードとサイクルモードの自動判別機能付き

🌟 **特徴:**
- 9点高精度ヒルベルト変換アルゴリズム
- 量子コヒーレンス計算による位相安定性測定
- トレンドモード/サイクルモード自動判別
- 瞬時振幅・位相・周波数の同時計算
- Numba最適化による高速処理
- 複数の価格ソース対応

🎯 **用途:**
- 市場のトレンド/レンジ状態判別
- 瞬時振幅による変動性測定
- 位相解析による価格サイクル検出
- 高精度な市場状態分析
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import warnings

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource

warnings.filterwarnings('ignore')


@dataclass
class QuantumSupremeHilbertResult:
    """量子Supreme版ヒルベルト変換の計算結果"""
    amplitude: np.ndarray           # 瞬時振幅
    phase: np.ndarray              # 瞬時位相（-π to π）
    frequency: np.ndarray          # 瞬時周波数（正規化済み）
    quantum_coherence: np.ndarray  # 量子コヒーレンス（位相安定性）
    trend_mode: np.ndarray         # トレンドモード判別（1=トレンド、0=サイクル）
    market_state: np.ndarray       # 市場状態（0=レンジ、1=弱トレンド、2=強トレンド）
    cycle_strength: np.ndarray     # サイクル強度（0-1）
    trend_strength: np.ndarray     # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def quantum_supreme_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 量子Supreme版ヒルベルト変換（9点高精度版）
    
    Args:
        prices: 価格データ配列
    
    Returns:
        Tuple[amplitude, phase, frequency, quantum_coherence]
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    frequency = np.zeros(n)
    quantum_coherence = np.zeros(n)
    
    if n < 16:
        return amplitude, phase, frequency, quantum_coherence
    
    # 改良ヒルベルト変換 - より高精度な計算
    for i in range(8, n-8):
        real_part = prices[i]
        
        # 9点ヒルベルト変換（より高精度）
        imag_part = (
            (prices[i-7] - prices[i+7]) +
            3 * (prices[i-5] - prices[i+5]) +
            5 * (prices[i-3] - prices[i+3]) +
            7 * (prices[i-1] - prices[i+1])
        ) / 32.0
        
        # 瞬時振幅
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        
        # 瞬時周波数（位相の時間微分を改良）
        if i > 8:
            phase_diff = phase[i] - phase[i-1]
            # 位相のラッピング処理
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            
            # 周波数計算を改良（より感度を高める）
            frequency[i] = abs(phase_diff) * 2.0 / (2 * np.pi)  # 感度向上
        
        # 🔬 量子コヒーレンス計算の改良 - より敏感な位相安定性測定
        if i >= 16:
            # 過去14点での位相安定性（ウィンドウサイズ拡大）
            phase_diffs = np.zeros(14)
            for j in range(14):
                if i-j-1 >= 0:
                    phase_diff_j = phase[i-j] - phase[i-j-1]
                    # 位相ラッピング補正
                    if phase_diff_j > np.pi:
                        phase_diff_j -= 2 * np.pi
                    elif phase_diff_j < -np.pi:
                        phase_diff_j += 2 * np.pi
                    phase_diffs[j] = phase_diff_j
            
            # 位相差の統計量を計算
            mean_phase_diff = np.mean(phase_diffs)
            phase_variance = 0.0
            for j in range(14):
                phase_variance += (phase_diffs[j] - mean_phase_diff)**2
            phase_variance /= 14.0
            
            # 位相ジャンプをカウント（不安定性の指標）
            jump_count = 0
            for j in range(14):
                if abs(phase_diffs[j]) > 0.3:  # より敏感な閾値
                    jump_count += 1
            
            jump_penalty = jump_count / 14.0
            
            # コヒーレンス計算を改良（より敏感に）
            base_coherence = 1.0 / (1.0 + phase_variance * 20.0)  # 感度向上
            quantum_coherence[i] = base_coherence * (1.0 - jump_penalty * 0.5)
            quantum_coherence[i] = max(min(quantum_coherence[i], 1.0), 0.0)
    
    # 境界値処理
    for i in range(8):
        amplitude[i] = amplitude[8] if n > 8 else 0.0
        phase[i] = phase[8] if n > 8 else 0.0
        frequency[i] = frequency[8] if n > 8 else 0.0
        quantum_coherence[i] = quantum_coherence[8] if n > 8 else 0.0
    for i in range(n-8, n):
        amplitude[i] = amplitude[n-9] if n > 8 else 0.0
        phase[i] = phase[n-9] if n > 8 else 0.0
        frequency[i] = frequency[n-9] if n > 8 else 0.0
        quantum_coherence[i] = quantum_coherence[n-9] if n > 8 else 0.0
    
    return amplitude, phase, frequency, quantum_coherence


@njit(fastmath=True, cache=True)
def analyze_market_state_numba(
    amplitude: np.ndarray, 
    phase: np.ndarray, 
    frequency: np.ndarray, 
    quantum_coherence: np.ndarray,
    coherence_threshold: float = 0.6,        # より現実的な閾値に調整
    frequency_threshold: float = 0.05,       # 感度向上に合わせて調整  
    amplitude_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    市場状態分析（トレンドモード vs サイクルモード判別）
    
    エラーズ理論に基づく実用的判別ロジック:
    - DC成分優位（一方向性） = トレンドモード  
    - AC成分優位（振動性） = サイクルモード
    
    Args:
        amplitude: 瞬時振幅
        phase: 瞬時位相  
        frequency: 瞬時周波数
        quantum_coherence: 量子コヒーレンス
        coherence_threshold: コヒーレンス閾値（サイクル判定用）
        frequency_threshold: 周波数閾値（サイクル判定用）
        amplitude_threshold: 振幅閾値（変動性判定用）
    
    Returns:
        Tuple[trend_mode, market_state, cycle_strength, trend_strength]
    """
    n = len(amplitude)
    trend_mode = np.zeros(n, dtype=np.int8)  # 1=トレンド、0=サイクル
    market_state = np.zeros(n, dtype=np.int8)  # 0=レンジ、1=弱トレンド、2=強トレンド
    cycle_strength = np.zeros(n)
    trend_strength = np.zeros(n)
    
    # 分析ウィンドウサイズ
    window = 14
    
    for i in range(window, n):
        # 1. DC成分分析（エラーズ理論の核心）
        # 振幅データから一方向性を測定
        amplitude_values = amplitude[i-window+1:i+1]
        amplitude_mean = np.mean(amplitude_values)
        
        # 振幅の線形回帰で傾向を測定
        amplitude_trend_slope = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        
        for j in range(window):
            x = float(j)
            y = amplitude_values[j]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
        
        if sum_x2 * window - sum_x * sum_x != 0:
            amplitude_trend_slope = (sum_xy * window - sum_x * sum_y) / (sum_x2 * window - sum_x * sum_x)
        
        # DC成分の強さ（一方向性の程度）
        dc_strength = abs(amplitude_trend_slope) / max(amplitude_mean, 0.001)
        
        # 2. AC成分分析（振動成分）
        # 振幅の変動係数
        amplitude_variance = 0.0
        for j in range(window):
            amplitude_variance += (amplitude_values[j] - amplitude_mean)**2
        amplitude_variance /= window
        amplitude_std = np.sqrt(amplitude_variance)
        
        # 振動の規則性を測定（AC成分の特徴）
        oscillation_regularity = 0.0
        if amplitude_mean > 0:
            # 平均からの偏差の周期性を測定
            deviations = amplitude_values - amplitude_mean
            # 連続する偏差の符号変化をカウント
            sign_changes = 0
            for j in range(1, window):
                if deviations[j] * deviations[j-1] < 0:  # 符号が変わった
                    sign_changes += 1
            
            # 規則的な振動ほど符号変化が多い
            oscillation_regularity = sign_changes / (window - 1)
        
        # AC成分の強さ
        ac_strength = oscillation_regularity * (amplitude_std / max(amplitude_mean, 0.001))
        
        # 3. 価格変動の直接分析
        # 価格レンジの分析（エラーズ理論）
        price_range = np.max(amplitude_values) - np.min(amplitude_values)
        price_midpoint = (np.max(amplitude_values) + np.min(amplitude_values)) / 2
        
        # 価格が一方向に偏っているかを測定
        price_bias = 0.0
        if price_range > 0:
            current_position = (amplitude[i] - np.min(amplitude_values)) / price_range
            price_bias = abs(current_position - 0.5) * 2  # 0-1の範囲
        
        # 4. 周波数とコヒーレンスの補正分析
        avg_coherence = np.mean(quantum_coherence[i-window+1:i+1])
        avg_frequency = np.mean(frequency[i-window+1:i+1])
        
        # 周波数の安定性
        freq_variance = 0.0
        for j in range(window):
            freq_variance += (frequency[i-window+1+j] - avg_frequency)**2
        freq_variance /= window
        freq_stability = 1.0 / (1.0 + freq_variance * 100.0)
        
        # 5. エラーズ式の実用的実装
        # DC/AC ratio の計算
        dc_ac_ratio = 0.0
        if ac_strength > 0.001:
            dc_ac_ratio = dc_strength / (dc_strength + ac_strength)
        else:
            dc_ac_ratio = 1.0  # AC成分がない場合はDC優位
        
        # トレンド強度の計算（エラーズ理論ベース）
        trend_strength[i] = (
            dc_ac_ratio * 0.4 +                      # DC成分優位（最重要）
            price_bias * 0.25 +                      # 価格の一方向偏り
            (1.0 - freq_stability) * 0.2 +           # 周波数不安定（トレンド的）
            (1.0 - avg_coherence) * 0.1 +            # 位相不安定（トレンド的）
            min(avg_frequency * 20.0, 1.0) * 0.05    # 高周波数（トレンド的）
        )
        
        # サイクル強度の計算
        cycle_strength[i] = (
            (1.0 - dc_ac_ratio) * 0.4 +              # AC成分優位（最重要）
            oscillation_regularity * 0.25 +          # 規則的振動
            freq_stability * 0.2 +                   # 周波数安定（サイクル的）
            avg_coherence * 0.1 +                    # 位相安定（サイクル的）
            (1.0 - min(avg_frequency * 20.0, 1.0)) * 0.05  # 低周波数（サイクル的）
        )
        
        # 正規化
        total_strength = trend_strength[i] + cycle_strength[i]
        if total_strength > 0:
            trend_strength[i] /= total_strength
            cycle_strength[i] /= total_strength
        else:
            trend_strength[i] = 0.5
            cycle_strength[i] = 0.5
        
        # トレンドモード判定（エラーズ理論の実用的適用）
        # バランス調整版の判定ロジック
        
        # 主要指標のスコア計算
        trend_score = 0.0
        cycle_score = 0.0
        
        # 1. DC/AC比率による判定（最重要）
        if dc_ac_ratio > 0.65:
            trend_score += 2.0  # 強いトレンド指標
        elif dc_ac_ratio > 0.45:
            trend_score += 1.0  # 中程度のトレンド指標
        else:
            cycle_score += 1.5  # サイクル指標
        
        # 2. 振動規則性による判定
        if oscillation_regularity > 0.6:
            cycle_score += 1.5  # 規則的振動はサイクル的
        elif oscillation_regularity > 0.3:
            cycle_score += 0.5  # 中程度の振動
        else:
            trend_score += 1.0  # 非規則的はトレンド的
        
        # 3. 価格偏りによる判定
        if price_bias > 0.4:
            trend_score += 1.0  # 明確な偏り
        elif price_bias < 0.2:
            cycle_score += 0.5  # 中央付近はサイクル的
        
        # 4. 周波数とコヒーレンスによる補正
        if avg_coherence > 0.8 and freq_stability > 0.8:
            cycle_score += 0.5  # 高い安定性はサイクル的
        elif freq_stability < 0.6:
            trend_score += 0.5  # 不安定性はトレンド的
        
        # 最終判定
        if trend_score > cycle_score:
            trend_mode[i] = 1  # トレンドモード
        else:
            trend_mode[i] = 0  # サイクルモード
        
        # 市場状態判定（トレンド強度ベース）
        if trend_strength[i] > 0.7:
            market_state[i] = 2  # 強トレンド
        elif trend_strength[i] > 0.4:
            market_state[i] = 1  # 弱トレンド
        else:
            market_state[i] = 0  # レンジ（サイクル優位）
    
    # 境界値処理
    for i in range(window):
        trend_mode[i] = trend_mode[window] if n > window else 0
        market_state[i] = market_state[window] if n > window else 0
        cycle_strength[i] = cycle_strength[window] if n > window else 0.5
        trend_strength[i] = trend_strength[window] if n > window else 0.5
    
    return trend_mode, market_state, cycle_strength, trend_strength


class QuantumSupremeHilbert(Indicator):
    """
    量子Supreme版ヒルベルト変換インディケーター
    
    9点高精度ヒルベルト変換による瞬時振幅・位相・周波数解析と
    トレンドモード/サイクルモードの自動判別機能を提供
    
    特徴:
    - 高精度な9点ヒルベルト変換アルゴリズム
    - 量子コヒーレンスによる位相安定性測定
    - トレンド/サイクル状態の自動判別
    - 複数の価格ソース対応
    - キャッシュ機能による高速化
    """
    
    def __init__(
        self,
        src_type: str = 'close',                    # 価格ソース
        coherence_threshold: float = 0.6,           # コヒーレンス閾値（サイクル判定用）
        frequency_threshold: float = 0.05,          # 周波数閾値（サイクル判定用）
        amplitude_threshold: float = 0.5,           # 振幅閾値
        analysis_window: int = 14,                  # 分析ウィンドウサイズ
        min_periods: int = 32                       # 最小計算期間
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            coherence_threshold: コヒーレンス閾値（トレンド判定用）
            frequency_threshold: 周波数閾値（トレンド判定用）
            amplitude_threshold: 振幅閾値（変動性判定用）
            analysis_window: 市場状態分析ウィンドウサイズ
            min_periods: 最小計算期間
        """
        # 指標名の作成
        indicator_name = f"QuantumSupremeHilbert(src={src_type}, coh={coherence_threshold:.2f}, freq={frequency_threshold:.2f})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.src_type = src_type.lower()
        self.coherence_threshold = coherence_threshold
        self.frequency_threshold = frequency_threshold
        self.amplitude_threshold = amplitude_threshold
        self.analysis_window = analysis_window
        self.min_periods = min_periods
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    src_prices = PriceSource.calculate_source(data, self.src_type)
                    first_val = float(src_prices[0]) if len(src_prices) > 0 else 0.0
                    last_val = float(src_prices[-1]) if len(src_prices) > 0 else 0.0
                else:
                    first_val = last_val = 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータ情報
            params_sig = f"{self.src_type}_{self.coherence_threshold}_{self.frequency_threshold}_{self.analysis_window}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.src_type}_{self.coherence_threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QuantumSupremeHilbertResult:
        """
        量子Supreme版ヒルベルト変換を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            QuantumSupremeHilbertResult: ヒルベルト変換結果と市場状態分析
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return QuantumSupremeHilbertResult(
                    amplitude=cached_result.amplitude.copy(),
                    phase=cached_result.phase.copy(),
                    frequency=cached_result.frequency.copy(),
                    quantum_coherence=cached_result.quantum_coherence.copy(),
                    trend_mode=cached_result.trend_mode.copy(),
                    market_state=cached_result.market_state.copy(),
                    cycle_strength=cached_result.cycle_strength.copy(),
                    trend_strength=cached_result.trend_strength.copy()
                )
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            # データ長の検証
            data_length = len(src_prices)
            if data_length < self.min_periods:
                # データが不足している場合は空の結果を返す
                return self._create_empty_result(data_length)
            
            # NumPy配列に変換（float64型で統一）
            src_prices = np.asarray(src_prices, dtype=np.float64)
            
            # ヒルベルト変換の計算
            amplitude, phase, frequency, quantum_coherence = quantum_supreme_hilbert_transform_numba(
                src_prices
            )
            
            # 市場状態分析
            trend_mode, market_state, cycle_strength, trend_strength = analyze_market_state_numba(
                amplitude=amplitude,
                phase=phase,
                frequency=frequency,
                quantum_coherence=quantum_coherence,
                coherence_threshold=self.coherence_threshold,
                frequency_threshold=self.frequency_threshold,
                amplitude_threshold=self.amplitude_threshold
            )
            
            # 結果の作成
            result = QuantumSupremeHilbertResult(
                amplitude=amplitude.copy(),
                phase=phase.copy(),
                frequency=frequency.copy(),
                quantum_coherence=quantum_coherence.copy(),
                trend_mode=trend_mode.copy(),
                market_state=market_state.copy(),
                cycle_strength=cycle_strength.copy(),
                trend_strength=trend_strength.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = amplitude  # 基底クラスの要件を満たすため（振幅を主要な値として使用）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"QuantumSupremeHilbert計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            data_len = len(data) if hasattr(data, '__len__') else 0
            return self._create_empty_result(data_len)
    
    def _create_empty_result(self, length: int) -> QuantumSupremeHilbertResult:
        """空の結果を作成"""
        return QuantumSupremeHilbertResult(
            amplitude=np.full(length, np.nan),
            phase=np.full(length, np.nan),
            frequency=np.full(length, np.nan),
            quantum_coherence=np.full(length, np.nan),
            trend_mode=np.zeros(length, dtype=np.int8),
            market_state=np.zeros(length, dtype=np.int8),
            cycle_strength=np.full(length, 0.5),
            trend_strength=np.full(length, 0.5)
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """瞬時振幅値を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.amplitude.copy()
    
    def get_amplitude(self) -> Optional[np.ndarray]:
        """瞬時振幅を取得"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.amplitude.copy()
    
    def get_phase(self) -> Optional[np.ndarray]:
        """瞬時位相を取得"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.phase.copy()
    
    def get_frequency(self) -> Optional[np.ndarray]:
        """瞬時周波数を取得"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.frequency.copy()
    
    def get_quantum_coherence(self) -> Optional[np.ndarray]:
        """量子コヒーレンスを取得"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.quantum_coherence.copy()
    
    def get_trend_mode(self) -> Optional[np.ndarray]:
        """トレンドモード判別を取得（1=トレンド、0=サイクル）"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.trend_mode.copy()
    
    def get_market_state(self) -> Optional[np.ndarray]:
        """市場状態を取得（0=レンジ、1=弱トレンド、2=強トレンド）"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.market_state.copy()
    
    def get_cycle_strength(self) -> Optional[np.ndarray]:
        """サイクル強度を取得（0-1）"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.cycle_strength.copy()
    
    def get_trend_strength(self) -> Optional[np.ndarray]:
        """トレンド強度を取得（0-1）"""
        if not self._result_cache:
            return None
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        return result.trend_strength.copy()
    
    def is_trend_mode(self, index: int = -1) -> bool:
        """指定したインデックスでトレンドモードかどうかを判定"""
        trend_mode = self.get_trend_mode()
        if trend_mode is None or len(trend_mode) == 0:
            return False
        if index == -1:
            index = len(trend_mode) - 1
        return bool(trend_mode[index] == 1)
    
    def is_cycle_mode(self, index: int = -1) -> bool:
        """指定したインデックスでサイクルモードかどうかを判定"""
        return not self.is_trend_mode(index)
    
    def get_current_state(self) -> Dict[str, Union[float, int, str]]:
        """現在の市場状態情報を取得"""
        if not self._result_cache or not self._cache_keys:
            return {}
        
        result = self._result_cache[self._cache_keys[-1]]
        if len(result.amplitude) == 0:
            return {}
        
        last_idx = len(result.amplitude) - 1
        
        # 市場状態の文字列化
        market_state_str = {0: "レンジ", 1: "弱トレンド", 2: "強トレンド"}.get(
            int(result.market_state[last_idx]), "不明"
        )
        
        return {
            "amplitude": float(result.amplitude[last_idx]),
            "phase": float(result.phase[last_idx]),
            "frequency": float(result.frequency[last_idx]),
            "quantum_coherence": float(result.quantum_coherence[last_idx]),
            "trend_mode": bool(result.trend_mode[last_idx] == 1),
            "market_state": int(result.market_state[last_idx]),
            "market_state_str": market_state_str,
            "cycle_strength": float(result.cycle_strength[last_idx]),
            "trend_strength": float(result.trend_strength[last_idx])
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = [] 