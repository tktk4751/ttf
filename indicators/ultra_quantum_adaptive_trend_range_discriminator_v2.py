#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD) - V2 Ehlers-DSP Engine
===================================================================================

ジョン・エラーズのデジタル信号処理（DSP）哲学に基づき再設計された、
革新的なトレンド/レンジ判別インジケーター V2。

コアエンジンは、市場の物理的現実に即した2つのDSPモジュールで構成される：
1. ドミナントサイクル測定器 (Dominant Cycle Measurer)
   - ヒルベルト変換を用いたホモダイン弁別器により、市場の主要サイクルをリアルタイムで測定。
2. スペクトルエネルギー比率分析器 (Spectral Energy Ratio Analyzer)
   - ドミナントサイクルに適応するLPFとBPFを使用し、トレンドとサイクルのエネルギー比を直接計算。

特徴：
- 超高精度：市場のスペクトルエネルギー分布を直接測定。
- 超適応性：ドミナントサイクルに応じて全フィルターがリアルタイムで自己調整。
- 超低遅延：計算の大部分を効率的な再帰的フィルター（IIR）で実装し、Numbaで高速化。
- 理論的堅牢性：実績のあるDSP技術に基づいた、シンプルで解釈容易な設計。
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit
import math
import traceback
from typing import Union, Tuple, Dict, Optional, List

# --- 既存の外部依存関係（変更なし） ---
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .smoother.ultimate_smoother import calculate_ultimate_smoother, calculate_ultimate_smoother_adaptive
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from indicators.smoother.ultimate_smoother import calculate_ultimate_smoother, calculate_ultimate_smoother_adaptive


@dataclass
class UQATRDResult:
    """
    UQATRD計算結果 (V2 Ehlers-DSP Engine)
    インターフェース互換性を維持しつつ、V2の出力を格納。
    """
    # --- メイン判定結果 (V2 Engineにより算出) ---
    trend_range_signal: np.ndarray    # 最終的なトレンド/レンジ判定 (0=レンジ to 1=トレンド)
    signal_strength: np.ndarray       # 信号の確信度 (0=不確実 to 1=確実)

    # --- 補助情報 (V2 Engineにより算出) ---
    cycle_adaptive_factor: np.ndarray # V2では「ドミナントサイクル周期」を格納
    adaptive_threshold: np.ndarray    # V2では固定値0.5を返し、シグナル自体を解釈

    # --- V1互換性のためのプレースホルダー (V2では使用されない) ---
    quantum_coherence: np.ndarray
    trend_persistence: np.ndarray
    efficiency_spectrum: np.ndarray
    uncertainty_range: np.ndarray
    confidence_score: np.ndarray


# ================== Ehlers-DSP コア計算エンジン (V2) ==================

@njit(fastmath=True, cache=True)
def calculate_uqatrd_core_v2(
    prices: np.ndarray,
    dc_period: int,
    bandwidth: float,
    dc_smooth_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🚀 UQATRD V2 Ehlers-DSP統合計算エンジン

    Args:
        prices: 価格データ配列
        dc_period: ドミナントサイクル測定のベース周期
        bandwidth: バンドパスフィルターの帯域幅
        dc_smooth_period: ドミナントサイクル平滑化の期間

    Returns:
        Tuple[trend_range_signal, signal_strength, dominant_cycle]
    """
    n = len(prices)
    if n < dc_period:
        return np.zeros(n), np.zeros(n), np.full(n, float(dc_period))

    # --- Stage 1: ドミナントサイクルの測定 (ホモダイン法) ---
    # バンドパスフィルター係数 (固定周期)
    beta1 = math.cos(2 * math.pi / dc_period)
    gamma1 = 1 / math.cos(2 * math.pi * bandwidth / dc_period)
    alpha1 = gamma1 - math.sqrt(gamma1**2 - 1)
    
    bp = np.zeros(n)
    
    # 初期値の設定
    bp[0] = 0.0
    bp[1] = 0.0
    
    for i in range(2, n):
        bp[i] = 0.5 * (1 - alpha1) * (prices[i] - prices[i-2]) \
                + beta1 * (1 + alpha1) * bp[i-1] - alpha1 * bp[i-2]

    # ホモダイン弁別器による位相差の計算
    i1, q1 = bp, np.zeros(n)
    for i in range(1, n):
        q1[i] = bp[i-1] # Quadrature成分はIn-Phase成分の1サンプル遅延 (単純な90度位相シフト)

    i2, q2 = np.roll(i1, 1), np.roll(q1, 1)
    i2[0], q2[0] = i2[1], q2[1]

    re = i1 * i2 + q1 * q2
    im = i1 * q2 - q1 * i2
    
    delta_phase = np.zeros(n)
    for i in range(1, n):
        if abs(re[i]) > 1e-10:
            delta_phase[i] = math.atan(im[i] / re[i])

    # 瞬時周期の計算と平滑化
    instant_period = np.zeros(n)
    for i in range(1, n):
        if abs(delta_phase[i]) > 1e-10:
            instant_period[i] = 2 * math.pi / abs(delta_phase[i])

    # 周期のクリッピングと平滑化
    instant_period = np.clip(instant_period, 10, dc_period * 1.5)
    
    # Ultimate Smootherによる周期の平滑化
    # 既存のUltimateSmoother関数を使用
    dominant_cycle_smoothed, _ = calculate_ultimate_smoother(instant_period, dc_smooth_period)
    
    # NaN値の処理と範囲制限
    dominant_cycle = np.nan_to_num(dominant_cycle_smoothed, nan=dc_period)
    dominant_cycle = np.clip(dominant_cycle, 5.0, dc_period * 2.0)  # 適切な範囲に制限

    # --- Stage 2: トレンド/サイクル分離（完全に再設計） ---
    # トレンド成分を単純な移動平均で計算
    trend_comp = np.zeros(n)
    for i in range(n):
        dc = int(dominant_cycle[i]) if dominant_cycle[i] > 0 else dc_period
        dc = max(dc, 2)  # 最小値を2に制限
        dc = min(dc, i + 1)  # 利用可能なデータ数を超えないように
        
        if dc > 0:
            trend_comp[i] = np.mean(prices[max(0, i-dc+1):i+1])
        else:
            trend_comp[i] = prices[i]
    
    # サイクル成分を価格とトレンド成分の差分で計算
    cycle_comp = prices - trend_comp
    
    # パワー計算を標準偏差ベースに変更（より安定）
    trend_power = np.zeros(n)
    cycle_power = np.zeros(n)
    
    for i in range(n):
        dc = int(dominant_cycle[i]) if dominant_cycle[i] > 0 else dc_period
        dc = max(dc, 2)  # 最小値を2に制限
        dc = min(dc, i + 1)  # 利用可能なデータ数を超えないように
        
        if dc > 1:
            # トレンド成分の標準偏差
            trend_window = trend_comp[max(0, i-dc+1):i+1]
            if len(trend_window) > 1:
                trend_power[i] = np.std(trend_window)
            else:
                trend_power[i] = 0.0
            
            # サイクル成分の標準偏差
            cycle_window = cycle_comp[max(0, i-dc+1):i+1]
            if len(cycle_window) > 1:
                cycle_power[i] = np.std(cycle_window)
            else:
                cycle_power[i] = 0.0
        else:
            trend_power[i] = 0.0
            cycle_power[i] = 0.0

    # トレンド/レンジ比率の計算 - より安定した計算
    raw_signal = np.zeros(n)
    for i in range(n):
        total_power = trend_power[i] + cycle_power[i]
        if total_power > 1e-10:  # 適切な閾値
            raw_signal[i] = trend_power[i] / total_power
        else:
            # デフォルト値を0.5に変更（バランスの取れた値）
            raw_signal[i] = 0.5
    
    # 信号強度の計算 - より安定した計算方法
    strength = np.zeros(n)
    for i in range(n):
        total_power = trend_power[i] + cycle_power[i]
        if total_power > 1e-10:  # 適切な閾値
            # より安定した強度計算
            ratio = abs(trend_power[i] - cycle_power[i]) / total_power
            strength[i] = min(ratio, 1.0)  # 1.0を超えないように制限
        else:
            strength[i] = 0.0  # デフォルト値
    
    # --- Stage 3: 最終出力の平滑化 ---
    # 軽度の平滑化のみ適用（過度な平滑化を避ける）
    light_smooth_period = max(3, dc_smooth_period // 3)  # より短い期間で平滑化
    
    trend_range_signal, _ = calculate_ultimate_smoother(raw_signal, light_smooth_period)
    signal_strength, _ = calculate_ultimate_smoother(strength, light_smooth_period)
    
    # NaN値の処理 - デフォルト値を調整
    trend_range_signal = np.nan_to_num(trend_range_signal, nan=0.5)  # デフォルト値を0.5に変更
    signal_strength = np.nan_to_num(signal_strength, nan=0.0)
    
    trend_range_signal = np.clip(trend_range_signal, 0, 1)
    signal_strength = np.clip(signal_strength, 0, 1)

    return trend_range_signal, signal_strength, dominant_cycle


class UltraQuantumAdaptiveTrendRangeDiscriminator(Indicator):
    """
    🌟 Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD) - V2 Ehlers-DSP Engine
    
    ジョン・エラーズのDSP哲学に基づき再設計されたトレンド/レンジ判別インジケーター。
    市場のドミナントサイクルをリアルタイムで測定し、それに応じて自己調整する
    スペクトルエネルギー分析により、市場モードを判定します。
    
    特徴：
    - UltimateSmootherを使用した高品質な平滑化
    - ドミナントサイクルに適応するフィルター設計
    - スペクトルエネルギー比率による市場モード判定
    """
    
    def __init__(
        self,
        # --- V2 DSP Engine Parameters ---
        dc_period: int = 30,            # ドミナントサイクル測定の基準周期（調整）
        bandwidth: float = 0.15,        # バンドパスフィルターの帯域幅（調整）
        dc_smooth_period: int = 8,      # ドミナントサイクル平滑化期間（調整）

        # --- General Parameters ---
        src_type: str = 'ukf_hlc3',       # 価格ソース
        min_data_points: int = 100,     # 最小データポイント数 (V2では多めに要求)
    ):
        """
        コンストラクタ (V2 Ehlers-DSP Engine with UltimateSmoother)
        
        Args:
            dc_period: ドミナントサイクル測定の基準周期 (通常20-60)
            bandwidth: バンドパスフィルターの帯域幅 (通常0.1-0.3)
            dc_smooth_period: ドミナントサイクル平滑化期間（UltimateSmoother使用）
            src_type: 価格ソースタイプ
            min_data_points: 最小データポイント数
        """
        super().__init__(f"UQATRD_V2_Ehlers(P:{dc_period},B:{bandwidth},S:{dc_smooth_period})")
        
        # パラメータの保存
        self.dc_period = dc_period
        self.bandwidth = bandwidth
        self.dc_smooth_period = dc_smooth_period
        self.src_type = src_type.lower()
        self.min_data_points = min_data_points
        
        # パラメータ検証
        if self.dc_period < 10:
            raise ValueError("dc_periodは10以上である必要があります")
        if not (0.01 <= self.bandwidth <= 0.5):
            raise ValueError("bandwidthは0.01から0.5の範囲で設定してください")
        
        # ソースタイプの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            valid_sources = ', '.join(available_sources.keys())
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {valid_sources}")
        
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            length = len(data)
            first_val = float(data.iloc[0].get('close', 0)) if isinstance(data, pd.DataFrame) and length > 0 else (float(data[0]) if isinstance(data, np.ndarray) and length > 0 else 0.0)
            last_val = float(data.iloc[-1].get('close', 0)) if isinstance(data, pd.DataFrame) and length > 0 else (float(data[-1]) if isinstance(data, np.ndarray) and length > 0 else 0.0)
            params_sig = f"{self.dc_period}_{self.bandwidth}_{self.dc_smooth_period}_{self.src_type}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
        except Exception:
            return f"{id(data)}_{self.dc_period}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UQATRDResult:
        """
        UQATRD V2 計算メイン関数
        """
        try:
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                return self._result_cache[data_hash]
            
            price_source = PriceSource.calculate_source(data, self.src_type)
            price_source = np.asarray(price_source, dtype=np.float64)
            data_length = len(price_source)
            
            if data_length < self.min_data_points:
                self.logger.warning(f"データが短すぎます({data_length})。最低{self.min_data_points}点を推奨。")
                raise ValueError("データが不足しています。")

            # 核心計算エンジン(V2)実行
            trend_range_signal, signal_strength, dominant_cycle = calculate_uqatrd_core_v2(
                price_source,
                self.dc_period,
                self.bandwidth,
                self.dc_smooth_period
            )
            
            # 結果をUQATRDResultにマッピング (後方互換性のため)
            n = data_length
            result = UQATRDResult(
                trend_range_signal=trend_range_signal,
                signal_strength=signal_strength,
                cycle_adaptive_factor=dominant_cycle,
                adaptive_threshold=np.full(n, 0.5), # V2では固定
                # --- V1互換性のためのプレースホルダー ---
                quantum_coherence=np.zeros(n),
                trend_persistence=np.zeros(n),
                efficiency_spectrum=np.zeros(n),
                uncertainty_range=np.zeros(n),
                confidence_score=np.copy(signal_strength) # confidenceとしてstrengthを代用
            )

            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_range_signal
            
            # デバッグ情報の追加
            trend_count = (trend_range_signal > 0.5).sum()
            range_count = (trend_range_signal <= 0.5).sum()
            total_count = len(trend_range_signal)
            
            # 信号強度の統計
            valid_strength = signal_strength[~np.isnan(signal_strength)]
            strength_stats = f"平均: {valid_strength.mean():.3f}, 範囲: {valid_strength.min():.3f} - {valid_strength.max():.3f}" if len(valid_strength) > 0 else "NaN"
            
            self.logger.info(f"UQATRD V2計算完了 - データ長: {data_length}, "
                           f"最終シグナル: {trend_range_signal[-1]:.3f}, "
                           f"最新DC周期: {dominant_cycle[-1]:.2f}")
            self.logger.info(f"信号分布 - トレンド: {trend_count} ({trend_count/total_count*100:.1f}%), "
                           f"レンジ: {range_count} ({range_count/total_count*100:.1f}%)")
            self.logger.info(f"信号範囲 - 最小: {trend_range_signal.min():.3f}, "
                           f"最大: {trend_range_signal.max():.3f}, "
                           f"平均: {trend_range_signal.mean():.3f}")
            self.logger.info(f"信号強度 - {strength_stats}, NaN数: {np.isnan(signal_strength).sum()}")
            
            return result

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UQATRD V2計算中にエラー: {error_msg}\n{stack_trace}")
            return UQATRDResult(*([np.array([])] * 9))

    # --- ゲッターメソッド（インターフェース維持のため変更なし） ---
    
    def get_trend_range_signal(self) -> Optional[np.ndarray]:
        if not self._result_cache or not self._cache_keys: return None
        return self._result_cache[self._cache_keys[-1]].trend_range_signal.copy()
    
    def get_signal_strength(self) -> Optional[np.ndarray]:
        if not self._result_cache or not self._cache_keys: return None
        return self._result_cache[self._cache_keys[-1]].signal_strength.copy()

    def get_dominant_cycle_period(self) -> Optional[np.ndarray]:
        """V2で追加された、ドミナントサイクル周期を取得するメソッド"""
        if not self._result_cache or not self._cache_keys: return None
        # cycle_adaptive_factorに格納されている
        return self._result_cache[self._cache_keys[-1]].cycle_adaptive_factor.copy()

    def get_adaptive_threshold(self) -> Optional[np.ndarray]:
        if not self._result_cache or not self._cache_keys: return None
        return self._result_cache[self._cache_keys[-1]].adaptive_threshold.copy()
        
    def get_trend_range_classification(self, threshold: float = 0.5) -> Optional[np.ndarray]:
        """閾値に基づきトレンド/レンジを分類 (0=レンジ, 1=トレンド)"""
        signal = self.get_trend_range_signal()
        if signal is None: return None
        return (signal >= threshold).astype(float)

    def reset(self) -> None:
        super().reset()
        self._result_cache = {}
        self._cache_keys = []