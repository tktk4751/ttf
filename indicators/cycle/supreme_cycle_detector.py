#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supreme Cycle Detector - 究極のサイクル検出器

4つの高度なサイクル検出アルゴリズムを統合し、
市場の真のサイクルを超高精度・超低遅延・超追従性・超適応性で検出します。

統合アルゴリズム:
1. Homodyne Discriminator (HoDy) - リアルタイム性
2. Dual Differentiator (DuDi) - 高感度検出
3. Phase Accumulation (PhAc) - 安定性
4. Discrete Fourier Transform (DFT) - 高精度

特徴:
- 適応的重み付けアンサンブル
- カルマンフィルターによるノイズ除去
- 機械学習的な信頼度評価
- 動的パラメータ最適化
"""

from typing import Union, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from numba import jit, prange, float64, int32
from dataclasses import dataclass

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from .ehlers_hody_dce import calculate_hody_dce_numba
from .ehlers_dudi_dce import calculate_dudi_dce_numba
from .ehlers_phac_dce import calculate_phac_dce_numba
from .ehlers_dft_dominant_cycle import calculate_dft_dominant_cycle_numba
from ..kalman.unscented_kalman_filter import UnscentedKalmanFilter
from ..price_source import PriceSource


@dataclass
class SupremeCycleResult(DominantCycleResult):
    """Supreme Cycle Detectorの拡張結果"""
    component_cycles: Dict[str, np.ndarray]  # 各コンポーネントのサイクル値
    weights: Dict[str, np.ndarray]  # 各コンポーネントの重み
    confidence: np.ndarray  # 検出の信頼度 (0-1)
    volatility_state: np.ndarray  # ボラティリティ状態
    

@jit(nopython=True)
def calculate_adaptive_weights(
    hody_cycles: np.ndarray,
    dudi_cycles: np.ndarray,
    phac_cycles: np.ndarray,
    dft_cycles: np.ndarray,
    lookback: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    各サイクル検出器の適応的重みを計算
    
    Args:
        hody_cycles: HoDyサイクル値
        dudi_cycles: DuDiサイクル値
        phac_cycles: PhAcサイクル値
        dft_cycles: DFTサイクル値
        lookback: 評価期間
        
    Returns:
        各検出器の重みと信頼度
    """
    n = len(hody_cycles)
    
    # 重み配列の初期化
    w_hody = np.zeros(n)
    w_dudi = np.zeros(n)
    w_phac = np.zeros(n)
    w_dft = np.zeros(n)
    confidence = np.zeros(n)
    
    for i in range(lookback, n):
        # 各検出器の安定性を評価（標準偏差の逆数）
        hody_std = np.std(hody_cycles[i-lookback:i])
        dudi_std = np.std(dudi_cycles[i-lookback:i])
        phac_std = np.std(phac_cycles[i-lookback:i])
        dft_std = np.std(dft_cycles[i-lookback:i])
        
        # 安定性スコア（標準偏差が小さいほど高スコア）
        hody_stability = 1.0 / (1.0 + hody_std)
        dudi_stability = 1.0 / (1.0 + dudi_std)
        phac_stability = 1.0 / (1.0 + phac_std)
        dft_stability = 1.0 / (1.0 + dft_std)
        
        # 一貫性スコア（他の検出器との相関）
        values = np.array([hody_cycles[i], dudi_cycles[i], phac_cycles[i], dft_cycles[i]])
        mean_val = np.mean(values)
        
        hody_consistency = 1.0 / (1.0 + abs(hody_cycles[i] - mean_val))
        dudi_consistency = 1.0 / (1.0 + abs(dudi_cycles[i] - mean_val))
        phac_consistency = 1.0 / (1.0 + abs(phac_cycles[i] - mean_val))
        dft_consistency = 1.0 / (1.0 + abs(dft_cycles[i] - mean_val))
        
        # 総合スコア
        hody_score = hody_stability * hody_consistency * 1.2  # リアルタイム性ボーナス
        dudi_score = dudi_stability * dudi_consistency * 1.1  # 感度ボーナス
        phac_score = phac_stability * phac_consistency * 1.0  # バランス型
        dft_score = dft_stability * dft_consistency * 1.3   # 精度ボーナス
        
        # 正規化
        total_score = hody_score + dudi_score + phac_score + dft_score
        if total_score > 0:
            w_hody[i] = hody_score / total_score
            w_dudi[i] = dudi_score / total_score
            w_phac[i] = phac_score / total_score
            w_dft[i] = dft_score / total_score
        else:
            # デフォルト重み
            w_hody[i] = 0.25
            w_dudi[i] = 0.25
            w_phac[i] = 0.25
            w_dft[i] = 0.25
        
        # 信頼度計算（値の分散が小さいほど高信頼度）
        variance = np.var(values)
        confidence[i] = 1.0 / (1.0 + variance / mean_val if mean_val > 0 else 1.0)
    
    # 初期期間はデフォルト値
    for i in range(lookback):
        w_hody[i] = 0.25
        w_dudi[i] = 0.25
        w_phac[i] = 0.25
        w_dft[i] = 0.25
        confidence[i] = 0.5
    
    return w_hody, w_dudi, w_phac, w_dft, confidence


@jit(nopython=True)
def calculate_supreme_cycle_core(
    hody_cycles: np.ndarray,
    dudi_cycles: np.ndarray,
    phac_cycles: np.ndarray,
    dft_cycles: np.ndarray,
    w_hody: np.ndarray,
    w_dudi: np.ndarray,
    w_phac: np.ndarray,
    w_dft: np.ndarray,
    confidence: np.ndarray,
    smoothing_factor: float = 0.1
) -> np.ndarray:
    """
    重み付きアンサンブルによる最終サイクル値の計算
    
    Args:
        各サイクル値と重み
        smoothing_factor: 平滑化係数
        
    Returns:
        統合されたサイクル値
    """
    n = len(hody_cycles)
    supreme_cycles = np.zeros(n)
    
    for i in range(n):
        # 重み付き平均
        weighted_sum = (
            w_hody[i] * hody_cycles[i] +
            w_dudi[i] * dudi_cycles[i] +
            w_phac[i] * phac_cycles[i] +
            w_dft[i] * dft_cycles[i]
        )
        
        # 信頼度による調整
        if i > 0:
            # 低信頼度の場合は前回値に近づける
            supreme_cycles[i] = (
                confidence[i] * weighted_sum +
                (1 - confidence[i]) * supreme_cycles[i-1]
            )
            
            # 追加の平滑化
            supreme_cycles[i] = (
                (1 - smoothing_factor) * supreme_cycles[i] +
                smoothing_factor * supreme_cycles[i-1]
            )
        else:
            supreme_cycles[i] = weighted_sum
    
    return supreme_cycles


@jit(nopython=True) 
def detect_volatility_state(price: np.ndarray, window: int = 20) -> np.ndarray:
    """
    ボラティリティ状態を検出（高速版）
    
    Args:
        price: 価格データ
        window: 計算ウィンドウ
        
    Returns:
        ボラティリティ状態 (0: 低, 1: 中, 2: 高)
    """
    n = len(price)
    vol_state = np.zeros(n)
    
    for i in range(window, n):
        # 価格変化率の標準偏差
        returns = np.zeros(window)
        for j in range(window):
            if price[i-j-1] != 0:
                returns[j] = (price[i-j] - price[i-j-1]) / price[i-j-1]
        
        volatility = np.std(returns)
        
        # 閾値による分類
        if volatility < 0.005:  # 0.5%未満
            vol_state[i] = 0  # 低ボラティリティ
        elif volatility < 0.015:  # 1.5%未満
            vol_state[i] = 1  # 中ボラティリティ
        else:
            vol_state[i] = 2  # 高ボラティリティ
    
    # 初期期間
    for i in range(window):
        vol_state[i] = 1  # デフォルトは中
    
    return vol_state


class SupremeCycleDetector(EhlersDominantCycle):
    """
    Supreme Cycle Detector - 究極のサイクル検出器
    
    4つの高度なサイクル検出アルゴリズムを統合し、
    適応的重み付けアンサンブルにより最高精度のサイクル検出を実現。
    
    特徴:
    - 複数アルゴリズムの長所を統合
    - 動的重み調整による適応性
    - カルマンフィルターによるノイズ除去
    - ボラティリティ適応
    - 信頼度評価システム
    """
    
    def __init__(
        self,
        # 共通パラメータ
        lp_period: int = 10,
        hp_period: int = 48,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'hlc3',  # Ultimate Smoother HLC3をデフォルトに
        
        # DFT固有パラメータ
        dft_window: int = 50,
        
        # Supreme固有パラメータ
        use_ukf: bool = True,
        ukf_alpha: float = 0.001,
        smoothing_factor: float = 0.1,
        weight_lookback: int = 20,
        adaptive_params: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分の倍率
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ（スムーズ化ソースも使用可能）
            dft_window: DFT分析ウィンドウ
            use_ukf: UKFフィルタリングを使用するか
            ukf_alpha: UKFのアルファ値
            smoothing_factor: 最終平滑化係数
            weight_lookback: 重み計算の評価期間
            adaptive_params: パラメータの動的調整を行うか
        """
        super().__init__(
            f"SupremeCycleDetector({lp_period}, {hp_period})",
            cycle_part,
            hp_period,
            lp_period,
            max_output,
            min_output
        )
        
        # パラメータ保存
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.src_type = src_type
        self.dft_window = dft_window
        self.use_ukf = use_ukf
        self.ukf_alpha = ukf_alpha
        self.smoothing_factor = smoothing_factor
        self.weight_lookback = weight_lookback
        self.adaptive_params = adaptive_params
        
        # UKFフィルターの初期化
        if self.use_ukf:
            self.ukf = UnscentedKalmanFilter(
                src_type='close',
                alpha=ukf_alpha,
                beta=2.0,
                kappa=0.0,
                process_noise_scale=0.001,
                adaptive_noise=True
            )
        
        # コンポーネント検出器のキャッシュ
        self._component_cache = {}
        
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Supreme Cycle検出を実行
        
        Args:
            data: 価格データ
            
        Returns:
            検出されたサイクル値
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash
            
            # 価格ソースの取得（スムーズ化ソース対応）
            price = PriceSource.calculate_source(data, self.src_type)
            
            # UKFフィルタリング（オプション）
            if self.use_ukf:
                # UKFは1次元配列として処理
                ukf_result = self.ukf.calculate(price)
                filtered_price = ukf_result.filtered_values
            else:
                filtered_price = price
            
            # ボラティリティ状態の検出
            vol_state = detect_volatility_state(filtered_price)
            
            # 適応的パラメータ調整
            if self.adaptive_params:
                # 高ボラティリティ時は短期パラメータを使用
                high_vol_mask = vol_state == 2
                if np.any(high_vol_mask):
                    # 高ボラティリティ期間の割合に応じて調整
                    high_vol_ratio = np.sum(high_vol_mask) / len(vol_state)
                    final_lp = max(5, int(self.lp_period - 2 * high_vol_ratio))
                    final_hp = min(60, int(self.hp_period + 5 * high_vol_ratio))
                else:
                    final_lp = self.lp_period
                    final_hp = self.hp_period
            else:
                final_lp = self.lp_period
                final_hp = self.hp_period
            
            # 各コンポーネントのサイクル検出
            self.logger.debug("コンポーネントサイクルを計算中...")
            
            # HoDy
            hody_cycles, hody_raw, hody_smooth = calculate_hody_dce_numba(
                filtered_price, final_lp, final_hp, 
                self.cycle_part, self.max_output, self.min_output
            )
            
            # DuDi
            dudi_cycles, dudi_raw, dudi_smooth = calculate_dudi_dce_numba(
                filtered_price, final_lp, final_hp,
                self.cycle_part, self.max_output, self.min_output
            )
            
            # PhAc
            phac_cycles, phac_raw, phac_smooth = calculate_phac_dce_numba(
                filtered_price, final_lp, final_hp,
                self.cycle_part, self.max_output, self.min_output
            )
            
            # DFT
            dft_cycles, dft_raw, dft_smooth = calculate_dft_dominant_cycle_numba(
                filtered_price, self.dft_window,
                self.cycle_part, self.max_output, self.min_output
            )
            
            # 適応的重み計算
            self.logger.debug("適応的重みを計算中...")
            w_hody, w_dudi, w_phac, w_dft, confidence = calculate_adaptive_weights(
                hody_cycles, dudi_cycles, phac_cycles, dft_cycles,
                self.weight_lookback
            )
            
            # 最終サイクル値の計算
            self.logger.debug("最終サイクル値を計算中...")
            supreme_cycles = calculate_supreme_cycle_core(
                hody_cycles, dudi_cycles, phac_cycles, dft_cycles,
                w_hody, w_dudi, w_phac, w_dft, confidence,
                self.smoothing_factor
            )
            
            # 最終的な出力調整
            supreme_cycles = np.clip(supreme_cycles, self.min_output, self.max_output)
            
            # 結果の保存
            self._result = SupremeCycleResult(
                values=supreme_cycles,
                raw_period=np.mean([hody_raw, dudi_raw, phac_raw, dft_raw], axis=0),
                smooth_period=supreme_cycles,
                component_cycles={
                    'hody': hody_cycles,
                    'dudi': dudi_cycles,
                    'phac': phac_cycles,
                    'dft': dft_cycles
                },
                weights={
                    'hody': w_hody,
                    'dudi': w_dudi,
                    'phac': w_phac,
                    'dft': w_dft
                },
                confidence=confidence,
                volatility_state=vol_state
            )
            
            self._values = supreme_cycles
            
            # デバッグ情報
            avg_confidence = np.mean(confidence)
            self.logger.info(f"Supreme Cycle検出完了 - 平均信頼度: {avg_confidence:.2%}")
            
            return supreme_cycles
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"SupremeCycleDetector計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_component_info(self) -> Optional[Dict]:
        """
        コンポーネント情報を取得
        
        Returns:
            各コンポーネントの詳細情報
        """
        if self._result is None or not isinstance(self._result, SupremeCycleResult):
            return None
        
        result = self._result
        info = {
            'average_confidence': float(np.mean(result.confidence)),
            'component_weights': {
                'hody': float(np.mean(result.weights['hody'])),
                'dudi': float(np.mean(result.weights['dudi'])),
                'phac': float(np.mean(result.weights['phac'])),
                'dft': float(np.mean(result.weights['dft']))
            },
            'volatility_distribution': {
                'low': float(np.sum(result.volatility_state == 0) / len(result.volatility_state)),
                'medium': float(np.sum(result.volatility_state == 1) / len(result.volatility_state)),
                'high': float(np.sum(result.volatility_state == 2) / len(result.volatility_state))
            }
        }
        
        return info
    
    def get_best_component(self, index: int = -1) -> str:
        """
        指定時点で最も重みの高いコンポーネントを取得
        
        Args:
            index: 時点インデックス（デフォルトは最新）
            
        Returns:
            最高重みのコンポーネント名
        """
        if self._result is None or not isinstance(self._result, SupremeCycleResult):
            return "unknown"
        
        weights = self._result.weights
        w_values = [
            weights['hody'][index],
            weights['dudi'][index],
            weights['phac'][index],
            weights['dft'][index]
        ]
        
        components = ['hody', 'dudi', 'phac', 'dft']
        best_idx = np.argmax(w_values)
        
        return components[best_idx]