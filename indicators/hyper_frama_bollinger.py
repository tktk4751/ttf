#!/usr/bin/env python3
"""
HyperFRAMA Bollinger Bands Indicator

HyperFRAMAをミッドラインとしたボリンジャーバンド。
HyperERによる動的シグマ適応機能を実装。

Author: Claude
Date: 2025-08-09
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union, List
from numba import njit

from .indicator import Indicator
from .price_source import PriceSource
from .hyper_frama import HyperFRAMA
from .trend_filter.hyper_er import HyperER


class SigmaMode(Enum):
    """シグマモード"""
    FIXED = "fixed"
    DYNAMIC = "dynamic"


@dataclass
class HyperFRAMABollingerResult:
    """HyperFRAMAボリンジャーバンド結果"""
    # バンド値
    midline: np.ndarray
    upper_band: np.ndarray
    lower_band: np.ndarray
    bandwidth: np.ndarray
    
    # 統計値
    std_values: np.ndarray
    sigma_values: np.ndarray
    er_values: Optional[np.ndarray] = None
    
    # ボリンジャー信号
    band_position: np.ndarray = None  # -1: lower, 0: inside, 1: upper
    squeeze_signal: np.ndarray = None
    expansion_signal: np.ndarray = None
    
    # パーセント B
    percent_b: np.ndarray = None
    
    # 統計情報
    bandwidth_percentile: Optional[np.ndarray] = None
    volatility_regime: Optional[np.ndarray] = None


class HyperFRAMABollinger(Indicator):
    """
    HyperFRAMAボリンジャーバンドインジケーター
    
    HyperFRAMAをミッドラインとし、HyperERによる動的シグマ適応を実装
    ERが高い時はシグマ1.0、低い時はシグマ2.5に近づく
    """
    
    def __init__(
        self,
        period: int = 20,
        sigma_mode: str = "dynamic",
        fixed_sigma: float = 2.0,
        src_type: str = "close",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        hyper_frama_period: int = 16,
        hyper_frama_src_type: str = 'hl2',
        hyper_frama_fc: int = 1,
        hyper_frama_sc: int = 198,
        hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        hyper_frama_period_mode: str = 'fixed',
        hyper_frama_cycle_detector_type: str = 'hody_e',
        hyper_frama_lp_period: int = 13,
        hyper_frama_hp_period: int = 124,
        hyper_frama_cycle_part: float = 0.5,
        hyper_frama_max_cycle: int = 89,
        hyper_frama_min_cycle: int = 8,
        hyper_frama_max_output: int = 124,
        hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        hyper_frama_enable_indicator_adaptation: bool = True,
        hyper_frama_adaptation_indicator: str = 'hyper_er',
        hyper_frama_hyper_er_period: int = 14,
        hyper_frama_hyper_er_midline_period: int = 100,
        hyper_frama_hyper_adx_period: int = 14,
        hyper_frama_hyper_adx_midline_period: int = 100,
        hyper_frama_hyper_trend_index_period: int = 14,
        hyper_frama_hyper_trend_index_midline_period: int = 100,
        hyper_frama_fc_min: float = 1.0,
        hyper_frama_fc_max: float = 8.0,
        hyper_frama_sc_min: float = 50.0,
        hyper_frama_sc_max: float = 250.0,
        hyper_frama_period_min: int = 4,
        hyper_frama_period_max: int = 44,
        
        # === HyperER パラメータ ===
        hyper_er_period: int = 8,
        hyper_er_midline_period: int = 100,
        # ERパラメータ
        hyper_er_er_period: int = 13,
        hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        hyper_er_use_kalman_filter: bool = True,
        hyper_er_kalman_filter_type: str = 'simple',
        hyper_er_kalman_process_noise: float = 1e-5,
        hyper_er_kalman_min_observation_noise: float = 1e-6,
        hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,
        hyper_er_roofing_hp_cutoff: float = 55.0,
        hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        hyper_er_use_laguerre_filter: bool = False,
        hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        hyper_er_use_smoothing: bool = True,
        hyper_er_smoother_type: str = 'frama',
        hyper_er_smoother_period: int = 16,
        hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = True,
        hyper_er_detector_type: str = 'dft_dominant',
        hyper_er_lp_period: int = 13,
        hyper_er_hp_period: int = 124,
        hyper_er_cycle_part: float = 0.4,
        hyper_er_max_cycle: int = 124,
        hyper_er_min_cycle: int = 13,
        hyper_er_max_output: int = 89,
        hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,
        hyper_er_percentile_lookback_period: int = 50,
        hyper_er_percentile_low_threshold: float = 0.25,
        hyper_er_percentile_high_threshold: float = 0.75,
        
        # === HyperFRAMAボリンジャー独自パラメータ ===
        enable_signals: bool = True,
        enable_percentile: bool = True,
        percentile_period: int = 100,
        # シグマ範囲設定
        sigma_min: float = 1.0,
        sigma_max: float = 2.5,
    ):
        """
        Parameters:
        -----------
        period : int
            標準偏差計算期間
        sigma_mode : str
            シグマモード ("fixed" or "dynamic")
        fixed_sigma : float
            固定シグマ値
        src_type : str
            価格ソースタイプ
        sigma_min : float
            動的シグマ最小値 (ERが高い時)
        sigma_max : float
            動的シグマ最大値 (ERが低い時)
        """
        
        super().__init__("HyperFRAMABollinger")
        
        # 基本パラメータ
        self.period = period
        self.sigma_mode = SigmaMode(sigma_mode)
        self.fixed_sigma = fixed_sigma
        self.src_type = src_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # HyperFRAMAパラメータ
        self.hyper_frama_params = {
            "period": hyper_frama_period,
            "src_type": hyper_frama_src_type,
            "fc": hyper_frama_fc,
            "sc": hyper_frama_sc,
            "alpha_multiplier": hyper_frama_alpha_multiplier,
            "period_mode": hyper_frama_period_mode,
            "cycle_detector_type": hyper_frama_cycle_detector_type,
            "lp_period": hyper_frama_lp_period,
            "hp_period": hyper_frama_hp_period,
            "cycle_part": hyper_frama_cycle_part,
            "max_cycle": hyper_frama_max_cycle,
            "min_cycle": hyper_frama_min_cycle,
            "max_output": hyper_frama_max_output,
            "min_output": hyper_frama_min_output,
            "enable_indicator_adaptation": hyper_frama_enable_indicator_adaptation,
            "adaptation_indicator": hyper_frama_adaptation_indicator,
            "hyper_er_period": hyper_frama_hyper_er_period,
            "hyper_er_midline_period": hyper_frama_hyper_er_midline_period,
            "hyper_adx_period": hyper_frama_hyper_adx_period,
            "hyper_adx_midline_period": hyper_frama_hyper_adx_midline_period,
            "hyper_trend_index_period": hyper_frama_hyper_trend_index_period,
            "hyper_trend_index_midline_period": hyper_frama_hyper_trend_index_midline_period,
            "fc_min": hyper_frama_fc_min,
            "fc_max": hyper_frama_fc_max,
            "sc_min": hyper_frama_sc_min,
            "sc_max": hyper_frama_sc_max,
            "period_min": hyper_frama_period_min,
            "period_max": hyper_frama_period_max
        }
        
        # HyperERパラメータ
        self.hyper_er_params = {
            "period": hyper_er_period,
            "midline_period": hyper_er_midline_period,
            "er_period": hyper_er_er_period,
            "er_src_type": hyper_er_er_src_type,
            "use_kalman_filter": hyper_er_use_kalman_filter,
            "kalman_filter_type": hyper_er_kalman_filter_type,
            "kalman_process_noise": hyper_er_kalman_process_noise,
            "kalman_min_observation_noise": hyper_er_kalman_min_observation_noise,
            "kalman_adaptation_window": hyper_er_kalman_adaptation_window,
            "use_roofing_filter": hyper_er_use_roofing_filter,
            "roofing_hp_cutoff": hyper_er_roofing_hp_cutoff,
            "roofing_ss_band_edge": hyper_er_roofing_ss_band_edge,
            "use_laguerre_filter": hyper_er_use_laguerre_filter,
            "laguerre_gamma": hyper_er_laguerre_gamma,
            "use_smoothing": hyper_er_use_smoothing,
            "smoother_type": hyper_er_smoother_type,
            "smoother_period": hyper_er_smoother_period,
            "smoother_src_type": hyper_er_smoother_src_type,
            "use_dynamic_period": hyper_er_use_dynamic_period,
            "detector_type": hyper_er_detector_type,
            "lp_period": hyper_er_lp_period,
            "hp_period": hyper_er_hp_period,
            "cycle_part": hyper_er_cycle_part,
            "max_cycle": hyper_er_max_cycle,
            "min_cycle": hyper_er_min_cycle,
            "max_output": hyper_er_max_output,
            "min_output": hyper_er_min_output,
            "enable_percentile_analysis": hyper_er_enable_percentile_analysis,
            "percentile_lookback_period": hyper_er_percentile_lookback_period,
            "percentile_low_threshold": hyper_er_percentile_low_threshold,
            "percentile_high_threshold": hyper_er_percentile_high_threshold
        }
        
        # その他パラメータ
        self.enable_signals = enable_signals
        self.enable_percentile = enable_percentile
        self.percentile_period = percentile_period
        
        # インジケーターインスタンス
        self._setup_indicators()
        
        # キャッシュ
        self._cache = {}
    
    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """データを準備・正規化"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if data.shape[1] >= 5:  # OHLCV
                return pd.DataFrame({
                    'open': data[:, 0],
                    'high': data[:, 1], 
                    'low': data[:, 2],
                    'close': data[:, 3],
                    'volume': data[:, 4]
                })
            else:
                raise ValueError("NumPy配列はOHLCV形式である必要があります")
        else:
            raise ValueError("サポートされていないデータ型です")
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """キャッシュキーを生成"""
        return f"{self.name}_{len(data)}_{hash(str(data.iloc[0]) + str(data.iloc[-1]))}"
        
    def _setup_indicators(self):
        """必要なインジケーターインスタンスを初期化"""
        
        # HyperFRAMA（ミッドライン）
        self.midline_indicator = HyperFRAMA(**self.hyper_frama_params)
        
        # HyperER (動的シグマモード時のみ)
        if self.sigma_mode == SigmaMode.DYNAMIC:
            self.er_indicator = HyperER(**self.hyper_er_params)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperFRAMABollingerResult:
        """
        HyperFRAMAボリンジャーバンドを計算
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            OHLCV市場データ
            
        Returns:
        --------
        HyperFRAMABollingerResult
            計算結果
        """
        
        try:
            # データ前処理
            df = self._prepare_data(data)
            cache_key = self._generate_cache_key(df)
            
            # キャッシュチェック
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # 価格ソース計算
            price_source = PriceSource.calculate_source(df, self.src_type)
            
            # ミッドライン計算（HyperFRAMA）
            midline = self._calculate_midline(df)
            
            # 標準偏差計算
            std_values = self._calculate_standard_deviation(price_source, midline)
            
            # 動的シグマ計算
            sigma_values, er_values = self._calculate_dynamic_sigma(df)
            
            # ボリンジャーバンド計算
            upper_band, lower_band, bandwidth = self._calculate_bollinger_bands(
                midline, std_values, sigma_values
            )
            
            # パーセント B計算
            percent_b = self._calculate_percent_b(price_source, upper_band, lower_band)
            
            # シグナル計算
            band_position = None
            squeeze_signal = None
            expansion_signal = None
            bandwidth_percentile = None
            volatility_regime = None
            
            if self.enable_signals:
                band_position = self._calculate_band_position(price_source, upper_band, lower_band)
                squeeze_signal, expansion_signal = self._calculate_squeeze_expansion_signals(bandwidth)
                
            if self.enable_percentile:
                bandwidth_percentile = self._calculate_percentile_analysis(bandwidth)
                volatility_regime = self._calculate_volatility_regime(std_values)
            
            # 結果構築
            result = HyperFRAMABollingerResult(
                midline=midline,
                upper_band=upper_band,
                lower_band=lower_band,
                bandwidth=bandwidth,
                std_values=std_values,
                sigma_values=sigma_values,
                er_values=er_values,
                band_position=band_position,
                squeeze_signal=squeeze_signal,
                expansion_signal=expansion_signal,
                percent_b=percent_b,
                bandwidth_percentile=bandwidth_percentile,
                volatility_regime=volatility_regime
            )
            
            # キャッシュ保存
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"HyperFRAMABollinger計算エラー: {e}")
            raise
    
    def _calculate_midline(self, data: pd.DataFrame) -> np.ndarray:
        """ミッドライン計算（HyperFRAMA）"""
        
        result = self.midline_indicator.calculate(data)
        return result.frama_values
    
    def _calculate_standard_deviation(
        self, 
        price_source: np.ndarray, 
        midline: np.ndarray
    ) -> np.ndarray:
        """標準偏差計算"""
        
        if len(price_source) < self.period:
            return np.full(len(price_source), np.nan)
        
        std_values = np.full(len(price_source), np.nan)
        
        for i in range(self.period - 1, len(price_source)):
            if not np.isnan(midline[i]):
                # 移動窓での標準偏差計算
                window_prices = price_source[i - self.period + 1:i + 1]
                window_midline = midline[i - self.period + 1:i + 1]
                
                # 有効なデータのみ使用
                valid_mask = ~(np.isnan(window_prices) | np.isnan(window_midline))
                if np.sum(valid_mask) >= self.period // 2:
                    valid_prices = window_prices[valid_mask]
                    valid_midline = window_midline[valid_mask]
                    
                    # 各点でのミッドラインからの偏差の標準偏差
                    deviations = valid_prices - valid_midline
                    std_values[i] = np.std(deviations, ddof=1) if len(deviations) > 1 else 0.0
        
        return std_values
    
    def _calculate_dynamic_sigma(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """動的シグマ計算"""
        
        if self.sigma_mode == SigmaMode.FIXED:
            # 固定シグマ
            sigma_values = np.full(len(data), self.fixed_sigma)
            return sigma_values, None
        
        else:
            # 動的シグマ (HyperER使用)
            er_result = self.er_indicator.calculate(data)
            er_values = er_result.values
            
            # ERをシグマに変換
            sigma_values = self._convert_er_to_sigma(er_values)
            
            return sigma_values, er_values
    
    @staticmethod
    @njit
    def _convert_er_to_sigma(er_values: np.ndarray) -> np.ndarray:
        """効率比をシグマに変換"""
        
        sigma_values = np.full(len(er_values), 2.0)
        sigma_min = 1.0
        sigma_max = 2.5
        
        for i in range(len(er_values)):
            if not np.isnan(er_values[i]):
                # ERが高い時はシグマを小さく (1.0に近づく)
                # ERが低い時はシグマを大きく (2.5に近づく)
                normalized_er = max(0.0, min(1.0, er_values[i]))
                sigma_values[i] = sigma_max - (normalized_er * (sigma_max - sigma_min))
                
                # 範囲制限
                sigma_values[i] = max(sigma_min, min(sigma_max, sigma_values[i]))
        
        return sigma_values
    
    @staticmethod
    @njit
    def _calculate_bollinger_bands(
        midline: np.ndarray,
        std_values: np.ndarray, 
        sigma_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド計算"""
        
        upper_band = np.full(len(midline), np.nan)
        lower_band = np.full(len(midline), np.nan)
        bandwidth = np.full(len(midline), np.nan)
        
        for i in range(len(midline)):
            if not (np.isnan(midline[i]) or np.isnan(std_values[i]) or np.isnan(sigma_values[i])):
                band_width = std_values[i] * sigma_values[i]
                upper_band[i] = midline[i] + band_width
                lower_band[i] = midline[i] - band_width
                bandwidth[i] = band_width * 2  # 全幅
        
        return upper_band, lower_band, bandwidth
    
    @staticmethod
    @njit
    def _calculate_percent_b(
        price_source: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """パーセント B計算"""
        
        percent_b = np.full(len(price_source), np.nan)
        
        for i in range(len(price_source)):
            if not (np.isnan(price_source[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
                band_range = upper_band[i] - lower_band[i]
                if band_range > 0:
                    percent_b[i] = (price_source[i] - lower_band[i]) / band_range
        
        return percent_b
    
    @staticmethod
    @njit
    def _calculate_band_position(
        price_source: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """バンド内ポジション計算"""
        
        position = np.full(len(price_source), 0.0)
        
        for i in range(len(price_source)):
            if not (np.isnan(price_source[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
                if price_source[i] >= upper_band[i]:
                    position[i] = 1.0  # 上バンドブレイク
                elif price_source[i] <= lower_band[i]:
                    position[i] = -1.0  # 下バンドブレイク
                else:
                    position[i] = 0.0  # バンド内
        
        return position
    
    def _calculate_squeeze_expansion_signals(
        self, 
        bandwidth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """スクイーズ・エクスパンション信号計算"""
        
        if len(bandwidth) < self.period:
            return np.full(len(bandwidth), 0.0), np.full(len(bandwidth), 0.0)
        
        # バンド幅の移動平均とボラティリティ
        bandwidth_sma = pd.Series(bandwidth).rolling(self.period, min_periods=1).mean().values
        bandwidth_std = pd.Series(bandwidth).rolling(self.period, min_periods=1).std().values
        
        squeeze_signal = np.full(len(bandwidth), 0.0)
        expansion_signal = np.full(len(bandwidth), 0.0)
        
        for i in range(self.period, len(bandwidth)):
            if not (np.isnan(bandwidth[i]) or np.isnan(bandwidth_sma[i]) or np.isnan(bandwidth_std[i])):
                # スクイーズ: バンド幅が平均 - 1σ以下
                squeeze_threshold = bandwidth_sma[i] - bandwidth_std[i]
                if bandwidth[i] <= squeeze_threshold:
                    squeeze_signal[i] = 1.0
                
                # エクスパンション: バンド幅が平均 + 1σ以上
                expansion_threshold = bandwidth_sma[i] + bandwidth_std[i]
                if bandwidth[i] >= expansion_threshold:
                    expansion_signal[i] = 1.0
        
        return squeeze_signal, expansion_signal
    
    def _calculate_percentile_analysis(self, bandwidth: np.ndarray) -> np.ndarray:
        """パーセンタイル分析"""
        
        if len(bandwidth) < self.percentile_period:
            return np.full(len(bandwidth), 50.0)
        
        percentile_values = np.full(len(bandwidth), np.nan)
        
        for i in range(self.percentile_period - 1, len(bandwidth)):
            if not np.isnan(bandwidth[i]):
                window = bandwidth[i - self.percentile_period + 1:i + 1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    current_percentile = (np.sum(valid_window <= bandwidth[i]) / len(valid_window)) * 100
                    percentile_values[i] = current_percentile
        
        return percentile_values
    
    def _calculate_volatility_regime(self, std_values: np.ndarray) -> np.ndarray:
        """ボラティリティレジーム分析"""
        
        if len(std_values) < self.percentile_period:
            return np.full(len(std_values), 1.0)  # 通常レジーム
        
        regime = np.full(len(std_values), 1.0)
        
        for i in range(self.percentile_period - 1, len(std_values)):
            if not np.isnan(std_values[i]):
                window = std_values[i - self.percentile_period + 1:i + 1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    current_percentile = (np.sum(valid_window <= std_values[i]) / len(valid_window)) * 100
                    
                    if current_percentile >= 80:
                        regime[i] = 2.0  # 高ボラティリティ
                    elif current_percentile <= 20:
                        regime[i] = 0.0  # 低ボラティリティ
                    else:
                        regime[i] = 1.0  # 通常ボラティリティ
        
        return regime


# 便利関数
def create_hyper_frama_bollinger(
    period: int = 20,
    sigma_mode: str = "dynamic",
    **kwargs
) -> HyperFRAMABollinger:
    """HyperFRAMAボリンジャーバンド作成のヘルパー関数"""
    
    return HyperFRAMABollinger(
        period=period,
        sigma_mode=sigma_mode,
        **kwargs
    )