#!/usr/bin/env python3
"""
Hyper FRAMA Channel Indicator

@indicators/hyper_adaptive_channel.py をベースに、ミッドラインをHyperFRAMAのみに固定した
シンプルなチャネルインジケーター

Author: Claude
Date: 2025-08-04
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
from .volatility.x_atr import XATR
from .trend_filter.hyper_er import HyperER


class MultiplierMode(Enum):
    """乗数モード"""
    FIXED = "fixed"
    DYNAMIC = "dynamic"


@dataclass
class HyperFRAMAChannelResult:
    """HyperFRAMAチャネル結果"""
    # チャネル値
    midline: np.ndarray
    upper_band: np.ndarray
    lower_band: np.ndarray
    bandwidth: np.ndarray
    
    # 動的パラメータ
    atr_values: np.ndarray
    multiplier_values: np.ndarray
    er_values: Optional[np.ndarray] = None
    
    # チャネル信号
    channel_position: np.ndarray = None  # -1: lower, 0: inside, 1: upper
    squeeze_signal: np.ndarray = None
    expansion_signal: np.ndarray = None
    
    # 統計情報
    channel_width_percentile: Optional[np.ndarray] = None
    volatility_regime: Optional[np.ndarray] = None


class HyperFRAMAChannel(Indicator):
    """
    HyperFRAMAチャネルインジケーター
    
    HyperFRAMAをミッドラインとし、HyperERベースの動的乗数適応を実装
    ハイパーアダプティブチャネルのシンプル版
    """
    
    def __init__(
        self,
        period: int = 14,
        multiplier_mode: str = "dynamic",
        fixed_multiplier: float = 2.5,
        src_type: str = "hlc3",
        
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
        
        # === X_ATR パラメータ ===
        x_atr_period: float = 12.0,
        x_atr_tr_method: str = 'atr',
        x_atr_smoother_type: str = 'frama',
        x_atr_src_type: str = 'close',
        x_atr_enable_kalman: bool = False,
        x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        x_atr_period_mode: str = 'dynamic',
        x_atr_cycle_detector_type: str = 'practical',
        x_atr_cycle_detector_cycle_part: float = 0.5,
        x_atr_cycle_detector_max_cycle: int = 55,
        x_atr_cycle_detector_min_cycle: int = 5,
        x_atr_cycle_period_multiplier: float = 1.0,
        x_atr_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # ミッドラインパラメータ
        x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        x_atr_enable_percentile_analysis: bool = True,
        x_atr_percentile_lookback_period: int = 50,
        x_atr_percentile_low_threshold: float = 0.25,
        x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        x_atr_smoother_params: Optional[Dict[str, Any]] = None,
        # カルマンフィルターパラメータ
        x_atr_kalman_params: Optional[Dict[str, Any]] = None,
        
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
        
        # === HyperFRAMAチャネル独自パラメータ ===
        enable_signals: bool = True,
        enable_percentile: bool = True,
        percentile_period: int = 100,
    ):
        """
        Parameters:
        -----------
        period : int
            基本期間
        multiplier_mode : str
            乗数モード ("fixed" or "dynamic")
        fixed_multiplier : float
            固定乗数値
        src_type : str
            価格ソースタイプ
        """
        
        super().__init__("HyperFRAMAChannel")
        
        # 基本パラメータ
        self.period = period
        self.multiplier_mode = MultiplierMode(multiplier_mode)
        self.fixed_multiplier = fixed_multiplier
        self.src_type = src_type
        
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
        
        # X_ATRパラメータ
        self.x_atr_params = {
            "period": x_atr_period,
            "tr_method": x_atr_tr_method,
            "smoother_type": x_atr_smoother_type,
            "src_type": x_atr_src_type,
            "enable_kalman": x_atr_enable_kalman,
            "kalman_type": x_atr_kalman_type,
            "period_mode": x_atr_period_mode,
            "cycle_detector_type": x_atr_cycle_detector_type,
            "cycle_detector_cycle_part": x_atr_cycle_detector_cycle_part,
            "cycle_detector_max_cycle": x_atr_cycle_detector_max_cycle,
            "cycle_detector_min_cycle": x_atr_cycle_detector_min_cycle,
            "cycle_period_multiplier": x_atr_cycle_period_multiplier,
            "cycle_detector_period_range": x_atr_cycle_detector_period_range,
            "midline_period": x_atr_midline_period,
            "enable_percentile_analysis": x_atr_enable_percentile_analysis,
            "percentile_lookback_period": x_atr_percentile_lookback_period,
            "percentile_low_threshold": x_atr_percentile_low_threshold,
            "percentile_high_threshold": x_atr_percentile_high_threshold,
            "smoother_params": x_atr_smoother_params,
            "kalman_params": x_atr_kalman_params
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
        
        # HyperFRAMA（ミッドライン固定）
        self.midline_indicator = HyperFRAMA(**self.hyper_frama_params)
        
        # X_ATR
        self.atr_indicator = XATR(**self.x_atr_params)
        
        # HyperER (動的乗数モード時のみ)
        if self.multiplier_mode == MultiplierMode.DYNAMIC:
            self.er_indicator = HyperER(**self.hyper_er_params)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperFRAMAChannelResult:
        """
        HyperFRAMAチャネルを計算
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            OHLCV市場データ
            
        Returns:
        --------
        HyperFRAMAChannelResult
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
            
            # ミッドライン計算（HyperFRAMA固定）
            midline = self._calculate_midline(df, price_source)
            
            # ATR計算
            atr_result = self.atr_indicator.calculate(df)
            atr_values = atr_result.values
            
            # 動的乗数計算
            multiplier_values, er_values = self._calculate_dynamic_multiplier(df, price_source)
            
            # チャネルバンド計算
            upper_band, lower_band, bandwidth = self._calculate_channel_bands(
                midline, atr_values, multiplier_values
            )
            
            # シグナル計算
            channel_position = None
            squeeze_signal = None
            expansion_signal = None
            channel_width_percentile = None
            volatility_regime = None
            
            if self.enable_signals:
                channel_position = self._calculate_channel_position(price_source, upper_band, lower_band)
                squeeze_signal, expansion_signal = self._calculate_squeeze_expansion_signals(bandwidth)
                
            if self.enable_percentile:
                channel_width_percentile = self._calculate_percentile_analysis(bandwidth)
                volatility_regime = self._calculate_volatility_regime(atr_values)
            
            # 結果構築
            result = HyperFRAMAChannelResult(
                midline=midline,
                upper_band=upper_band,
                lower_band=lower_band,
                bandwidth=bandwidth,
                atr_values=atr_values,
                multiplier_values=multiplier_values,
                er_values=er_values,
                channel_position=channel_position,
                squeeze_signal=squeeze_signal,
                expansion_signal=expansion_signal,
                channel_width_percentile=channel_width_percentile,
                volatility_regime=volatility_regime
            )
            
            # キャッシュ保存
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"HyperFRAMAChannel計算エラー: {e}")
            raise
    
    def _calculate_midline(self, data: pd.DataFrame, price_source: np.ndarray) -> np.ndarray:
        """ミッドライン計算（HyperFRAMA固定）"""
        
        result = self.midline_indicator.calculate(data)
        return result.frama_values  # 通常のFRAMA値を使用
    
    def _calculate_dynamic_multiplier(
        self, 
        data: pd.DataFrame, 
        price_source: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """動的乗数計算"""
        
        if self.multiplier_mode == MultiplierMode.FIXED:
            # 固定乗数
            multiplier_values = np.full(len(data), self.fixed_multiplier)
            return multiplier_values, None
        
        else:
            # 動的乗数 (HyperER使用)
            er_result = self.er_indicator.calculate(data)
            er_values = er_result.values
            
            # ERを乗数に変換 (0.5-3.0の範囲)
            multiplier_values = self._convert_er_to_multiplier(er_values)
            
            return multiplier_values, er_values
    
    @staticmethod
    @njit
    def _convert_er_to_multiplier(er_values: np.ndarray) -> np.ndarray:
        """効率比を乗数に変換"""
        
        multiplier_values = np.full(len(er_values), 2.5)
        
        for i in range(len(er_values)):
            if not np.isnan(er_values[i]):
                # ERが高い時は乗数を小さく (0.5-1.5)
                # ERが低い時は乗数を大きく (2.0-3.0)
                if er_values[i] > 0.65:
                    multiplier_values[i] = 0.5 + (1.0 - er_values[i]) * 2.0
                else:
                    multiplier_values[i] = 2.0 + (1.0 - er_values[i]) * 1.0
                    
                # 範囲制限
                multiplier_values[i] = max(0.5, min(3.0, multiplier_values[i]))
        
        return multiplier_values
    
    @staticmethod
    @njit
    def _calculate_channel_bands(
        midline: np.ndarray,
        atr_values: np.ndarray, 
        multiplier_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """チャネルバンド計算"""
        
        upper_band = np.full(len(midline), np.nan)
        lower_band = np.full(len(midline), np.nan)
        bandwidth = np.full(len(midline), np.nan)
        
        for i in range(len(midline)):
            if not (np.isnan(midline[i]) or np.isnan(atr_values[i]) or np.isnan(multiplier_values[i])):
                band_width = atr_values[i] * multiplier_values[i]
                upper_band[i] = midline[i] + band_width
                lower_band[i] = midline[i] - band_width
                bandwidth[i] = band_width * 2  # 全幅
        
        return upper_band, lower_band, bandwidth
    
    @staticmethod
    @njit
    def _calculate_channel_position(
        price_source: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """チャネル内ポジション計算"""
        
        position = np.full(len(price_source), 0.0)
        
        for i in range(len(price_source)):
            if not (np.isnan(price_source[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
                if price_source[i] >= upper_band[i]:
                    position[i] = 1.0  # 上バンドブレイク
                elif price_source[i] <= lower_band[i]:
                    position[i] = -1.0  # 下バンドブレイク
                else:
                    position[i] = 0.0  # チャネル内
        
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
    
    def _calculate_volatility_regime(self, atr_values: np.ndarray) -> np.ndarray:
        """ボラティリティレジーム分析"""
        
        if len(atr_values) < self.percentile_period:
            return np.full(len(atr_values), 1.0)  # 通常レジーム
        
        regime = np.full(len(atr_values), 1.0)
        
        for i in range(self.percentile_period - 1, len(atr_values)):
            if not np.isnan(atr_values[i]):
                window = atr_values[i - self.percentile_period + 1:i + 1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    current_percentile = (np.sum(valid_window <= atr_values[i]) / len(valid_window)) * 100
                    
                    if current_percentile >= 80:
                        regime[i] = 2.0  # 高ボラティリティ
                    elif current_percentile <= 20:
                        regime[i] = 0.0  # 低ボラティリティ
                    else:
                        regime[i] = 1.0  # 通常ボラティリティ
        
        return regime


# 便利関数
def create_hyper_frama_channel(
    period: int = 14,
    multiplier_mode: str = "dynamic",
    **kwargs
) -> HyperFRAMAChannel:
    """HyperFRAMAチャネル作成のヘルパー関数"""
    
    return HyperFRAMAChannel(
        period=period,
        multiplier_mode=multiplier_mode,
        **kwargs
    )