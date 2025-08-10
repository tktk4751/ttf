#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64
import math

from ..indicator import Indicator
from ..price_source import PriceSource
from ..smoother.unified_smoother import UnifiedSmoother
from ..utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        EHLERS_UNIFIED_DC_AVAILABLE = True
    except ImportError:
        EhlersUnifiedDC = None
        EHLERS_UNIFIED_DC_AVAILABLE = False

try:
    from ..kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False


@dataclass
class XHurstResult:
    """X_Hurstの計算結果"""
    values: np.ndarray               # X_Hurst値（0-1の範囲、高い値=持続性トレンド）
    raw_hurst: np.ndarray           # 生のDFAハースト値（カルマンフィルター適用前）
    filtered_hurst: np.ndarray      # カルマンフィルター適用後のハースト値
    smoothed_hurst: np.ndarray      # 平滑化されたハースト値（オプション）
    midline: np.ndarray             # ミッドライン値
    trend_signal: np.ndarray        # トレンド判定信号（1=持続性トレンド、-1=反持続性レンジ）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_dfa_hurst_numba(
    prices: np.ndarray,
    period: int,
    scales: np.ndarray
) -> np.ndarray:
    """
    DFA (Detrended Fluctuation Analysis) によるハースト指数を計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        period: 計算期間
        scales: DFAスケール配列
        
    Returns:
        DFAハースト指数値の配列（0.0-1.0の範囲）
    """
    length = len(prices)
    hurst_values = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(period, length):
        window_start = i - period + 1
        window_data = prices[window_start:i+1]
        
        # 対数リターン計算
        log_returns = np.zeros(len(window_data) - 1)
        for j in range(len(window_data) - 1):
            if window_data[j] > 0 and window_data[j+1] > 0:
                log_returns[j] = math.log(window_data[j+1] / window_data[j])
            else:
                log_returns[j] = 0.0
        
        if len(log_returns) < int(scales[-1]) + 5:
            continue
        
        # プロファイル計算（累積合計）
        mean_return = np.mean(log_returns)
        profile = np.zeros(len(log_returns) + 1)
        for j in range(len(log_returns)):
            profile[j+1] = profile[j] + (log_returns[j] - mean_return)
        
        # 各スケールでの揺動計算
        log_scales = np.zeros(len(scales))
        log_fluctuations = np.zeros(len(scales))
        
        valid_count = 0
        for k, scale in enumerate(scales):
            scale_int = int(scale)
            if scale_int >= len(profile) - 1:
                continue
            
            n_segments = len(profile) // scale_int
            if n_segments == 0:
                continue
            
            segment_variances = np.zeros(n_segments)
            
            for seg_idx in range(n_segments):
                start_idx = seg_idx * scale_int
                end_idx = start_idx + scale_int
                
                if end_idx <= len(profile):
                    segment = profile[start_idx:end_idx]
                    
                    # 線形トレンド除去 (y = ax + b の最小二乗法)
                    x_vals = np.arange(len(segment), dtype=np.float64)
                    n_seg = len(segment)
                    
                    sum_x = np.sum(x_vals)
                    sum_y = np.sum(segment)
                    sum_xy = np.sum(x_vals * segment)
                    sum_x2 = np.sum(x_vals ** 2)
                    
                    denominator = n_seg * sum_x2 - sum_x ** 2
                    if abs(denominator) > 1e-10:
                        a = (n_seg * sum_xy - sum_x * sum_y) / denominator
                        b = (sum_y - a * sum_x) / n_seg
                        
                        # トレンド除去された残差の分散計算
                        residual_variance = 0.0
                        for j in range(len(segment)):
                            detrended = segment[j] - (a * x_vals[j] + b)
                            residual_variance += detrended ** 2
                        
                        segment_variances[seg_idx] = residual_variance / n_seg
            
            # 平均揺動計算
            mean_variance = np.mean(segment_variances[:n_segments])
            if mean_variance > 1e-10:
                log_scales[valid_count] = math.log(scale)
                log_fluctuations[valid_count] = 0.5 * math.log(mean_variance)
                valid_count += 1
        
        # ハースト指数をlog-log回帰の傾きから計算
        if valid_count >= 3:
            valid_log_scales = log_scales[:valid_count]
            valid_log_fluctuations = log_fluctuations[:valid_count]
            
            # 最小二乗法
            n_points = valid_count
            sum_x = np.sum(valid_log_scales)
            sum_y = np.sum(valid_log_fluctuations)
            sum_xy = np.sum(valid_log_scales * valid_log_fluctuations)
            sum_x2 = np.sum(valid_log_scales ** 2)
            
            denominator = n_points * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                hurst_values[i] = slope
    
    # 0-1の範囲にクリップ
    for i in range(length):
        if not np.isnan(hurst_values[i]):
            hurst_values[i] = max(0.0, min(1.0, hurst_values[i]))
    
    return hurst_values


@njit(fastmath=True, cache=True)
def calculate_midline_and_trend_signal_numba(
    x_hurst: np.ndarray,
    midline_period: int
) -> tuple:
    """
    X_Hurstのミッドラインとトレンド信号を計算する（Numba最適化版）
    X_Choppinessと同様の方式：100期間の最高値と最安値の中点を使用
    
    Args:
        x_hurst: X_Hurst値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        (midline, trend_signal) のタプル
    """
    length = len(x_hurst)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, 0.0, dtype=np.float64)
    
    # ミッドライン計算（期間内の最高値と最安値の中点）
    for i in range(midline_period - 1, length):
        # 期間内のデータを取得
        period_data = x_hurst[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:  # 有効データが半分以上ある場合
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
    
    # トレンド信号計算
    for i in range(length):
        if not np.isnan(x_hurst[i]) and not np.isnan(midline[i]):
            if x_hurst[i] > midline[i]:
                # ハースト指数がミッドラインより高い = 持続性トレンド
                if x_hurst[i] > 0.55:  # 強い持続性閾値
                    trend_signal[i] = 1.0
                else:
                    trend_signal[i] = 0.5  # 弱い持続性
            else:
                # ハースト指数がミッドラインより低い = 反持続性レンジ
                if x_hurst[i] < 0.45:  # 強い反持続性閾値
                    trend_signal[i] = -1.0
                else:
                    trend_signal[i] = -0.5  # 弱い反持続性
    
    return midline, trend_signal


class XHurst(Indicator, PercentileAnalysisMixin):
    """
    X_Hurst（Xハースト）インジケーター
    
    DFA (Detrended Fluctuation Analysis) によるハースト指数を基にした
    トレンド持続性検出インジケーター。
    
    特徴:
    - DFAハースト指数の計算
    - カルマンフィルター統合（オプション）
    - 動的期間適応（オプション）
    - スムーサー統合（オプション）
    - 100期間ミッドラインによるトレンド判定
    
    ハースト指数の解釈:
    - H > 0.5: 持続性（トレンド継続傾向）
    - H = 0.5: ランダムウォーク（記憶なし）
    - H < 0.5: 反持続性（平均回帰傾向）
    """
    
    def __init__(self,
                 period: int = 55,
                 midline_period: int = 100,
                 # DFAハーストパラメータ
                 hurst_src_type: str = 'hlc3',
                 min_scale: int = 4,
                 max_scale: int = 20,
                 scale_steps: int = 8,
                 # 平滑化オプション
                 use_smoothing: bool = True,
                 smoother_type: str = 'super_smoother',
                 smoother_period: int = 8,
                 smoother_src_type: str = 'close',
                 # エラーズ統合サイクル検出器パラメータ
                 use_dynamic_period: bool = True,
                 detector_type: str = 'phac_e',
                 lp_period: int = 13,
                 hp_period: int = 124,
                 cycle_part: float = 0.5,
                 max_cycle: int = 124,
                 min_cycle: int = 13,
                 max_output: int = 124,
                 min_output: int = 13,
                 # 統合カルマンフィルターパラメータ
                 use_kalman_filter: bool = False,
                 kalman_filter_type: str = 'unscented',
                 kalman_process_noise: float = 0.01,
                 kalman_observation_noise: float = 0.001,
                 # パーセンタイルベーストレンド分析パラメータ
                 enable_percentile_analysis: bool = True,
                 percentile_lookback_period: int = 50,
                 percentile_low_threshold: float = 0.25,
                 percentile_high_threshold: float = 0.75):
        """
        X_Hurstインジケーターを初期化する
        
        Args:
            period: X_Hurst計算期間
            midline_period: ミッドライン計算期間
            hurst_src_type: ハースト計算用ソースタイプ
            min_scale: DFA最小スケール
            max_scale: DFA最大スケール
            scale_steps: DFAスケールステップ数
            use_smoothing: 平滑化を使用するか
            smoother_type: 統合スムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサーソースタイプ
            use_dynamic_period: 動的期間適応を使用するか
            detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_observation_noise: カルマンフィルター観測ノイズ
            enable_percentile_analysis: パーセンタイル分析を有効にするか
            percentile_lookback_period: パーセンタイル分析のルックバック期間
            percentile_low_threshold: パーセンタイル分析の低闾値
            percentile_high_threshold: パーセンタイル分析の高闾値
        """
        super().__init__("XHurst")
        
        # 基本パラメータ
        self.period = max(10, period)
        self.midline_period = max(10, midline_period)
        self.hurst_src_type = hurst_src_type
        
        # DFAパラメータ
        self.min_scale = max(2, min_scale)
        self.max_scale = max(self.min_scale + 1, max_scale)
        self.scale_steps = max(3, scale_steps)
        
        # DFAスケール配列を生成
        self.scales = np.logspace(
            np.log10(self.min_scale), 
            np.log10(self.max_scale), 
            self.scale_steps
        )
        
        # 平滑化設定
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = max(2, smoother_period)
        self.smoother_src_type = smoother_src_type
        
        # 動的期間適応設定
        self.use_dynamic_period = use_dynamic_period and EHLERS_UNIFIED_DC_AVAILABLE
        if self.use_dynamic_period:
            self.detector_type = detector_type
            self.lp_period = max(2, lp_period)
            self.hp_period = max(self.lp_period + 1, hp_period)
            self.cycle_part = max(0.1, min(1.0, cycle_part))
            self.max_cycle = max(10, max_cycle)
            self.min_cycle = max(2, min_cycle)
            self.max_output = max(5, max_output)
            self.min_output = max(2, min_output)
        
        # カルマンフィルター設定
        self.use_kalman_filter = use_kalman_filter and UNIFIED_KALMAN_AVAILABLE
        if self.use_kalman_filter:
            self.kalman_filter_type = kalman_filter_type
            self.kalman_process_noise = max(0.0001, kalman_process_noise)
            self.kalman_observation_noise = max(0.0001, kalman_observation_noise)
        
        # パーセンタイルベーストレンド分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # 計算結果のキャッシュ
        self._cache = {}
        self._cache_key = None
    
    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """データのキャッシュキーを生成する"""
        data_hash = hash((
            data.index.min().isoformat() if not data.empty else "",
            data.index.max().isoformat() if not data.empty else "",
            len(data),
            self.period,
            self.midline_period,
            self.hurst_src_type,
            self.min_scale,
            self.max_scale,
            self.scale_steps,
            self.use_smoothing,
            self.smoother_type if self.use_smoothing else "",
            self.smoother_period if self.use_smoothing else 0,
            self.use_dynamic_period,
            self.detector_type if self.use_dynamic_period else "",
            self.use_kalman_filter,
            self.kalman_filter_type if self.use_kalman_filter else "",
            self.enable_percentile_analysis,
            self.percentile_lookback_period if self.enable_percentile_analysis else 0,
            self.percentile_low_threshold if self.enable_percentile_analysis else 0,
            self.percentile_high_threshold if self.enable_percentile_analysis else 0
        ))
        return str(data_hash)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XHurstResult:
        """
        X_Hurstを計算する
        
        Args:
            data: OHLCV データ (DataFrame) または価格配列 (ndarray)
            
        Returns:
            XHurstResult: 計算結果
        """
        try:
            # データの前処理
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("空のDataFrameが渡されました")
                
                # キャッシュチェック
                cache_key = self._get_cache_key(data)
                if cache_key == self._cache_key and cache_key in self._cache:
                    self.logger.info("キャッシュからX_Hurst結果を返します")
                    cached_result = self._cache[cache_key]
                    return XHurstResult(
                        values=cached_result.values,
                        raw_hurst=cached_result.raw_hurst,
                        filtered_hurst=cached_result.filtered_hurst,
                        smoothed_hurst=cached_result.smoothed_hurst,
                        midline=cached_result.midline,
                        trend_signal=cached_result.trend_signal,
                        percentiles=cached_result.percentiles if hasattr(cached_result, 'percentiles') else None,
                        trend_state=cached_result.trend_state if hasattr(cached_result, 'trend_state') else None,
                        trend_intensity=cached_result.trend_intensity if hasattr(cached_result, 'trend_intensity') else None
                    )
                
                # OHLCVデータの検証
                required_columns = ['open', 'high', 'low', 'close']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"必要な列が不足しています: {missing_columns}")
                
                # 価格ソースの計算
                source_prices = PriceSource.calculate_source(data, self.hurst_src_type)
                length = len(data)
                
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    raise ValueError("空のndarrayが渡されました")
                
                source_prices = data.flatten()
                length = len(source_prices)
                
            else:
                raise ValueError("サポートされていないデータ型です")
            
            # 最小データ数チェック
            min_required = max(self.period, self.midline_period) + 20
            if length < min_required:
                self.logger.warning(f"データ数が不足しています。必要: {min_required}, 実際: {length}")
                return XHurstResult(
                    values=np.full(length, np.nan),
                    raw_hurst=np.full(length, np.nan),
                    filtered_hurst=np.full(length, np.nan),
                    smoothed_hurst=np.full(length, np.nan),
                    midline=np.full(length, np.nan),
                    trend_signal=np.full(length, 0.0),
                    percentiles=None,
                    trend_state=None,
                    trend_intensity=None
                )
            
            # 1. カルマンフィルターの適用（オプション）
            filtered_prices = source_prices.copy()
            if self.use_kalman_filter:
                try:
                    kalman = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        process_noise=self.kalman_process_noise,
                        observation_noise=self.kalman_observation_noise
                    )
                    
                    if isinstance(data, pd.DataFrame):
                        kalman_result = kalman.calculate(data)
                        if hasattr(kalman_result, 'filtered_values') and kalman_result.filtered_values is not None:
                            filtered_prices = kalman_result.filtered_values
                    else:
                        # ndarrayの場合は直接フィルター適用
                        df_temp = pd.DataFrame({
                            'open': source_prices,
                            'high': source_prices,
                            'low': source_prices,
                            'close': source_prices,
                            'volume': np.ones_like(source_prices)
                        })
                        kalman_result = kalman.calculate(df_temp)
                        if hasattr(kalman_result, 'filtered_values') and kalman_result.filtered_values is not None:
                            filtered_prices = kalman_result.filtered_values
                    
                    self.logger.info("カルマンフィルターを適用しました")
                except Exception as e:
                    self.logger.warning(f"カルマンフィルターの適用に失敗: {e}")
            
            # 2. 動的期間の計算（オプション）
            dynamic_periods = None
            if self.use_dynamic_period:
                try:
                    ehlers_dc = EhlersUnifiedDC(
                        detector_type=self.detector_type,
                        lp_period=self.lp_period,
                        hp_period=self.hp_period,
                        cycle_part=self.cycle_part,
                        max_cycle=self.max_cycle,
                        min_cycle=self.min_cycle,
                        max_output=self.max_output,
                        min_output=self.min_output
                    )
                    
                    if isinstance(data, pd.DataFrame):
                        dc_result = ehlers_dc.calculate(data)
                        dynamic_periods = dc_result.cycle_values
                    else:
                        # ndarrayの場合
                        df_temp = pd.DataFrame({
                            'open': filtered_prices,
                            'high': filtered_prices,
                            'low': filtered_prices,
                            'close': filtered_prices,
                            'volume': np.ones_like(filtered_prices)
                        })
                        dc_result = ehlers_dc.calculate(df_temp)
                        dynamic_periods = dc_result.cycle_values
                    
                    self.logger.info("動的期間適応を適用しました")
                except Exception as e:
                    self.logger.warning(f"動的期間適応の計算に失敗: {e}")
            
            # 3. DFAハースト指数の計算
            self.logger.info("DFAハースト指数を計算中...")
            raw_hurst = calculate_dfa_hurst_numba(
                filtered_prices,
                self.period,
                self.scales
            )
            
            # 4. X_Hurst値の計算（正規化）
            x_hurst_values = raw_hurst.copy()
            
            # 5. ミッドラインとトレンド信号の計算
            midline, trend_signal = calculate_midline_and_trend_signal_numba(
                x_hurst_values,
                self.midline_period
            )
            
            # 6. 平滑化の適用（オプション）
            smoothed_hurst = np.full(length, np.nan)
            if self.use_smoothing:
                try:
                    smoother = UnifiedSmoother(
                        smoother_type=self.smoother_type,
                        period=self.smoother_period,
                        src_type=self.smoother_src_type
                    )
                    
                    if isinstance(data, pd.DataFrame):
                        # X_Hurst値をDataFrameに追加して平滑化
                        temp_df = data.copy()
                        temp_df['x_hurst'] = x_hurst_values
                        smoother_result = smoother.calculate(temp_df)
                        smoothed_hurst = smoother_result.smoothed_values
                    else:
                        # ndarrayの場合
                        df_temp = pd.DataFrame({
                            'open': x_hurst_values,
                            'high': x_hurst_values,
                            'low': x_hurst_values,
                            'close': x_hurst_values,
                            'volume': np.ones_like(x_hurst_values),
                            'x_hurst': x_hurst_values
                        })
                        smoother_result = smoother.calculate(df_temp)
                        smoothed_hurst = smoother_result.smoothed_values
                    
                    self.logger.info("平滑化を適用しました")
                except Exception as e:
                    self.logger.warning(f"平滑化の適用に失敗: {e}")
            
            # 7. パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                x_hurst_values, 'trend'
            )
            
            # 結果の作成
            result = XHurstResult(
                values=x_hurst_values,
                raw_hurst=raw_hurst,
                filtered_hurst=filtered_prices,
                smoothed_hurst=smoothed_hurst,
                midline=midline,
                trend_signal=trend_signal,
                percentiles=percentiles.copy() if percentiles is not None else None,
                trend_state=trend_state.copy() if trend_state is not None else None,
                trend_intensity=trend_intensity.copy() if trend_intensity is not None else None
            )
            
            # キャッシュに保存
            if isinstance(data, pd.DataFrame):
                self._cache_key = cache_key
                self._cache[cache_key] = result
            
            # 統計情報をログ出力
            valid_count = (~np.isnan(x_hurst_values)).sum()
            persistent_count = (x_hurst_values > 0.5).sum()
            anti_persistent_count = (x_hurst_values < 0.5).sum()
            
            self.logger.info(f"X_Hurst計算完了 - 有効値: {valid_count}/{length}")
            self.logger.info(f"持続性: {persistent_count}, 反持続性: {anti_persistent_count}")
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_Hurst計算エラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return XHurstResult(
                values=empty_array,
                raw_hurst=empty_array,
                filtered_hurst=empty_array,
                smoothed_hurst=empty_array,
                midline=empty_array,
                trend_signal=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        if not self._cache:
            return None
        
        result = next(iter(self._cache.values()))
        return result.percentiles.copy() if result.percentiles is not None else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        if not self._cache:
            return None
        
        result = next(iter(self._cache.values()))
        return result.trend_state.copy() if result.trend_state is not None else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        if not self._cache:
            return None
        
        result = next(iter(self._cache.values()))
        return result.trend_intensity.copy() if result.trend_intensity is not None else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': 'X_Hurst',
            'period': self.period,
            'midline_period': self.midline_period,
            'hurst_src_type': self.hurst_src_type,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'scale_steps': self.scale_steps,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'percentile_lookback_period': self.percentile_lookback_period if self.enable_percentile_analysis else None,
            'percentile_low_threshold': self.percentile_low_threshold if self.enable_percentile_analysis else None,
            'percentile_high_threshold': self.percentile_high_threshold if self.enable_percentile_analysis else None,
            'description': 'DFAハースト指数ベースのトレンド持続性検出インジケーター（0-1範囲、高値=持続性トレンド、カルマンフィルター・動的期間・パーセンタイル分析対応）'
        }


# 便利関数
def calculate_x_hurst(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 55,
    midline_period: int = 100,
    hurst_src_type: str = 'hlc3',
    min_scale: int = 4,
    max_scale: int = 20,
    scale_steps: int = 8,
    use_smoothing: bool = True,
    smoother_type: str = 'super_smoother',
    use_dynamic_period: bool = True,
    use_kalman_filter: bool = False,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    X_Hurstの計算（便利関数）
    
    Args:
        data: 価格データ
        period: X_Hurst計算期間
        midline_period: ミッドライン計算期間
        hurst_src_type: ハースト計算用ソースタイプ
        min_scale: DFA最小スケール
        max_scale: DFA最大スケール
        scale_steps: DFAスケールステップ数
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_dynamic_period: 動的期間適応を使用するか
        use_kalman_filter: カルマンフィルターを使用するか
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル分析のルックバック期間
        percentile_low_threshold: パーセンタイル分析の低閾値
        percentile_high_threshold: パーセンタイル分析の高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        X_Hurst値
    """
    indicator = XHurst(
        period=period,
        midline_period=midline_period,
        hurst_src_type=hurst_src_type,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_steps=scale_steps,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        use_dynamic_period=use_dynamic_period,
        use_kalman_filter=use_kalman_filter,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


# テスト用のコード
if __name__ == "__main__":
    # 簡単なテスト
    import matplotlib.pyplot as plt
    
    # テストデータ生成
    np.random.seed(42)
    n = 500
    trend_data = np.cumsum(np.random.randn(n) * 0.02) + 100
    noise = np.random.randn(n) * 0.5
    test_prices = trend_data + noise
    
    # DataFrameを作成
    test_df = pd.DataFrame({
        'open': test_prices,
        'high': test_prices + np.abs(np.random.randn(n) * 0.5),
        'low': test_prices - np.abs(np.random.randn(n) * 0.5),
        'close': test_prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # X_Hurstを計算
    x_hurst = XHurst(
        period=50,
        midline_period=100,
        use_smoothing=True,
        use_dynamic_period=False,  # テスト時は無効
        use_kalman_filter=False   # テスト時は無効
    )
    
    result = x_hurst.calculate(test_df)
    
    print(f"X_Hurst計算完了")
    print(f"有効データ数: {(~np.isnan(result.values)).sum()}")
    print(f"平均ハースト値: {np.nanmean(result.values):.4f}")
    print(f"持続性レジーム割合: {(result.values > 0.5).sum() / len(result.values) * 100:.1f}%")
    
    # 簡単なプロット
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(test_prices, label='Price', alpha=0.7)
    plt.title('Test Price Data')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(result.values, label='X-Hurst', color='blue')
    plt.plot(result.midline, label='Midline', color='black', linestyle='--')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.title('X-Hurst Exponent')
    plt.ylabel('Hurst Value')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(result.trend_signal, label='Trend Signal', color='orange')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.title('Trend Signal')
    plt.ylabel('Signal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()