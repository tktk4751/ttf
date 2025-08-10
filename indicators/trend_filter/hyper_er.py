#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64

from ..indicator import Indicator
from ..price_source import PriceSource
from ..smoother.unified_smoother import UnifiedSmoother
from ..smoother.roofing_filter import RoofingFilter
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
class HyperERResult:
    """Hyper_ERの計算結果"""
    values: np.ndarray               # Hyper_ER値（0-1の範囲、高い値=効率的トレンド）
    raw_er: np.ndarray              # 生のER値（ルーフィングフィルター適用前）
    filtered_er: np.ndarray         # ルーフィングフィルター適用後のER値
    smoothed_er: np.ndarray         # 平滑化されたER値（オプション）
    midline: np.ndarray             # ミッドライン値
    trend_signal: np.ndarray        # トレンド判定信号（1=トレンド、-1=レンジ）
    roofing_values: np.ndarray      # ルーフィングフィルター値
    filtered_prices: np.ndarray     # カルマンフィルタリング後の価格（オプション）
    cycle_periods: np.ndarray       # サイクル期間値（動的期間使用時）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）




@njit(fastmath=True, cache=True)
def calculate_efficiency_ratio_numba(
    prices: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Efficiency Ratioを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        period: 計算期間
        
    Returns:
        Efficiency Ratio値の配列（0-1の範囲）
    """
    length = len(prices)
    er_values = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(period, length):
        # 期間内での価格変化
        price_change = abs(prices[i] - prices[i - period])
        
        # 期間内での価格変動の合計（ボラティリティ）
        volatility = 0.0
        for j in range(i - period, i):
            volatility += abs(prices[j + 1] - prices[j])
        
        # Efficiency Ratioの計算
        if volatility > 1e-10:  # ゼロ除算防止
            er_values[i] = price_change / volatility
        else:
            er_values[i] = 0.0
    
    # 0-1の範囲にクリップ
    for i in range(length):
        if not np.isnan(er_values[i]):
            er_values[i] = max(0.0, min(1.0, er_values[i]))
    
    return er_values


@njit(fastmath=True, cache=True)
def calculate_hyper_er_numba(
    er_values: np.ndarray,
    period: int,
    dynamic_periods: np.ndarray = None
) -> np.ndarray:
    """
    Hyper_ERを計算する（Numba最適化版）
    
    Args:
        er_values: Efficiency Ratio値の配列
        period: 基本計算期間
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        Hyper_ER値の配列（0-1の範囲）
    """
    length = len(er_values)
    hyper_er = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(period - 1, length):
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
        
        # 現在のインデックスが期間に対して十分かチェック
        if i < current_period - 1:
            continue
            
        # 期間内のER値の平均を計算
        er_sum = 0.0
        valid_count = 0
        
        for j in range(i - current_period + 1, i + 1):
            if not np.isnan(er_values[j]):
                er_sum += er_values[j]
                valid_count += 1
        
        if valid_count >= current_period // 2:  # 有効なデータが半分以上の場合
            avg_er = er_sum / valid_count
            
            # Hyper_ERは平均化されたER値そのものを使用（0-1の範囲は既に保証されている）
            hyper_er[i] = avg_er
    
    return hyper_er


@njit(fastmath=True, cache=True)
def calculate_midline_and_signal_er(
    hyper_er: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとトレンド信号を計算する（Numba最適化版）
    
    Args:
        hyper_er: Hyper_ER値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, トレンド信号)
    """
    length = len(hyper_er)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_data = hyper_er[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # トレンド信号の判定
            if not np.isnan(hyper_er[i]):
                if hyper_er[i] > midline[i]:
                    trend_signal[i] = 1.0   # 効率的トレンド相場
                else:
                    trend_signal[i] = -1.0  # 非効率的（レンジ）相場
    
    return midline, trend_signal


class HyperER(Indicator, PercentileAnalysisMixin):
    """
    Hyper_ER（Hyper-Efficiency Ratio）インジケーター
    
    Efficiency Ratioをベースとした改良版トレンド効率性指標。
    ラゲールフィルターは廃止され、代わりに統合カルマンフィルターが導入されました。
    
    特徴:
    - 0-1の値範囲で効率性を表現
    - 高い値=効率的なトレンド相場、低い値=非効率的なレンジ相場
    - 100期間ミッドラインによるトレンド判定機能
    - 統合カルマンフィルター、統合サイクル検出器、統合スムーサー対応
    - 動的期間適応機能
    
    計算手順 (Hyper Trend Indexと同じフロー):
    1. ソース価格→統合カルマンフィルターによるフィルター処理(オプション)
    2. サイクル検出器による期間検出
    3. ルーフィングフィルターによるフィルター処理(オプション)
    4. フィルター処理済価格とサイクル期間によるER計算
    5. 平滑化フィルターによる処理(オプション)
    6. ミッドライン計算とトレンド判定
    
    注意: ラゲールフィルター関連パラメーターは後方互換性のために保持されていますが、機能的には無効です。
    """
    
    def __init__(
        self,
        period: int = 8,
        midline_period: int = 100,
        # ERパラメータ
        er_period: int = 13,
        er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        use_kalman_filter: bool = True,
        kalman_filter_type: str = 'unscented',
        kalman_process_noise: float = 1e-5,
        kalman_min_observation_noise: float = 1e-6,
        kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        use_roofing_filter: bool = True,
        roofing_hp_cutoff: float = 55.0,
        roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        use_laguerre_filter: bool = False,
        laguerre_gamma: float = 0.5,
        # 平滑化オプション
        use_smoothing: bool = True,
        smoother_type: str = 'laguerre',
        smoother_period: int = 12,
        smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = True,
        detector_type: str = 'dft_dominant',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.4,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 89,
        min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        enable_percentile_analysis: bool = False,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ
        
        Args:
            period: Hyper_ER計算期間（デフォルト: 14）
            midline_period: ミッドライン計算期間（デフォルト: 100）
            er_period: ER期間（デフォルト: 13）
            er_src_type: ERソースタイプ（デフォルト: 'oc2'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: True）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'adaptive'）
            kalman_process_noise: カルマンフィルターのプロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: 適応ウィンドウ（デフォルト: 5）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: True）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            use_laguerre_filter: ラゲールフィルターを使用するか（廃止済み、後方互換性のためのみ、常にFalse）
            laguerre_gamma: ラゲールガンマ（廃止済み、後方互換性のためのみ）
            use_smoothing: 平滑化を使用するか（デフォルト: True）
            smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 12）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: True）
            detector_type: サイクル検出器タイプ（デフォルト: 'dft_dominant'）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            cycle_part: サイクル部分（デフォルト: 0.4）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 89）
            min_output: 最小出力値（デフォルト: 5）
        """
        indicator_name = f"Hyper_ER({period}, midline={midline_period}, ER={er_period}({er_src_type})"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_dynamic_period:
            indicator_name += f", dynamic={detector_type}"
        if use_roofing_filter:
            indicator_name += f", roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})"
        if use_smoothing:
            indicator_name += f", smooth={smoother_type}({smoother_period})"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.period = period
        self.midline_period = midline_period
        self.er_period = er_period
        self.er_src_type = er_src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_min_observation_noise = kalman_min_observation_noise
        self.kalman_adaptation_window = kalman_adaptation_window
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        # 後方互換性のため残すが使用しない
        self.use_laguerre_filter = False  # 常にFalse
        self.laguerre_gamma = laguerre_gamma
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        
        # エラーズ統合サイクル検出器パラメータ
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # パーセンタイルベーストレンド分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.midline_period <= 0:
            raise ValueError("midline_periodは0より大きい必要があります")
        if self.er_period <= 0:
            raise ValueError("er_periodは0より大きい必要があります")
        if self.use_dynamic_period and self.max_cycle <= self.min_cycle:
            raise ValueError("max_cycleはmin_cycleより大きい必要があります")
        if self.use_roofing_filter and self.roofing_ss_band_edge >= self.roofing_hp_cutoff:
            raise ValueError("roofing_ss_band_edgeはroofing_hp_cutoffより小さい必要があります")
        
        # 統合カルマンフィルターの初期化（価格フィルタリングが有効な場合）
        self.kalman_filter = None
        if self.use_kalman_filter:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.use_kalman_filter = False
                self.logger.warning("カルマンフィルター機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.er_src_type,
                        process_noise=self.kalman_process_noise,
                        min_observation_noise=self.kalman_min_observation_noise,
                        adaptation_window=self.kalman_adaptation_window
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
        # エラーズ統合サイクル検出器の初期化（動的期間適応が有効な場合）
        self.cycle_detector = None
        
        if self.use_dynamic_period:
            if not EHLERS_UNIFIED_DC_AVAILABLE:
                self.logger.error("エラーズ統合サイクル検出器が利用できません。indicators.cycle.ehlers_unified_dcをインポートできません。")
                self.use_dynamic_period = False
                self.logger.warning("動的期間適応機能を無効にしました")
            else:
                try:
                    self.cycle_detector = EhlersUnifiedDC(
                        detector_type=self.detector_type,
                        cycle_part=self.cycle_part,
                        max_cycle=self.max_cycle,
                        min_cycle=self.min_cycle,
                        max_output=self.max_output,
                        min_output=self.min_output,
                        src_type='hlc3',
                        use_kalman_filter=False,
                        lp_period=self.lp_period,
                        hp_period=self.hp_period
                    )
                    self.logger.info(f"エラーズ統合サイクル検出器を初期化しました: {self.detector_type}")
                except Exception as e:
                    self.logger.error(f"エラーズ統合サイクル検出器の初期化に失敗: {e}")
                    self.use_dynamic_period = False
                    self.logger.warning("動的期間適応機能を無効にしました")
        
        # ルーフィングフィルターの初期化（ルーフィングフィルターが有効な場合）
        self.roofing_filter = None
        if self.use_roofing_filter:
            try:
                self.roofing_filter = RoofingFilter(
                    src_type=self.er_src_type,
                    hp_cutoff=self.roofing_hp_cutoff,
                    ss_band_edge=self.roofing_ss_band_edge
                )
                self.logger.info(f"ルーフィングフィルターを初期化しました: hp={self.roofing_hp_cutoff}, ss={self.roofing_ss_band_edge}")
            except Exception as e:
                self.logger.error(f"ルーフィングフィルターの初期化に失敗: {e}")
                self.use_roofing_filter = False
                self.logger.warning("ルーフィングフィルター機能を無効にしました")
        
        # 統合スムーサーの初期化（オプション）
        self.smoother = None
        if self.use_smoothing:
            try:
                self.smoother = UnifiedSmoother(
                    smoother_type=self.smoother_type,
                    src_type=self.smoother_src_type,
                    period=self.smoother_period
                )
            except Exception as e:
                self.logger.error(f"統合スムーサーの初期化に失敗: {e}")
                self.use_smoothing = False
                self.logger.warning("平滑化機能を無効にしました")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
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
            param_str = (f"{self.period}_{self.midline_period}_{self.er_period}_"
                        f"{self.er_src_type}_{self.use_roofing_filter}_{self.roofing_hp_cutoff}_"
                        f"{self.roofing_ss_band_edge}_{self.use_smoothing}_{self.smoother_type}_"
                        f"{self.smoother_period}_{self.smoother_src_type}")
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.midline_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperERResult:
        """
        Hyper_ERを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close, open（ルーフィングフィルター用）
        
        Returns:
            HyperERResult: Hyper_ERの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return HyperERResult(
                    values=cached_result.values.copy(),
                    raw_er=cached_result.raw_er.copy(),
                    filtered_er=cached_result.filtered_er.copy(),
                    smoothed_er=cached_result.smoothed_er.copy(),
                    midline=cached_result.midline.copy(),
                    trend_signal=cached_result.trend_signal.copy(),
                    roofing_values=cached_result.roofing_values.copy(),
                    filtered_prices=cached_result.filtered_prices.copy(),
                    cycle_periods=cached_result.cycle_periods.copy(),
                    percentiles=cached_result.percentiles.copy() if cached_result.percentiles is not None else None,
                    trend_state=cached_result.trend_state.copy() if cached_result.trend_state is not None else None,
                    trend_intensity=cached_result.trend_intensity.copy() if cached_result.trend_intensity is not None else None
                )
            
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                if self.use_roofing_filter:
                    required_cols.extend(['open'])  # ルーフィングフィルター用
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < max(self.period, self.midline_period, self.er_period):
                self.logger.warning(f"データ長（{data_length}）が必要な期間（{max(self.period, self.midline_period, self.er_period)}）より短いです")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.er_src_type)
            
            # NumPy配列に変換
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. 動的期間の計算（オプション）
            dynamic_periods = None
            cycle_periods = np.full(data_length, self.period, dtype=np.float64)
            
            if self.use_dynamic_period and self.cycle_detector is not None:
                try:
                    cycle_values = self.cycle_detector.calculate(data)
                    
                    # サイクル値から期間を計算
                    valid_cycles = cycle_values[~np.isnan(cycle_values) & (cycle_values > 0)]
                    if len(valid_cycles) > 0:
                        # サイクル値を期間に変換（例：サイクル値の半分を期間とする）
                        dynamic_periods = np.where(
                            ~np.isnan(cycle_values) & (cycle_values > 0),
                            np.clip(cycle_values * 0.5, 5, 50),
                            self.period
                        )
                        cycle_periods = cycle_values.copy()
                    else:
                        dynamic_periods = np.full(len(data), self.period)
                        
                    self.logger.debug(f"動的期間を計算しました。範囲: {np.min(dynamic_periods):.1f} - {np.max(dynamic_periods):.1f}")
                        
                except Exception as e:
                    self.logger.warning(f"動的期間計算中にエラー: {e}。固定期間を使用します。")
                    dynamic_periods = None
            
            # 3. ルーフィングフィルターによる価格データのフィルタリング（オプション）
            filtered_prices = source_prices
            roofing_values = np.full_like(source_prices, np.nan)
            # ラゲールフィルター値は後方互換性のため変数名を保持（実際はfiltered_pricesを使用）
            # laguerre_values = np.full_like(source_prices, np.nan)  # 廃止
            
            if self.use_roofing_filter and self.roofing_filter is not None:
                try:
                    roofing_result = self.roofing_filter.calculate(data)
                    roofing_values = roofing_result.values
                    
                    # ルーフィングフィルターの結果を使用
                    # ルーフィングフィルターは価格の振動成分を抽出するので、直接フィルタリング済み価格として使用
                    # NaN値が多い場合は元の価格を使用
                    valid_roofing = np.sum(~np.isnan(roofing_values))
                    if valid_roofing > len(roofing_values) * 0.5:  # 有効値が50%以上の場合
                        # ルーフィングフィルター値をソース価格のスケールに調整
                        roofing_range = np.nanmax(roofing_values) - np.nanmin(roofing_values)
                        price_range = np.nanmax(source_prices) - np.nanmin(source_prices)
                        if roofing_range > 0 and price_range > 0:
                            scale_factor = price_range / roofing_range * 0.1  # 10%の影響度
                            filtered_prices = source_prices + roofing_values * scale_factor
                        else:
                            filtered_prices = source_prices
                    else:
                        filtered_prices = source_prices
                    
                    # NumPy配列として確保
                    if not isinstance(filtered_prices, np.ndarray):
                        filtered_prices = np.array(filtered_prices)
                    if filtered_prices.dtype != np.float64:
                        filtered_prices = filtered_prices.astype(np.float64)
                    
                    self.logger.debug("ルーフィングフィルターによる価格データのフィルタリングを適用しました")
                        
                except Exception as e:
                    self.logger.warning(f"ルーフィングフィルター適用中にエラー: {e}。元の値を使用します。")
                    filtered_prices = source_prices
                    roofing_values = np.full_like(source_prices, np.nan)
            
            # 4. ラゲールフィルター（後方互換性のために残されているが、現在は無効）
            # 注意: ラゲールフィルター機能は廃止されました。パラメーターは後方互換性のために保持されています。
            
            # NumPy配列として確保
            if not isinstance(filtered_prices, np.ndarray):
                filtered_prices = np.array(filtered_prices)
            if filtered_prices.dtype != np.float64:
                filtered_prices = filtered_prices.astype(np.float64)
            
            # 5. Efficiency Ratioを計算
            raw_er = calculate_efficiency_ratio_numba(filtered_prices, self.er_period)
            
            # 6. Hyper_ERを計算
            hyper_er_values = calculate_hyper_er_numba(
                raw_er, self.period, dynamic_periods
            )
            
            # 7. 平滑化（オプション）
            smoothed_er = hyper_er_values.copy()  # デフォルトで生の値を使用
            if self.use_smoothing and self.smoother is not None:
                try:
                    # NaN値の処理 - スムーサーが正常に動作するようにNaN値を前方補完
                    clean_hyper_er = hyper_er_values.copy()
                    nan_mask = np.isnan(clean_hyper_er)
                    
                    if np.any(nan_mask):
                        # 最初の有効値を見つけて前方補完
                        first_valid_idx = np.where(~nan_mask)[0]
                        if len(first_valid_idx) > 0:
                            first_valid = first_valid_idx[0]
                            first_value = clean_hyper_er[first_valid]
                            # 最初の有効値より前をその値で補完
                            clean_hyper_er[:first_valid] = first_value
                            
                            # 残りのNaN値は前方補完
                            for i in range(len(clean_hyper_er)):
                                if np.isnan(clean_hyper_er[i]) and i > 0:
                                    clean_hyper_er[i] = clean_hyper_er[i-1]
                    
                    # Hyper_ER値をDataFrame形式に変換
                    # 一部のスムーサー（FRAMA等）で必要なカラムを含める
                    if isinstance(data, pd.DataFrame):
                        # 元のデータがDataFrameの場合、必要なカラムを保持しつつHyper_ER値を使用
                        er_df = data.copy()
                        er_df['close'] = clean_hyper_er
                        # インデックスの長さを調整
                        if len(er_df) != len(clean_hyper_er):
                            er_df = er_df.iloc[:len(clean_hyper_er)].copy()
                            er_df['close'] = clean_hyper_er
                    else:
                        # NumPy配列の場合は基本的なDataFrameを作成
                        er_df = pd.DataFrame({'close': clean_hyper_er})
                        # high, lowカラムもcloseと同じ値で作成（FRAMAなど用）
                        er_df['high'] = clean_hyper_er
                        er_df['low'] = clean_hyper_er
                        er_df['open'] = clean_hyper_er
                    
                    # 平滑化を適用
                    smoother_result = self.smoother.calculate(er_df)
                    if smoother_result is not None and hasattr(smoother_result, 'values'):
                        smooth_values = smoother_result.values
                        # 有効な平滑化結果がある場合のみ使用
                        if np.sum(~np.isnan(smooth_values)) > 0:
                            # 元のNaN位置を復元
                            if np.any(nan_mask):
                                smooth_values[nan_mask] = np.nan
                            smoothed_er = smooth_values
                            self.logger.debug(f"平滑化処理完了: 有効値数 {np.sum(~np.isnan(smooth_values))}")
                        else:
                            self.logger.warning("平滑化結果がすべてNaN。生の値を使用します。")
                    else:
                        self.logger.warning("平滑化結果が無効。生の値を使用します。")
                except Exception as e:
                    self.logger.warning(f"平滑化処理中にエラー: {e}。生の値を使用します。")
            
            # 最終的なHyper_ER値（平滑化が有効で成功した場合は平滑化値、そうでなければ生の値）
            final_hyper_er = smoothed_er if self.use_smoothing else hyper_er_values
            
            # 7. ミッドラインとトレンド信号を計算
            midline, trend_signal = calculate_midline_and_signal_er(
                final_hyper_er, self.midline_period
            )
            
            # 8. パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                final_hyper_er, 'trend'
            )
            
            # 結果の作成
            result = HyperERResult(
                values=final_hyper_er.copy(),
                raw_er=raw_er.copy(),
                filtered_er=hyper_er_values.copy(),
                smoothed_er=smoothed_er.copy(),
                midline=midline.copy(),
                trend_signal=trend_signal.copy(),
                roofing_values=roofing_values.copy(),
                filtered_prices=filtered_prices.copy(),
                cycle_periods=cycle_periods.copy(),
                percentiles=percentiles.copy() if percentiles is not None else None,
                trend_state=trend_state.copy() if trend_state is not None else None,
                trend_intensity=trend_intensity.copy() if trend_intensity is not None else None
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = final_hyper_er
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Hyper_ER計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return HyperERResult(
                values=empty_array,
                raw_er=empty_array,
                filtered_er=empty_array,
                smoothed_er=empty_array,
                midline=empty_array,
                trend_signal=empty_array,
                roofing_values=empty_array,
                filtered_prices=empty_array,
                cycle_periods=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """Hyper_ER値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_raw_er(self) -> Optional[np.ndarray]:
        """生のER値を取得"""
        result = self._get_latest_result()
        return result.raw_er.copy() if result else None
    
    def get_filtered_er(self) -> Optional[np.ndarray]:
        """フィルタリングされたER値を取得"""
        result = self._get_latest_result()
        return result.filtered_er.copy() if result else None
    
    def get_smoothed_er(self) -> Optional[np.ndarray]:
        """平滑化されたER値を取得"""
        result = self._get_latest_result()
        return result.smoothed_er.copy() if result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_trend_signal(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        result = self._get_latest_result()
        return result.trend_signal.copy() if result else None
    
    def get_roofing_values(self) -> Optional[np.ndarray]:
        """ルーフィングフィルター値を取得"""
        result = self._get_latest_result()
        return result.roofing_values.copy() if result else None
    
    def get_laguerre_values(self) -> Optional[np.ndarray]:
        """ラゲールフィルター値を取得（後方互換性のため。実際はfiltered_pricesを返す）"""
        result = self._get_latest_result()
        return result.filtered_prices.copy() if result else None
    
    def get_cycle_periods(self) -> Optional[np.ndarray]:
        """サイクル期間値を取得"""
        result = self._get_latest_result()
        return result.cycle_periods.copy() if result else None
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        result = self._get_latest_result()
        return result.percentiles.copy() if result and result.percentiles is not None else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        result = self._get_latest_result()
        return result.trend_state.copy() if result and result.trend_state is not None else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        result = self._get_latest_result()
        return result.trend_intensity.copy() if result and result.trend_intensity is not None else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'midline_period': self.midline_period,
            'er_period': self.er_period,
            'er_src_type': self.er_src_type,
            'use_roofing_filter': self.use_roofing_filter,
            'roofing_hp_cutoff': self.roofing_hp_cutoff if self.use_roofing_filter else None,
            'roofing_ss_band_edge': self.roofing_ss_band_edge if self.use_roofing_filter else None,
            'use_laguerre_filter': self.use_laguerre_filter,
            'laguerre_gamma': None,  # ラゲールフィルターは廃止（後方互換性のためパラメーターは保持）
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'percentile_lookback_period': self.percentile_lookback_period if self.enable_percentile_analysis else None,
            'percentile_low_threshold': self.percentile_low_threshold if self.enable_percentile_analysis else None,
            'percentile_high_threshold': self.percentile_high_threshold if self.enable_percentile_analysis else None,
            'description': 'Efficiency Ratioベースの改良効率性指標（0-1範囲、高値=効率的トレンド、ルーフィングフィルター・動的期間・パーセンタイル分析対応）'
        }
    
    def _get_latest_result(self) -> Optional[HyperERResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        if self.smoother:
            self.smoother.reset()
        if self.cycle_detector:
            self.cycle_detector.reset()
        if self.roofing_filter:
            self.roofing_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_hyper_er(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 14,
    midline_period: int = 100,
    er_period: int = 13,
    er_src_type: str = 'hlc3',
    use_roofing_filter: bool = True,
    roofing_hp_cutoff: float = 48.0,
    roofing_ss_band_edge: float = 10.0,
    use_smoothing: bool = True,
    smoother_type: str = 'super_smoother',
    use_dynamic_period: bool = True,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    Hyper_ERの計算（便利関数）
    
    Args:
        data: 価格データ
        period: Hyper_ER計算期間
        midline_period: ミッドライン計算期間
        er_period: ER期間
        er_src_type: ERソースタイプ
        use_roofing_filter: ルーフィングフィルターを使用するか
        roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_dynamic_period: 動的期間適応を使用するか
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル分析のルックバック期間
        percentile_low_threshold: パーセンタイル分析の低閾値
        percentile_high_threshold: パーセンタイル分析の高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        Hyper_ER値
    """
    indicator = HyperER(
        period=period,
        midline_period=midline_period,
        er_period=er_period,
        er_src_type=er_src_type,
        use_roofing_filter=use_roofing_filter,
        roofing_hp_cutoff=roofing_hp_cutoff,
        roofing_ss_band_edge=roofing_ss_band_edge,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        use_dynamic_period=use_dynamic_period,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Hyper_ER インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 効率的トレンド相場
            change = 0.003 + np.random.normal(0, 0.008)
        elif i < 100:  # 非効率的レンジ相場
            change = np.random.normal(0, 0.012)
        elif i < 150:  # 非常に効率的なトレンド相場
            change = 0.005 + np.random.normal(0, 0.006)
        else:  # レンジ相場
            change = np.random.normal(0, 0.010)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Hyper_ERを計算（基本版）
    print("\\n基本版Hyper_ERをテスト中...")
    hyper_er = HyperER(period=14, midline_period=50, use_roofing_filter=False)
    result = hyper_er.calculate(df)
    
    valid_count = np.sum(~np.isnan(result.values))
    mean_hyper_er = np.nanmean(result.values)
    trend_ratio = np.sum(result.trend_signal == 1) / np.sum(~np.isnan(result.trend_signal))
    
    print(f"  有効値数: {valid_count}/{len(df)}")
    print(f"  平均Hyper_ER: {mean_hyper_er:.4f}")
    print(f"  効率的トレンド信号比率: {trend_ratio:.2%}")
    
    # ルーフィングフィルター版をテスト
    print("\\nルーフィングフィルター版Hyper_ERをテスト中...")
    hyper_er_roofing = HyperER(
        period=14,
        midline_period=50,
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0,
        use_smoothing=True,
        smoother_type='frama'
    )
    result_roofing = hyper_er_roofing.calculate(df)
    
    valid_count_roofing = np.sum(~np.isnan(result_roofing.values))
    mean_hyper_er_roofing = np.nanmean(result_roofing.values)
    
    print(f"  有効値数: {valid_count_roofing}/{len(df)}")
    print(f"  平均Hyper_ER（ルーフィング+平滑化）: {mean_hyper_er_roofing:.4f}")
    
    # 比較統計
    if valid_count > 0 and valid_count_roofing > 0:
        correlation = np.corrcoef(
            result.values[~np.isnan(result.values)][-min(valid_count, valid_count_roofing):],
            result_roofing.values[~np.isnan(result_roofing.values)][-min(valid_count, valid_count_roofing):]
        )[0, 1]
        print(f"  基本版とルーフィング版の相関: {correlation:.4f}")
    
    print("\\n=== テスト完了 ===")