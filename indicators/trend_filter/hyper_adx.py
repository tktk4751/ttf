#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Hyper ADX - 統合型Average Directional Index** 🎯

ハイパーERと同じ計算フローを採用したADXインジケーター。
ルーフィングフィルターとカルマンフィルター統合により、ノイズ除去と精度向上を実現。

🌟 **主要機能:**
1. **統合カルマンフィルター**: 価格データのノイズ除去
2. **ルーフィングフィルター**: 追加のフィルタリング
3. **統合スムーサー**: DX値の必須平滑化（ADX作成）
4. **動的期間対応**: サイクル検出器による期間適応
5. **100期間ミッドライン**: トレンド判定ロジック

📊 **処理フロー（ハイパーERと同じ）:**
1. ソース価格→統合カルマンフィルターによるフィルター処理(オプション)
2. サイクル検出器による期間検出
3. ルーフィングフィルターによるフィルター処理(オプション)
4. フィルター処理済価格による DX 計算
5. 統合スムーサーによる DX の平滑化（ADX作成、必須機能）
6. ミッドライン計算とトレンド判定

🔧 **パラメータ:**
- 統合カルマンフィルター: ノイズ除去とフィルタリング
- ルーフィングフィルター: 追加フィルタリング
- 統合スムーサー: DX→ADXの必須平滑化
- 動的期間: サイクル検出器による適応期間
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from ..indicator import Indicator
from ..price_source import PriceSource
from ..utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from ..smoother.unified_smoother import UnifiedSmoother
    UNIFIED_SMOOTHER_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.smoother.unified_smoother import UnifiedSmoother
        UNIFIED_SMOOTHER_AVAILABLE = True
    except ImportError:
        UnifiedSmoother = None
        UNIFIED_SMOOTHER_AVAILABLE = False

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

try:
    from ..smoother.roofing_filter import RoofingFilter
    ROOFING_FILTER_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.smoother.roofing_filter import RoofingFilter
        ROOFING_FILTER_AVAILABLE = True
    except ImportError:
        RoofingFilter = None
        ROOFING_FILTER_AVAILABLE = False


@dataclass
class HyperADXResult:
    """Hyper ADXの計算結果"""
    values: np.ndarray                    # Hyper ADX値（統合スムーサーで平滑化されたADX）
    raw_dx: np.ndarray                    # 生のDX値（平滑化前）
    secondary_smoothed: Optional[np.ndarray]  # 二次平滑化されたADX値（オプション）
    midline: np.ndarray                   # ミッドライン値
    trend_signal: np.ndarray              # トレンド判定信号（1=トレンド、-1=レンジ）
    filtered_prices: np.ndarray           # カルマンフィルタリング後の価格（オプション）
    roofing_values: np.ndarray            # ルーフィングフィルター値（オプション）
    plus_di: np.ndarray                   # +DI値
    minus_di: np.ndarray                  # -DI値
    cycle_periods: np.ndarray             # サイクル期間値（動的期間使用時）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_directional_movement_hyper(
    high: np.ndarray,
    low: np.ndarray
) -> tuple:
    """
    Directional Movement（+DM, -DM）を計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length, dtype=np.float64)
    minus_dm = np.zeros(length, dtype=np.float64)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0.0
            
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0.0
    
    return plus_dm, minus_dm


@njit(fastmath=True, cache=True)
def calculate_true_range_hyper(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Rangeを計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        
    Returns:
        True Range値の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の値
    tr[0] = high[0] - low[0]
    
    # 2番目以降
    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True, cache=True)
def calculate_dx_values_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    dynamic_periods: np.ndarray = None
) -> tuple:
    """
    DX値と+DI、-DI値を計算する（Numba最適化版）
    平滑化は統合スムーサーで後から行う
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 基本計算期間
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (DX値, +DI値, -DI値)
    """
    length = len(high)
    dx_values = np.full(length, np.nan, dtype=np.float64)
    plus_di = np.full(length, np.nan, dtype=np.float64)
    minus_di = np.full(length, np.nan, dtype=np.float64)
    
    # +DM, -DM, TRの計算
    plus_dm, minus_dm = calculate_directional_movement_hyper(high, low)
    tr_values = calculate_true_range_hyper(high, low, close)
    
    # 指数移動平均による平滑化
    alpha = 2.0 / (period + 1.0)
    
    # 平滑化用の変数
    smoothed_tr = np.zeros(length, dtype=np.float64)
    smoothed_plus_dm = np.zeros(length, dtype=np.float64)
    smoothed_minus_dm = np.zeros(length, dtype=np.float64)
    
    for i in range(period - 1, length):
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
        
        current_alpha = 2.0 / (current_period + 1.0)
        
        # 最初の値の初期化
        if i == period - 1:
            smoothed_tr[i] = tr_values[i]
            smoothed_plus_dm[i] = plus_dm[i]
            smoothed_minus_dm[i] = minus_dm[i]
        else:
            # 指数移動平均による平滑化
            smoothed_tr[i] = (tr_values[i] * current_alpha) + (smoothed_tr[i-1] * (1 - current_alpha))
            smoothed_plus_dm[i] = (plus_dm[i] * current_alpha) + (smoothed_plus_dm[i-1] * (1 - current_alpha))
            smoothed_minus_dm[i] = (minus_dm[i] * current_alpha) + (smoothed_minus_dm[i-1] * (1 - current_alpha))
        
        # +DI, -DIの計算
        if smoothed_tr[i] > 0:
            plus_di[i] = smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = smoothed_minus_dm[i] / smoothed_tr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        
        # DXの計算（ADXの前段階、平滑化なし）
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx_values[i] = abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx_values[i] = 0.0
    
    return dx_values, plus_di, minus_di


@njit(fastmath=True, cache=True)
def calculate_midline_and_trend_signal_hyper(
    hyper_adx: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとトレンド信号を計算する（Numba最適化版）
    
    Args:
        hyper_adx: Hyper ADX値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, トレンド信号)
    """
    length = len(hyper_adx)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_data = hyper_adx[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # トレンド信号の判定
            if not np.isnan(hyper_adx[i]):
                if hyper_adx[i] > midline[i]:
                    trend_signal[i] = 1.0  # トレンド
                else:
                    trend_signal[i] = -1.0  # レンジ
    
    return midline, trend_signal


class HyperADX(Indicator, PercentileAnalysisMixin):
    """
    Hyper ADX（統合型Average Directional Index）インジケーター
    
    ハイパーERと同じ計算フローを採用したADXインジケーター。
    統合カルマンフィルター、ルーフィングフィルター、統合スムーサーによる高精度なトレンド検出を実現。
    
    特徴:
    - 0-1の値範囲でトレンド強度を表現
    - 高い値=強いトレンド、低い値=レンジ相場
    - 100期間ミッドラインによるトレンド判定機能
    - 統合カルマンフィルター、ルーフィングフィルター、統合スムーサー対応
    - 動的期間適応機能
    
    計算手順（ハイパーERと同じフロー）:
    1. ソース価格→統合カルマンフィルターによるフィルター処理(オプション)
    2. サイクル検出器による期間検出
    3. ルーフィングフィルターによるフィルター処理(オプション)
    4. フィルター処理済価格による DX 計算
    5. 統合スムーサーによる DX の平滑化（ADX作成、必須機能）
    6. ミッドライン計算とトレンド判定
    
    注意: 平滑化は XADX と異なり必須機能です。DX値を統合スムーサーでADXに変換します。
    """
    
    def __init__(self,
        period: int = 14,
        midline_period: int = 100,
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
        # 統合スムーサーパラメータ（必須機能）
        smoother_type: str = 'frama',
        smoother_period: int = 24,
        smoother_src_type: str = 'close',
        # 二次平滑化パラメータ（オプション機能）
        use_secondary_smoothing: bool = False,
        secondary_smoother_type: str = 'zlema',
        secondary_smoother_period: int = 8,
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = True,
        detector_type: str = 'dft_dominant',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.4,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 3,
        min_output: int = 34,
        # パーセンタイルベーストレンド分析パラメータ
        enable_percentile_analysis: bool = False,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ
        
        Args:
            period: ADX計算期間（デフォルト: 14）
            midline_period: ミッドライン計算期間（デフォルト: 100）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: True）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'adaptive'）
            kalman_process_noise: カルマンフィルターのプロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: 適応ウィンドウ（デフォルト: 5）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: True）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            smoother_type: 統合スムーサータイプ（必須機能、デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 12）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_secondary_smoothing: 二次平滑化を使用するか（デフォルト: False）
            secondary_smoother_type: 二次スムーサータイプ（デフォルト: 'zlema'）
            secondary_smoother_period: 二次スムーサー期間（デフォルト: 8）
            use_dynamic_period: 動的期間を使用するか（デフォルト: True）
            detector_type: サイクル検出器タイプ（デフォルト: 'dft_dominant'）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            cycle_part: サイクル部分（デフォルト: 0.4）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 89）
            min_output: 最小出力値（デフォルト: 5）
            enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            percentile_lookback_period: パーセンタイル計算のルックバック期間（デフォルト: 50）
            percentile_low_threshold: パーセンタイル低閾値（デフォルト: 0.25）
            percentile_high_threshold: パーセンタイル高閾値（デフォルト: 0.75）
        """
        super().__init__(f"HyperADX(p={period},mid={midline_period})")
        
        self.period = period
        self.midline_period = midline_period
        
        # 統合カルマンフィルターパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_min_observation_noise = kalman_min_observation_noise
        self.kalman_adaptation_window = kalman_adaptation_window
        
        # ルーフィングフィルターパラメータ
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        
        # 統合スムーサーパラメータ（必須機能）
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        
        # 二次平滑化パラメータ（オプション機能）
        self.use_secondary_smoothing = use_secondary_smoothing
        self.secondary_smoother_type = secondary_smoother_type
        self.secondary_smoother_period = secondary_smoother_period
        
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
        
        # パーセンタイル分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # 依存インジケーターの初期化
        self._init_dependencies()
        
        # キャッシュとリザルト管理
        self._result_cache = {}
        self._cache_keys = []
        self._max_cache_size = 10
        self._latest_result = None
    
    def _init_dependencies(self):
        """依存するインジケーターを初期化"""
        # 統合カルマンフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter and UNIFIED_KALMAN_AVAILABLE:
            try:
                kalman_params = {
                    'kalman_type': self.kalman_filter_type,
                    'process_noise': self.kalman_process_noise,
                    'min_observation_noise': self.kalman_min_observation_noise,
                    'adaptation_window': self.kalman_adaptation_window
                }
                self.kalman_filter = UnifiedKalman(**kalman_params)
                self.logger.debug(f"統合カルマンフィルター({self.kalman_filter_type})を初期化しました")
            except Exception as e:
                self.logger.warning(f"統合カルマンフィルターの初期化に失敗しました: {e}")
                self.use_kalman_filter = False
        elif self.use_kalman_filter and not UNIFIED_KALMAN_AVAILABLE:
            self.logger.warning("UnifiedKalmanが利用できません。カルマンフィルターは無効化されます。")
            self.use_kalman_filter = False
        
        # ルーフィングフィルターの初期化
        self.roofing_filter = None
        if self.use_roofing_filter and ROOFING_FILTER_AVAILABLE:
            try:
                self.roofing_filter = RoofingFilter(
                    hp_cutoff=self.roofing_hp_cutoff,
                    ss_band_edge=self.roofing_ss_band_edge
                )
                self.logger.debug(f"ルーフィングフィルター(hp={self.roofing_hp_cutoff}, ss={self.roofing_ss_band_edge})を初期化しました")
            except Exception as e:
                self.logger.warning(f"ルーフィングフィルターの初期化に失敗しました: {e}")
                self.use_roofing_filter = False
        elif self.use_roofing_filter and not ROOFING_FILTER_AVAILABLE:
            self.logger.warning("RoofingFilterが利用できません。ルーフィングフィルターは無効化されます。")
            self.use_roofing_filter = False
        
        # 統合スムーサーの初期化（必須機能）
        self.smoother = None
        if UNIFIED_SMOOTHER_AVAILABLE:
            try:
                # 動的期間対応スムーサーのチェック
                dynamic_supported_smoothers = ['ultimate_smoother', 'frama', 'super_smoother', 'zero_lag_ema', 'zlema']
                smoother_period_mode = 'fixed'  # デフォルトは固定期間
                
                if self.use_dynamic_period:
                    if self.smoother_type in dynamic_supported_smoothers:
                        smoother_period_mode = 'dynamic'
                        self.logger.debug(f"{self.smoother_type}は動的期間に対応しています。動的モードで初期化します。")
                    else:
                        self.logger.warning(
                            f"{self.smoother_type}は動的期間に対応していません。"
                            f"固定期間モードでスムージングします。"
                            f"動的期間対応スムーサー: {', '.join(dynamic_supported_smoothers)}"
                        )
                
                # スムーサーパラメータの設定
                smoother_params = {
                    'smoother_type': self.smoother_type,
                    'period': self.smoother_period,
                    'src_type': self.smoother_src_type,
                    'period_mode': smoother_period_mode
                }
                
                # 動的期間パラメータの追加
                if smoother_period_mode == 'dynamic':
                    smoother_params.update({
                        'cycle_detector_type': self.detector_type,
                        'cycle_part': self.cycle_part,
                        'max_cycle': self.max_cycle,
                        'min_cycle': self.min_cycle,
                        'max_output': self.max_output,
                        'min_output': self.min_output,
                        'lp_period': self.lp_period,
                        'hp_period': self.hp_period
                    })
                
                self.smoother = UnifiedSmoother(**smoother_params)
                self.logger.debug(f"統合スムーサー({self.smoother_type})を初期化しました（必須機能）")
                
            except Exception as e:
                self.logger.error(f"統合スムーサーの初期化に失敗しました（必須機能）: {e}")
                raise RuntimeError(f"Hyper ADXでは統合スムーサーが必須ですが、初期化に失敗しました: {e}")
        else:
            self.logger.error("UnifiedSmootherが利用できません（必須機能）")
            raise RuntimeError("Hyper ADXでは統合スムーサーが必須ですが、利用できません")
        
        # 二次統合スムーサーの初期化（オプション機能）
        self.secondary_smoother = None
        if self.use_secondary_smoothing and UNIFIED_SMOOTHER_AVAILABLE:
            try:
                # 二次スムーサーは動的期間対応しない（シンプルに保つ）
                secondary_smoother_params = {
                    'smoother_type': self.secondary_smoother_type,
                    'period': self.secondary_smoother_period,
                    'src_type': 'close',  # 固定でclose
                    'period_mode': 'fixed'  # 固定期間モード
                }
                
                self.secondary_smoother = UnifiedSmoother(**secondary_smoother_params)
                self.logger.debug(f"二次統合スムーサー({self.secondary_smoother_type})を初期化しました（オプション機能）")
                
            except Exception as e:
                self.logger.warning(f"二次統合スムーサーの初期化に失敗しました: {e}。二次平滑化機能を無効にします。")
                self.use_secondary_smoothing = False
        elif self.use_secondary_smoothing and not UNIFIED_SMOOTHER_AVAILABLE:
            self.logger.warning("UnifiedSmootherが利用できません。二次平滑化機能を無効にします。")
            self.use_secondary_smoothing = False
        
        # エラーズ統合サイクル検出器の初期化
        self.dc_detector = None
        if self.use_dynamic_period and EHLERS_UNIFIED_DC_AVAILABLE:
            try:
                self.dc_detector = EhlersUnifiedDC(
                    detector_type=self.detector_type,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period,
                    cycle_part=self.cycle_part,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type='hlc3'
                )
                self.logger.debug(f"エラーズ統合サイクル検出器({self.detector_type})を初期化しました")
            except Exception as e:
                self.logger.warning(f"エラーズ統合サイクル検出器の初期化に失敗しました: {e}")
                self.use_dynamic_period = False
        elif self.use_dynamic_period and not EHLERS_UNIFIED_DC_AVAILABLE:
            self.logger.warning("EhlersUnifiedDCが利用できません。動的期間は無効化されます。")
            self.use_dynamic_period = False
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperADXResult:
        """
        Hyper ADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            HyperADXResult: Hyper ADX値とトレンド情報を含む結果
        """
        try:
            # データの検証と準備
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                close = data['close'].values.astype(np.float64)
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                
                high = data[:, 1].astype(np.float64)
                low = data[:, 2].astype(np.float64)
                close = data[:, 3].astype(np.float64)
            
            length = len(high)
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, 'hlc3')
            if isinstance(source_prices, pd.Series):
                source_prices = source_prices.values
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 統合カルマンフィルターによるフィルター処理（オプション）
            filtered_prices = source_prices.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    # カルマンフィルターを適用
                    kalman_result = self.kalman_filter.calculate(source_prices)
                    if kalman_result is not None:
                        # UnifiedKalmanResultオブジェクトの場合
                        if hasattr(kalman_result, 'filtered_values'):
                            filtered_values = kalman_result.filtered_values
                        elif hasattr(kalman_result, 'values'):
                            filtered_values = kalman_result.values
                        else:
                            filtered_values = kalman_result
                        
                        if len(filtered_values) == length:
                            filtered_prices = filtered_values.astype(np.float64)
                        
                        self.logger.debug("統合カルマンフィルターによる価格データのフィルタリングを適用しました")
                    else:
                        self.logger.debug("カルマンフィルターの結果が無効。元の価格を使用します")
                        
                except Exception as e:
                    self.logger.warning(f"統合カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
            
            # 2. サイクル検出器による期間検出
            dynamic_periods = None
            cycle_periods = np.full(length, self.period, dtype=np.float64)
            
            if self.use_dynamic_period and self.dc_detector is not None:
                try:
                    # サイクル検出器を使用してサイクル値を取得
                    cycle_result = self.dc_detector.calculate(data)
                    
                    # cycle_resultの型を確認してvaluesを取得
                    if hasattr(cycle_result, 'values'):
                        cycle_values = cycle_result.values
                    elif hasattr(cycle_result, 'cycle_periods'):
                        cycle_values = cycle_result.cycle_periods
                    elif isinstance(cycle_result, np.ndarray):
                        cycle_values = cycle_result
                    else:
                        cycle_values = np.array(cycle_result) if cycle_result is not None else np.full(length, self.period)
                    
                    # 配列の長さを調整
                    if len(cycle_values) != length:
                        if len(cycle_values) > length:
                            cycle_values = cycle_values[:length]
                        else:
                            # 不足分は最後の値で埋める
                            extended_cycles = np.full(length, self.period, dtype=np.float64)
                            extended_cycles[:len(cycle_values)] = cycle_values
                            if len(cycle_values) > 0:
                                last_valid = cycle_values[-1] if not np.isnan(cycle_values[-1]) else self.period
                                extended_cycles[len(cycle_values):] = last_valid
                            cycle_values = extended_cycles
                    
                    # サイクル値から期間を計算（ハイパーERと同じパターン）
                    valid_cycles = cycle_values[~np.isnan(cycle_values) & (cycle_values > 0)]
                    if len(valid_cycles) > 0:
                        # サイクル値を期間に変換（例：サイクル値の半分を期間とする）
                        dynamic_periods = np.where(
                            ~np.isnan(cycle_values) & (cycle_values > 0),
                            np.clip(cycle_values * 0.5, 5, 50),
                            self.period
                        )
                        cycle_periods = cycle_values.copy()
                        self.logger.debug(f"動的期間を計算: 範囲 {np.min(dynamic_periods):.1f} - {np.max(dynamic_periods):.1f}")
                    else:
                        dynamic_periods = np.full(length, self.period)
                        self.logger.warning("有効なサイクル値が見つからないため、固定期間を使用します")
                    
                except Exception as e:
                    self.logger.warning(f"動的期間計算中にエラー: {e}。固定期間を使用します。")
                    dynamic_periods = None
            
            # 3. ルーフィングフィルターによる価格データのフィルタリング（オプション、ハイパーERと同じ）
            roofing_values = np.full_like(filtered_prices, np.nan)
            
            if self.use_roofing_filter and self.roofing_filter is not None:
                try:
                    roofing_result = self.roofing_filter.calculate(data)
                    roofing_values = roofing_result.values
                    
                    # ルーフィングフィルターの結果を使用（ハイパーERと同じパターン）
                    # ルーフィングフィルターは価格の振動成分を抽出するので、直接フィルタリング済み価格として使用
                    # NaN値が多い場合は元の価格を使用
                    valid_roofing = np.sum(~np.isnan(roofing_values))
                    if valid_roofing > len(roofing_values) * 0.5:  # 有効値が50%以上の場合
                        # ルーフィングフィルター値をソース価格のスケールに調整
                        roofing_range = np.nanmax(roofing_values) - np.nanmin(roofing_values)
                        price_range = np.nanmax(filtered_prices) - np.nanmin(filtered_prices)
                        if roofing_range > 0 and price_range > 0:
                            scale_factor = price_range / roofing_range * 0.1  # 10%の影響度
                            filtered_prices = filtered_prices + roofing_values * scale_factor
                        else:
                            filtered_prices = filtered_prices
                    else:
                        filtered_prices = filtered_prices
                    
                    # NumPy配列として確保
                    if not isinstance(filtered_prices, np.ndarray):
                        filtered_prices = np.array(filtered_prices)
                    if filtered_prices.dtype != np.float64:
                        filtered_prices = filtered_prices.astype(np.float64)
                    
                    self.logger.debug("ルーフィングフィルターによる価格データのフィルタリングを適用しました")
                        
                except Exception as e:
                    self.logger.warning(f"ルーフィングフィルター適用中にエラー: {e}。元の値を使用します。")
                    roofing_values = np.full_like(filtered_prices, np.nan)
            
            # フィルタリング済み価格でHLC価格を調整（ハイパーERでは効率比計算にfiltered_pricesを使用）
            # ADXでは、フィルタリング済み価格を使ってHLC全体を調整
            if np.any(~np.isnan(filtered_prices)):
                # フィルタリング済み価格と元のソース価格の比率を計算
                adjustment_ratio = np.divide(filtered_prices, source_prices, 
                                           out=np.ones_like(filtered_prices), where=source_prices!=0)
                # HLC価格を調整
                high = high * adjustment_ratio
                low = low * adjustment_ratio 
                close = close * adjustment_ratio
            
            # NumPy配列として確保
            if not isinstance(close, np.ndarray):
                close = np.array(close)
            if close.dtype != np.float64:
                close = close.astype(np.float64)
            
            # 4. フィルター処理済価格による DX 計算
            raw_dx, plus_di, minus_di = calculate_dx_values_numba(
                high, low, close, self.period, dynamic_periods
            )
            
            # 5. 統合スムーサーによる DX の平滑化（ADX作成、必須機能）
            final_adx = np.full(length, np.nan, dtype=np.float64)
            if self.smoother is not None:
                try:
                    # NaN値の処理 - スムーサーが正常に動作するようにNaN値を前方補完
                    clean_dx = raw_dx.copy()
                    nan_mask = np.isnan(clean_dx)
                    
                    if np.any(nan_mask):
                        # 最初の有効値を見つけて前方補完
                        first_valid_idx = np.where(~nan_mask)[0]
                        if len(first_valid_idx) > 0:
                            first_valid = first_valid_idx[0]
                            first_value = clean_dx[first_valid]
                            # 最初の有効値より前をその値で補完
                            clean_dx[:first_valid] = first_value
                    
                    # DXをDataFrameに変換
                    if isinstance(data, pd.DataFrame):
                        dx_df = data.copy()
                        dx_df['close'] = clean_dx  # DX値をcloseとして使用
                    else:
                        dx_df = pd.DataFrame({
                            'open': data[:, 0],
                            'high': data[:, 1],
                            'low': data[:, 2],
                            'close': clean_dx,  # DX値をcloseとして使用
                            'volume': np.ones(length)
                        })
                    
                    # 統合スムーサーでADXを計算
                    smoother_result = self.smoother.calculate(dx_df)
                    if smoother_result is not None:
                        # UnifiedSmootherResultオブジェクトの場合
                        if hasattr(smoother_result, 'values'):
                            smoothed_values = smoother_result.values
                        else:
                            smoothed_values = smoother_result
                        
                        if len(smoothed_values) == length:
                            final_adx = smoothed_values.astype(np.float64)
                            self.logger.debug("統合スムーサーによるDX値の平滑化（ADX作成）を適用しました")
                        else:
                            self.logger.warning("統合スムーサーの結果が無効です")
                            final_adx = clean_dx.copy()
                    else:
                        self.logger.warning("統合スムーサーの結果が無効です")
                        final_adx = clean_dx.copy()
                        
                except Exception as e:
                    self.logger.error(f"統合スムーサー適用中にエラー: {e}")
                    # フォールバック: 単純なEMAでADXを計算
                    for i in range(self.period - 1, length):
                        if not np.isnan(raw_dx[i]):
                            alpha = 2.0 / (self.period + 1.0)
                            if i == self.period - 1:
                                final_adx[i] = raw_dx[i]
                            else:
                                final_adx[i] = (raw_dx[i] * alpha) + (final_adx[i-1] * (1 - alpha))
            else:
                raise RuntimeError("統合スムーサーが利用できません（必須機能）")
            
            # 6. 二次平滑化処理（オプション）
            secondary_smoothed_adx = None
            if self.use_secondary_smoothing and self.secondary_smoother is not None:
                try:
                    # 一次平滑化されたADX値（final_adx）をさらに平滑化
                    # NaN値の処理
                    clean_adx = final_adx.copy()
                    nan_mask = np.isnan(clean_adx)
                    
                    if np.any(nan_mask):
                        # 最初の有効値を見つけて前方補完
                        first_valid_idx = np.where(~nan_mask)[0]
                        if len(first_valid_idx) > 0:
                            first_valid = first_valid_idx[0]
                            first_value = clean_adx[first_valid]
                            # 最初の有効値より前をその値で補完
                            clean_adx[:first_valid] = first_value
                    
                    # ADXをDataFrameに変換
                    if isinstance(data, pd.DataFrame):
                        adx_df = data.copy()
                        adx_df['close'] = clean_adx  # ADX値をcloseとして使用
                    else:
                        adx_df = pd.DataFrame({
                            'open': data[:, 0],
                            'high': data[:, 1],
                            'low': data[:, 2],
                            'close': clean_adx,  # ADX値をcloseとして使用
                            'volume': np.ones(length)
                        })
                    
                    # 二次統合スムーサーでさらに平滑化
                    secondary_result = self.secondary_smoother.calculate(adx_df)
                    if secondary_result is not None:
                        # UnifiedSmootherResultオブジェクトの場合
                        if hasattr(secondary_result, 'values'):
                            secondary_values = secondary_result.values
                        else:
                            secondary_values = secondary_result
                        
                        if len(secondary_values) == length:
                            secondary_smoothed_adx = secondary_values.astype(np.float64)
                            # 元のNaN位置を復元
                            if np.any(nan_mask):
                                secondary_smoothed_adx[nan_mask] = np.nan
                            self.logger.debug("二次統合スムーサーによるADX値の平滑化を適用しました")
                        else:
                            self.logger.warning("二次統合スムーサーの結果が無効です")
                    else:
                        self.logger.warning("二次統合スムーサーの結果が無効です")
                        
                except Exception as e:
                    self.logger.warning(f"二次統合スムーサー適用中にエラー: {e}")
            
            # 7. ミッドライン計算とトレンド判定
            midline, trend_signal = calculate_midline_and_trend_signal_hyper(
                final_adx, self.midline_period
            )
            
            # パーセンタイル分析の実行
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                final_adx, 'trend'
            )
            
            # 結果の作成
            result = HyperADXResult(
                values=final_adx.copy(),
                raw_dx=raw_dx.copy(),
                secondary_smoothed=secondary_smoothed_adx.copy() if secondary_smoothed_adx is not None else None,
                midline=midline.copy(),
                trend_signal=trend_signal.copy(),
                filtered_prices=filtered_prices.copy(),
                roofing_values=roofing_values.copy(),
                plus_di=plus_di.copy(),
                minus_di=minus_di.copy(),
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
            
            self._latest_result = result
            
            # 基底クラス用の値設定
            self._values = final_adx
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Hyper ADX '{self.name}' 計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は NaN で埋めた結果を返す
            length = len(data) if hasattr(data, '__len__') else 100
            empty_array = np.full(length, np.nan)
            error_result = HyperADXResult(
                values=empty_array,
                raw_dx=empty_array,
                secondary_smoothed=None,
                midline=empty_array,
                trend_signal=empty_array,
                filtered_prices=empty_array,
                roofing_values=empty_array,
                plus_di=empty_array,
                minus_di=empty_array,
                cycle_periods=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
            return error_result
    
    def _get_latest_result(self) -> Optional[HyperADXResult]:
        """最新の結果を取得"""
        return self._latest_result
    
    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._latest_result = None
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存インジケーターのリセット
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        if self.roofing_filter and hasattr(self.roofing_filter, 'reset'):
            self.roofing_filter.reset()
        if self.smoother and hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
    
    # 追加のgetterメソッド
    def get_raw_dx(self) -> Optional[np.ndarray]:
        """生のDX値を取得"""
        result = self._get_latest_result()
        return result.raw_dx.copy() if result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_trend_signal(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        result = self._get_latest_result()
        return result.trend_signal.copy() if result else None
    
    def get_filtered_prices(self) -> Optional[np.ndarray]:
        """カルマンフィルター値を取得"""
        result = self._get_latest_result()
        return result.filtered_prices.copy() if result else None
    
    def get_roofing_values(self) -> Optional[np.ndarray]:
        """ルーフィングフィルター値を取得"""
        result = self._get_latest_result()
        return result.roofing_values.copy() if result else None
    
    def get_plus_di(self) -> Optional[np.ndarray]:
        """+DI値を取得"""
        result = self._get_latest_result()
        return result.plus_di.copy() if result else None
    
    def get_minus_di(self) -> Optional[np.ndarray]:
        """-DI値を取得"""
        result = self._get_latest_result()
        return result.minus_di.copy() if result else None
    
    def get_cycle_periods(self) -> Optional[np.ndarray]:
        """サイクル期間値を取得"""
        result = self._get_latest_result()
        return result.cycle_periods.copy() if result else None
    
    # パーセンタイル分析関連のgetter メソッド
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        result = self._get_latest_result()
        return result.percentiles.copy() if result else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        result = self._get_latest_result()
        return result.trend_state.copy() if result else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        result = self._get_latest_result()
        return result.trend_intensity.copy() if result else None
    
    def get_indicator_info(self) -> dict:
        """インジケーター情報を取得"""
        info = {
            'name': self.name,
            'period': self.period,
            'midline_period': self.midline_period,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'use_roofing_filter': self.use_roofing_filter,
            'roofing_hp_cutoff': self.roofing_hp_cutoff if self.use_roofing_filter else None,
            'roofing_ss_band_edge': self.roofing_ss_band_edge if self.use_roofing_filter else None,
            'smoother_type': self.smoother_type,  # 必須機能
            'smoother_period': self.smoother_period,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'description': 'ADXベースの統合型トレンドインジケーター（0-1範囲、高値=強いトレンド、カルマン・ルーフィング・スムーサー統合）'
        }
        return info


# 便利関数
def calculate_hyper_adx(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 14,
    midline_period: int = 100,
    use_kalman_filter: bool = True,
    kalman_filter_type: str = 'adaptive',
    use_roofing_filter: bool = True,
    smoother_type: str = 'frama',
    smoother_period: int = 12,
    use_dynamic_period: bool = True,
    detector_type: str = 'dft_dominant',
    enable_percentile_analysis: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Hyper ADX（統合型Average Directional Index）を計算する便利関数
    
    Args:
        data: 価格データ（DataFrameまたはNumPy配列）
        period: ADX計算期間
        midline_period: ミッドライン計算期間
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        use_roofing_filter: ルーフィングフィルターを使用するか
        smoother_type: 統合スムーサータイプ（必須機能）
        smoother_period: スムーサー期間
        use_dynamic_period: 動的期間適応を使用するか
        detector_type: サイクル検出器タイプ
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        **kwargs: その他のパラメータ
        
    Returns:
        Hyper ADX値の配列
    """
    indicator = HyperADX(
        period=period,
        midline_period=midline_period,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        use_roofing_filter=use_roofing_filter,
        smoother_type=smoother_type,
        smoother_period=smoother_period,
        use_dynamic_period=use_dynamic_period,
        detector_type=detector_type,
        enable_percentile_analysis=enable_percentile_analysis,
        **kwargs
    )
    
    result = indicator.calculate(data)
    return result.values