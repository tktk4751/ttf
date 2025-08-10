#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .str import STR
from .ultimate_ma import UltimateMA
from .smoother.ultimate_smoother import UltimateSmoother


@dataclass
class UltimateTrendResult:
    """アルティメットトレンドの計算結果"""
    values: np.ndarray           # スーパートレンドライン値（Ultimate MAフィルタ済み）
    upper_band: np.ndarray       # 上側のバンド価格（表示用）
    lower_band: np.ndarray       # 下側のバンド価格（表示用）
    final_upper_band: np.ndarray # 調整済み上側バンド（計算用）
    final_lower_band: np.ndarray # 調整済み下側バンド（計算用）
    trend: np.ndarray           # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    str_values: np.ndarray      # 使用されたSTR値
    filtered_midline: np.ndarray # フィルタ済みミッドライン（選択されたフィルタリングレベル）
    raw_midline: np.ndarray     # 元のHLC3ミッドライン
    ukf_values: np.ndarray      # UKFフィルター後の値
    ultimate_smooth_values: np.ndarray # アルティメットスムーザー後の値
    zero_lag_values: np.ndarray # ゼロラグEMA後の値
    amplitude: np.ndarray       # ヒルベルト変換振幅
    phase: np.ndarray          # ヒルベルト変換位相
    filtering_mode: int         # 使用されたフィルタリングモード
    midline_type: str          # 使用されたミッドラインタイプ ('ultimate_ma' or 'ultimate_smoother')
    # Ultimate MAのトレンドシグナル情報
    trend_signals: np.ndarray   # Ultimate MAのトレンドシグナル（1=up, -1=down, 0=range）
    current_trend: str          # Ultimate MAの現在のトレンド（'up', 'down', 'range'）
    current_trend_value: int    # Ultimate MAの現在のトレンド値（1, -1, 0）


# Ultimate MAから各段階のフィルタリング結果を取得


def calculate_ultimate_trend_bands(ultimate_ma_result=None, ultimate_smoother_result=None, 
                                  close: np.ndarray = None, str_values: np.ndarray = None, 
                                  multiplier: float = 3.0, filtering_mode: int = 0,
                                  midline_type: str = 'ultimate_ma') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    アルティメットトレンドを計算する（Ultimate MAまたはUltimateSmootherを使用）
    
    Args:
        ultimate_ma_result: UltimateMAの計算結果（midline_type='ultimate_ma'の場合）
        ultimate_smoother_result: UltimateSmootherの計算結果（midline_type='ultimate_smoother'の場合）
        close: 終値の配列（トレンド判定用）
        str_values: STRの配列
        multiplier: STR乗数
        filtering_mode: フィルタリングモード
                       0 = カルマンフィルター後の値をミッドラインに使用（①のみ）
                       1 = 完全5段階フィルタ済み値をミッドラインに使用（①②③④⑤まで）
        midline_type: ミッドラインタイプ ('ultimate_ma' or 'ultimate_smoother')
    
    Returns:
        Tuple: (上側バンド, 下側バンド, トレンド方向, ミッドライン, 元HLC3, UKF値, アルティメットスムーザー値, ゼロラグ, 振幅, 位相, 調整済み上側バンド, 調整済み下側バンド, ミッドラインタイプ)
    """
    if midline_type == 'ultimate_ma':
        # Ultimate MAの結果から必要な値を取得
        final_filtered = ultimate_ma_result.values          # Ultimate MAの最終結果
        raw_hlc3 = ultimate_ma_result.raw_values           # 元のHLC3
        ukf_values = ultimate_ma_result.ukf_values         # hlc3フィルター後
        ultimate_smooth_values = ultimate_ma_result.ultimate_smooth_values  # アルティメットスムーザー後
        zero_lag_values = ultimate_ma_result.zero_lag_values  # ゼロラグEMA後
        amplitude = ultimate_ma_result.amplitude           # ヒルベルト変換振幅
        phase = ultimate_ma_result.phase                   # ヒルベルト変換位相
        
        # フィルタリングモードに応じてミッドラインを選択
        if filtering_mode == 0:
            # モード0: UKFフィルター後の値をミッドラインに使用（①のみ）
            midline = ukf_values
        else:
            # モード1: 完全5段階フィルタ済み値をミッドラインに使用（①②③④⑤まで）
            midline = final_filtered
            
    elif midline_type == 'ultimate_smoother':
        # UltimateSmootherの結果から必要な値を取得
        ultimate_smooth_values = ultimate_smoother_result.values  # アルティメットスムーザー値
        raw_hlc3 = ultimate_smoother_result.values  # UltimateSmootherの場合は元値も同じ
        ukf_values = ultimate_smoother_result.values  # UltimateSmootherの場合はUKF値も同じ
        zero_lag_values = ultimate_smoother_result.values  # UltimateSmootherの場合はゼロラグ値も同じ
        amplitude = np.full(len(ultimate_smooth_values), np.nan, dtype=np.float64)  # UltimateSmootherには振幅情報なし
        phase = np.full(len(ultimate_smooth_values), np.nan, dtype=np.float64)      # UltimateSmootherには位相情報なし
        
        # UltimateSmootherの場合は常にアルティメットスムーザー値をミッドラインに使用
        midline = ultimate_smooth_values
        
    else:
        raise ValueError(f"無効なmidline_type: {midline_type}. 'ultimate_ma' または 'ultimate_smoother' を指定してください。")
    
    length = len(midline)
    
    # 選択されたミッドラインを基準としたバンド計算（初期値）
    basic_upper_band = midline + multiplier * str_values
    basic_lower_band = midline - multiplier * str_values
    
    # 調整可能なバンド配列（スーパートレンドロジック用）
    final_upper_band = basic_upper_band.copy()
    final_lower_band = basic_lower_band.copy()
    
    # トレンド方向の配列を初期化
    trend = np.zeros(length, dtype=np.int8)
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if (not np.isnan(basic_upper_band[i]) and 
            not np.isnan(basic_lower_band[i]) and 
            not np.isnan(close[i]) and
            not np.isnan(midline[i])):
            first_valid_idx = i
            break
    
    # 有効な値が見つからない場合は全てNaN/0を返す
    if first_valid_idx < 0:
        upper_band[:] = np.nan
        lower_band[:] = np.nan
        final_upper_band[:] = np.nan
        final_lower_band[:] = np.nan
        return upper_band, lower_band, trend, midline, raw_hlc3, ukf_values, ultimate_smooth_values, zero_lag_values, amplitude, phase, final_upper_band, final_lower_band, midline_type
    
    # 最初の値を設定（終値とフィルタ済みHLC3上側バンドで比較）
    trend[first_valid_idx] = 1 if close[first_valid_idx] > final_upper_band[first_valid_idx] else -1
    
    # 最初の有効インデックスまでは無効値
    for i in range(first_valid_idx):
        upper_band[i] = np.nan
        lower_band[i] = np.nan
        trend[i] = 0
    
    # 最初の有効値のバンド設定
    if trend[first_valid_idx] == 1:
        upper_band[first_valid_idx] = np.nan
        lower_band[first_valid_idx] = final_lower_band[first_valid_idx]
    else:
        upper_band[first_valid_idx] = final_upper_band[first_valid_idx]
        lower_band[first_valid_idx] = np.nan
    
    # バンドとトレンドの計算（スーパートレンドロジック完全準拠）
    for i in range(first_valid_idx + 1, length):
        # データが無効な場合は前の値を維持
        if (np.isnan(close[i]) or 
            np.isnan(basic_upper_band[i]) or 
            np.isnan(basic_lower_band[i]) or
            np.isnan(midline[i])):
            trend[i] = trend[i-1]
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            final_upper_band[i] = final_upper_band[i-1]
            final_lower_band[i] = final_lower_band[i-1]
            continue
        
        # トレンド判定（スーパートレンドロジック完全準拠）
        if close[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif close[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        # バンドの調整（スーパートレンドロジック完全準拠）
        # 現在のバンドの基本値を設定
        final_upper_band[i] = basic_upper_band[i]
        final_lower_band[i] = basic_lower_band[i]
        
        # トレンド継続時のみバンドを調整
        if trend[i] == trend[i-1]:
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定（スーパートレンドロジック完全準拠）
        if trend[i] == 1:
            # 上昇トレンド：上側バンドは非表示、下側バンドのみ表示
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            # 下降トレンド：下側バンドは非表示、上側バンドのみ表示
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return upper_band, lower_band, trend, midline, raw_hlc3, ukf_values, ultimate_smooth_values, zero_lag_values, amplitude, phase, final_upper_band, final_lower_band, midline_type


@njit(fastmath=True, cache=True)
def calculate_ultimate_trend_line(upper_band: np.ndarray, lower_band: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """
    アルティメットトレンドラインを計算する
    
    Args:
        upper_band: 上側バンド（表示用）
        lower_band: 下側バンド（表示用）
        trend: トレンド方向
    
    Returns:
        アルティメットトレンドラインの配列
    """
    length = len(trend)
    ultimate_trend = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if trend[i] == 1:
            # 上昇トレンド時は下側バンドを使用
            ultimate_trend[i] = lower_band[i] if not np.isnan(lower_band[i]) else np.nan
        elif trend[i] == -1:
            # 下降トレンド時は上側バンドを使用
            ultimate_trend[i] = upper_band[i] if not np.isnan(upper_band[i]) else np.nan
        else:
            # トレンドが0の場合はNaN
            ultimate_trend[i] = np.nan
    
    return ultimate_trend 


class UltimateTrend(Indicator):
    """
    🚀 **アルティメットトレンド - Ultimate MAフィルタ統合スーパートレンド V2.0**
    
    🎯 **Ultimate MA統合システム:**
    - **Ultimate MA**: 6段階革新的フィルタリングをミッドラインに使用
      1. 適応的カルマンフィルター: 動的ノイズレベル推定・リアルタイム除去
      2. スーパースムーザーフィルター: John Ehlers改良版・ゼロ遅延設計
      3. ゼロラグEMA: 遅延完全除去・予測的補正
      4. ヒルベルト変換フィルター: 位相遅延ゼロ・瞬時振幅/位相
      5. 適応的ノイズ除去: AI風学習型・振幅連動調整
      6. リアルタイムトレンド検出: 超低遅延・即座反応
    
    - **スーパートレンドロジック**: 従来のスーパートレンドアルゴリズムを継承
      - ATRベースのバンド計算
      - 動的トレンド判定
      - ブレイクアウト検出
    
    🏆 **革新的特徴:**
    - **Ultimate MAミッドライン**: フィルタ済みHLC3をミッドラインに使用
    - **ノイズ除去**: Ultimate MAの6段階革新的フィルタリング
    - **超低遅延**: リアルタイム処理最適化
    - **位相遅延ゼロ**: ヒルベルト変換適用
    - **適応的学習**: AI風ノイズレベル推定
    - **完全統合処理**: Ultimate MAの各段階結果も取得可能
    
    🎨 **表示情報:**
    - アルティメットトレンドメインライン（緑=上昇トレンド、赤=下降トレンド）
    - Ultimate MAフィルタ済みミッドライン（参考線）
    - Ultimate MAの各フィルター段階の中間結果（オプション）
    """
    
    def __init__(self, 
                 length: int = 13,
                 multiplier: float = 3.0,
                 ultimate_smoother_period: int = 10,
                 zero_lag_period: int = 21,
                 filtering_mode: int = 1,
                 midline_type: str = 'ultimate_ma',  # ミッドラインタイプ
                 # Ultimate MAの動的適応パラメータ
                 zero_lag_period_mode: str = 'dynamic',
                 realtime_window_mode: str = 'dynamic',
                 # Ultimate MAのゼロラグ用サイクル検出器パラメータ
                 zl_cycle_detector_type: str = 'absolute_ultimate',
                 zl_cycle_detector_cycle_part: float = 0.5,
                 zl_cycle_detector_max_cycle: int = 120,
                 zl_cycle_detector_min_cycle: int = 5,
                 zl_cycle_period_multiplier: float = 1.0,
                 zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
                 # UltimateSmootherのパラメータ
                 us_period: float = 20.0,  # UltimateSmootherの期間
                 us_src_type: str = 'hlc3',  # UltimateSmootherのソースタイプ
                 us_period_mode: str = 'dynamic',  # UltimateSmootherの期間モード
                 us_ukf_params: Optional[Dict] = None,  # UltimateSmootherのUKFパラメータ
                 us_cycle_detector_type: str = 'absolute_ultimate',  # UltimateSmootherのサイクル検出器タイプ
                 us_cycle_detector_cycle_part: float = 1.0,  # UltimateSmootherのサイクル検出器のサイクル部分倍率
                 us_cycle_detector_max_cycle: int = 120,  # UltimateSmootherのサイクル検出器の最大サイクル期間
                 us_cycle_detector_min_cycle: int = 5,  # UltimateSmootherのサイクル検出器の最小サイクル期間
                 us_cycle_period_multiplier: float = 1.0,  # UltimateSmootherのサイクル期間の乗数
                 us_cycle_detector_period_range: Tuple[int, int] = (5, 120)  # UltimateSmootherのサイクル検出器の周期範囲
                 ):
        """
        アルティメットトレンドインジケーターのコンストラクタ
        
        Args:
            length: ATR計算期間（デフォルト: 13）
            multiplier: ATR乗数（デフォルト: 2.0）
            ultimate_smoother_period: スーパースムーザー期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            filtering_mode: フィルタリングモード（デフォルト: 1）
                           0 = カルマンフィルター後の値をミッドラインに使用（①のみ）
                           1 = 完全5段階フィルタ済み値をミッドラインに使用（①②③④⑤まで）
            midline_type: ミッドラインタイプ ('ultimate_ma' or 'ultimate_smoother')
            # Ultimate MAの動的適応パラメータ
            zero_lag_period_mode: ゼロラグEMA期間モード（'dynamic' or 'fixed'）
            realtime_window_mode: リアルタイムウィンドウモード（'dynamic' or 'fixed'）
            # Ultimate MAのゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル部分
            zl_cycle_detector_max_cycle: ゼロラグ用最大サイクル
            zl_cycle_detector_min_cycle: ゼロラグ用最小サイクル
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間乗数
            zl_cycle_detector_period_range: ゼロラグ用period_rangeパラメータ
            # UltimateSmootherのパラメータ
            us_period: UltimateSmootherの期間
            us_src_type: UltimateSmootherのソースタイプ
            us_period_mode: UltimateSmootherの期間モード
            us_ukf_params: UltimateSmootherのUKFパラメータ
            us_cycle_detector_type: UltimateSmootherのサイクル検出器タイプ
            us_cycle_detector_cycle_part: UltimateSmootherのサイクル検出器のサイクル部分倍率
            us_cycle_detector_max_cycle: UltimateSmootherのサイクル検出器の最大サイクル期間
            us_cycle_detector_min_cycle: UltimateSmootherのサイクル検出器の最小サイクル期間
            us_cycle_period_multiplier: UltimateSmootherのサイクル期間の乗数
            us_cycle_detector_period_range: UltimateSmootherのサイクル検出器の周期範囲
        """
        # 指標名の作成
        mode_desc = "Kalman" if filtering_mode == 0 else "FullFiltered"
        indicator_name = f"UltimateTrend(STR={length},mult={multiplier},ss={ultimate_smoother_period},zl={zero_lag_period},mode={mode_desc},midline={midline_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.length = length
        self.multiplier = multiplier
        self.ultimate_smoother_period = ultimate_smoother_period
        self.zero_lag_period = zero_lag_period
        self.filtering_mode = filtering_mode
        self.midline_type = midline_type.lower()  # ミッドラインタイプを保存
        
        # Ultimate MAの動的適応パラメータを保存
        self.zero_lag_period_mode = zero_lag_period_mode
        self.realtime_window_mode = realtime_window_mode
        self.zl_cycle_detector_type = zl_cycle_detector_type
        self.zl_cycle_detector_cycle_part = zl_cycle_detector_cycle_part
        self.zl_cycle_detector_max_cycle = zl_cycle_detector_max_cycle
        self.zl_cycle_detector_min_cycle = zl_cycle_detector_min_cycle
        self.zl_cycle_period_multiplier = zl_cycle_period_multiplier
        self.zl_cycle_detector_period_range = zl_cycle_detector_period_range
        
        # UltimateSmootherのパラメータを保存
        self.us_period = us_period
        self.us_src_type = us_src_type
        self.us_period_mode = us_period_mode
        self.us_ukf_params = us_ukf_params
        self.us_cycle_detector_type = us_cycle_detector_type
        self.us_cycle_detector_cycle_part = us_cycle_detector_cycle_part
        self.us_cycle_detector_max_cycle = us_cycle_detector_max_cycle
        self.us_cycle_detector_min_cycle = us_cycle_detector_min_cycle
        self.us_cycle_period_multiplier = us_cycle_period_multiplier
        self.us_cycle_detector_period_range = us_cycle_detector_period_range
        
        # パラメータ検証
        if self.length <= 0:
            raise ValueError("lengthは0より大きい必要があります")
        if self.multiplier <= 0:
            raise ValueError("multiplierは0より大きい必要があります")
        if self.ultimate_smoother_period <= 0:
            raise ValueError("ultimate_smoother_periodは0より大きい必要があります")
        if self.zero_lag_period <= 0:
            raise ValueError("zero_lag_periodは0より大きい必要があります")
        if self.filtering_mode not in [0, 1]:
            raise ValueError("filtering_modeは0または1である必要があります")
        if self.midline_type not in ['ultimate_ma', 'ultimate_smoother']:
            raise ValueError(f"無効なmidline_type: {midline_type}. 'ultimate_ma' または 'ultimate_smoother' を指定してください。")
        
        # STRインジケーターを初期化
        self.str_indicator = STR(period=self.length)
        
        # ミッドラインタイプに応じてインジケーターを初期化
        if self.midline_type == 'ultimate_ma':
            # Ultimate MAインジケーターを初期化（全パラメータを含む）
            self.ultimate_ma = UltimateMA(
                ultimate_smoother_period=self.ultimate_smoother_period,
                zero_lag_period=self.zero_lag_period,
                src_type='hlc3',
                # 動的適応パラメータ
                zero_lag_period_mode=self.zero_lag_period_mode,
                realtime_window_mode=self.realtime_window_mode,
                # ゼロラグ用サイクル検出器パラメータ
                zl_cycle_detector_type=self.zl_cycle_detector_type,
                zl_cycle_detector_cycle_part=self.zl_cycle_detector_cycle_part,
                zl_cycle_detector_max_cycle=self.zl_cycle_detector_max_cycle,
                zl_cycle_detector_min_cycle=self.zl_cycle_detector_min_cycle,
                zl_cycle_period_multiplier=self.zl_cycle_period_multiplier,
                zl_cycle_detector_period_range=self.zl_cycle_detector_period_range
            )
            self.ultimate_smoother = None  # Ultimate MA使用時はUltimateSmootherは不要
        else:
            # UltimateSmootherインジケーターを初期化
            self.ultimate_smoother = UltimateSmoother(
                period=self.us_period,
                src_type=self.us_src_type,
                ukf_params=self.us_ukf_params,
                period_mode=self.us_period_mode,
                cycle_detector_type=self.us_cycle_detector_type,
                cycle_detector_cycle_part=self.us_cycle_detector_cycle_part,
                cycle_detector_max_cycle=self.us_cycle_detector_max_cycle,
                cycle_detector_min_cycle=self.us_cycle_detector_min_cycle,
                cycle_period_multiplier=self.us_cycle_period_multiplier,
                cycle_detector_period_range=self.us_cycle_detector_period_range
            )
            self.ultimate_ma = None  # UltimateSmoother使用時はUltimate MAは不要
        
        self._cache = {}
        self._result: Optional[UltimateTrendResult] = None

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateTrendResult:
        """
        アルティメットトレンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            UltimateTrendResult: アルティメットトレンドの値と関連情報を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache:
                cached_result = self._cache[data_hash]
                return UltimateTrendResult(
                    values=cached_result.values.copy(),
                    upper_band=cached_result.upper_band.copy(),
                    lower_band=cached_result.lower_band.copy(),
                    final_upper_band=cached_result.final_upper_band.copy(),
                    final_lower_band=cached_result.final_lower_band.copy(),
                    trend=cached_result.trend.copy(),
                    str_values=cached_result.str_values.copy(),
                    filtered_midline=cached_result.filtered_midline.copy(),
                    raw_midline=cached_result.raw_midline.copy(),
                    ukf_values=cached_result.ukf_values.copy(),
                    ultimate_smooth_values=cached_result.ultimate_smooth_values.copy(),
                    zero_lag_values=cached_result.zero_lag_values.copy(),
                    amplitude=cached_result.amplitude.copy(),
                    phase=cached_result.phase.copy(),
                    filtering_mode=cached_result.filtering_mode,
                    midline_type=cached_result.midline_type,
                    trend_signals=cached_result.trend_signals.copy(),
                    current_trend=cached_result.current_trend,
                    current_trend_value=cached_result.current_trend_value
                )
            
            # データの検証
            if data is None or len(data) == 0:
                return self._create_empty_result()
            
            # 終値の取得
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameに'close'カラムが必要です")
                close = data['close'].values
            else:
                # NumPy配列の場合、最後の列を終値として使用
                close = data[:, -1] if data.ndim > 1 else data
            
            # STRの計算
            str_result = self.str_indicator.calculate(data)
            str_array = str_result.values
            
            # ミッドラインタイプに応じて計算
            if self.midline_type == 'ultimate_ma':
                # Ultimate MAの計算
                ultimate_ma_result = self.ultimate_ma.calculate(data)
                
                # アルティメットトレンドバンドの計算
                (upper_band, lower_band, trend, filtered_midline, raw_hlc3, 
                 ukf_values, ultimate_smooth_values, zero_lag_values, 
                 amplitude, phase, final_upper_band, final_lower_band, midline_type) = calculate_ultimate_trend_bands(
                    ultimate_ma_result=ultimate_ma_result, 
                    ultimate_smoother_result=None,
                    close=close, 
                    str_values=str_array, 
                    multiplier=self.multiplier, 
                    filtering_mode=self.filtering_mode, 
                    midline_type=self.midline_type
                )
                
                # トレンド統計の計算
                current_trend = self._calculate_trend_stats(trend)
                current_trend_value = 1 if current_trend == 'up' else (-1 if current_trend == 'down' else 0)
                
                # 結果の作成
                result = UltimateTrendResult(
                    values=calculate_ultimate_trend_line(upper_band, lower_band, trend),
                    upper_band=upper_band,
                    lower_band=lower_band,
                    final_upper_band=final_upper_band,
                    final_lower_band=final_lower_band,
                    trend=trend,
                    str_values=str_array,
                    filtered_midline=filtered_midline,
                    raw_midline=raw_hlc3,
                    ukf_values=ukf_values,
                    ultimate_smooth_values=ultimate_smooth_values,
                    zero_lag_values=zero_lag_values,
                    amplitude=amplitude,
                    phase=phase,
                    filtering_mode=self.filtering_mode,
                    midline_type=midline_type,
                    # Ultimate MAのトレンドシグナル情報
                    trend_signals=ultimate_ma_result.trend_signals,
                    current_trend=current_trend,
                    current_trend_value=current_trend_value
                )
                
            else:  # midline_type == 'ultimate_smoother'
                # UltimateSmootherの計算
                ultimate_smoother_result = self.ultimate_smoother.calculate(data)
                
                # アルティメットトレンドバンドの計算
                (upper_band, lower_band, trend, filtered_midline, raw_hlc3, 
                 ukf_values, ultimate_smooth_values, zero_lag_values, 
                 amplitude, phase, final_upper_band, final_lower_band, midline_type) = calculate_ultimate_trend_bands(
                    ultimate_ma_result=None,
                    ultimate_smoother_result=ultimate_smoother_result,
                    close=close, 
                    str_values=str_array, 
                    multiplier=self.multiplier, 
                    filtering_mode=self.filtering_mode, 
                    midline_type=self.midline_type
                )
                
                # トレンド統計の計算
                current_trend = self._calculate_trend_stats(trend)
                current_trend_value = 1 if current_trend == 'up' else (-1 if current_trend == 'down' else 0)
                
                # 結果の作成
                result = UltimateTrendResult(
                    values=calculate_ultimate_trend_line(upper_band, lower_band, trend),
                    upper_band=upper_band,
                    lower_band=lower_band,
                    final_upper_band=final_upper_band,
                    final_lower_band=final_lower_band,
                    trend=trend,
                    str_values=str_array,
                    filtered_midline=filtered_midline,
                    raw_midline=raw_hlc3,
                    ukf_values=ukf_values,
                    ultimate_smooth_values=ultimate_smooth_values,
                    zero_lag_values=zero_lag_values,
                    amplitude=amplitude,
                    phase=phase,
                    filtering_mode=self.filtering_mode,
                    midline_type=midline_type,
                    # UltimateSmootherの場合は空のトレンドシグナル情報
                    trend_signals=np.zeros(len(trend), dtype=np.int8),
                    current_trend=current_trend,
                    current_trend_value=current_trend_value
                )
            
            # キャッシュに保存
            self._cache[data_hash] = result
            self._values = result.values  # 基底クラスの要件を満たすため
            
            self.logger.debug(f"Ultimate Trend 計算完了 - ミッドラインタイプ: {self.midline_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"UltimateTrend計算中にエラー: {e}")
            return self._create_empty_result()

    def _create_empty_result(self, length: int = 0) -> UltimateTrendResult:
        """空の結果を作成する"""
        return UltimateTrendResult(
            values=np.full(length, np.nan, dtype=np.float64),
            upper_band=np.full(length, np.nan, dtype=np.float64),
            lower_band=np.full(length, np.nan, dtype=np.float64),
            final_upper_band=np.full(length, np.nan, dtype=np.float64),
            final_lower_band=np.full(length, np.nan, dtype=np.float64),
            trend=np.zeros(length, dtype=np.int8),
            str_values=np.full(length, np.nan, dtype=np.float64),
            filtered_midline=np.full(length, np.nan, dtype=np.float64),
            raw_midline=np.full(length, np.nan, dtype=np.float64),
            ukf_values=np.full(length, np.nan, dtype=np.float64),
            ultimate_smooth_values=np.full(length, np.nan, dtype=np.float64),
            zero_lag_values=np.full(length, np.nan, dtype=np.float64),
            amplitude=np.full(length, np.nan, dtype=np.float64),
            phase=np.full(length, np.nan, dtype=np.float64),
            filtering_mode=self.filtering_mode,
            midline_type='ultimate_ma',
            trend_signals=np.zeros(length, dtype=np.int8),
            current_trend='range',
            current_trend_value=0
        )

    def _calculate_trend_stats(self, trend: np.ndarray) -> str:
        """トレンド統計を計算する"""
        valid_trends = trend[trend != 0]
        if len(valid_trends) == 0:
            return "有効なトレンドなし"
        
        uptrend_count = np.sum(valid_trends == 1)
        downtrend_count = np.sum(valid_trends == -1)
        total_valid = len(valid_trends)
        
        uptrend_pct = (uptrend_count / total_valid) * 100
        downtrend_pct = (downtrend_count / total_valid) * 100
        
        return f"上昇トレンド: {uptrend_pct:.1f}%, 下降トレンド: {downtrend_pct:.1f}%"

    def get_values(self) -> Optional[np.ndarray]:
        """アルティメットトレンドラインを取得する"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_trend(self) -> Optional[np.ndarray]:
        """トレンド方向を取得する"""
        if self._result is not None:
            return self._result.trend.copy()
        return None

    def get_upper_band(self) -> Optional[np.ndarray]:
        """上側バンドを取得する"""
        if self._result is not None:
            return self._result.upper_band.copy()
        return None

    def get_lower_band(self) -> Optional[np.ndarray]:
        """下側バンドを取得する"""
        if self._result is not None:
            return self._result.lower_band.copy()
        return None

    def get_final_upper_band(self) -> Optional[np.ndarray]:
        """調整済み上側バンドを取得する"""
        if self._result is not None:
            return self._result.final_upper_band.copy()
        return None

    def get_final_lower_band(self) -> Optional[np.ndarray]:
        """調整済み下側バンドを取得する"""
        if self._result is not None:
            return self._result.final_lower_band.copy()
        return None

    def get_filtered_midline(self) -> Optional[np.ndarray]:
        """フィルタ済みHLC3ミッドラインを取得する"""
        if self._result is not None:
            return self._result.filtered_midline.copy()
        return None

    def get_raw_midline(self) -> Optional[np.ndarray]:
        """元のHLC3ミッドラインを取得する"""
        if self._result is not None:
            return self._result.raw_midline.copy()
        return None

    def get_ukf_values(self) -> Optional[np.ndarray]:
        """UKF値を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.ukf_values.copy()

    def get_ultimate_smooth_values(self) -> Optional[np.ndarray]:
        """アルティメットスムーザー後の値を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.ultimate_smooth_values.copy()

    def get_zero_lag_values(self) -> Optional[np.ndarray]:
        """ゼロラグEMA後の値を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.zero_lag_values.copy()

    def get_amplitude(self) -> Optional[np.ndarray]:
        """ヒルベルト変換振幅を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.amplitude.copy()

    def get_phase(self) -> Optional[np.ndarray]:
        """ヒルベルト変換位相を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.phase.copy()

    def get_str_values(self) -> Optional[np.ndarray]:
        """STR値を取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.str_values.copy()

    def get_filtering_mode(self) -> int:
        """フィルタリングモードを取得する"""
        return self.filtering_mode
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """Ultimate MAのトレンドシグナルを取得する"""
        if not self._cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.trend_signals.copy()
    
    def get_current_trend(self) -> str:
        """現在のトレンドを取得する"""
        if not self._cache:
            return 'range'
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.current_trend
    
    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得する"""
        if not self._cache:
            return 0
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return result.current_trend_value

    def get_filtering_stats(self) -> dict:
        """フィルタリング統計情報を取得する"""
        if not self._cache:
            return {}
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._cache.values()))
            
        return {
            'filtering_mode': result.filtering_mode,
            'midline_type': result.midline_type,
            'current_trend': result.current_trend,
            'current_trend_value': result.current_trend_value,
            'trend_signals_available': len(result.trend_signals) > 0 and np.any(result.trend_signals != 0)
        }

    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._cache = {}
        self._cache_keys = []
        if self.ultimate_ma is not None:
            self.ultimate_ma.reset()
        if self.ultimate_smoother is not None:
            self.ultimate_smoother.reset()
        if self.str_indicator is not None:
            self.str_indicator.reset()

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
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
            
            # 最小限のパラメータ情報
            data_sig = (length, first_val, last_val)
            param_sig = (f"{self.length}_{self.multiplier}_{self.ultimate_smoother_period}_"
                        f"{self.zero_lag_period}_{self.filtering_mode}_{self.midline_type}")
            
            # 超高速ハッシュ
            return f"{hash(data_sig)}_{hash(param_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.length}_{self.multiplier}_{self.midline_type}" 