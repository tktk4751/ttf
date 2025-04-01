#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange, vectorize

from .indicator import Indicator
from .hurst_exponent import calculate_rs, calculate_hurst_for_point
from .ehlers_dudi_dce import EhlersDuDiDCE
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZHurstExponentResult:
    """Zハースト指数の計算結果"""
    values: np.ndarray            # Zハースト指数値（0-1の範囲）
    rs_values: np.ndarray         # R/S統計量
    trend_strength: np.ndarray    # トレンド強度（0.5からの距離）
    er: np.ndarray                # サイクル効率比（CER）
    adaptive_windows: np.ndarray  # 適応的な分析ウィンドウ
    adaptive_min_lags: np.ndarray # 適応的な最小ラグ
    adaptive_max_lags: np.ndarray # 適応的な最大ラグ
    adaptive_steps: np.ndarray    # 適応的なステップ
    dc_values: np.ndarray         # ドミナントサイクル値
    adaptive_thresholds: np.ndarray  # 適応的なしきい値


@njit(fastmath=True)
def calculate_dynamic_window(er: np.ndarray, max_window: np.ndarray, min_window: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的な分析ウィンドウを計算する
    
    Args:
        er: 効率比の配列
        max_window: 最大ウィンドウサイズの配列
        min_window: 最小ウィンドウサイズの配列
    
    Returns:
        動的な分析ウィンドウの配列
    """
    size = len(er)
    dynamic_window = np.zeros(size, dtype=np.int32)  # 整数型として初期化
    
    for i in range(size):
        if np.isnan(er[i]):
            dynamic_window[i] = int(max_window[i])  # 明示的に整数に変換
        else:
            # ERが高い（トレンドが強い）ほどウィンドウは短く、
            # ERが低い（トレンドが弱い）ほどウィンドウは長くなる
            dynamic_window[i] = int(min_window[i] + (1.0 - abs(er[i])) * (max_window[i] - min_window[i]))
    
    return dynamic_window


@njit(fastmath=True)
def calculate_dynamic_lag(er: np.ndarray, max_lag: np.ndarray, min_lag: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的なラグを計算する
    
    Args:
        er: 効率比の配列
        max_lag: 最大ラグの配列
        min_lag: 最小ラグの配列
    
    Returns:
        動的なラグの配列
    """
    size = len(er)
    dynamic_lag = np.zeros(size, dtype=np.int32)  # 整数型として初期化
    
    for i in range(size):
        if np.isnan(er[i]):
            dynamic_lag[i] = int(max_lag[i])  # 明示的に整数に変換
        else:
            # ERが高い（トレンドが強い）ほどラグは短く、
            # ERが低い（トレンドが弱い）ほどラグは長くなる
            dynamic_lag[i] = int(min_lag[i] + (1.0 - abs(er[i])) * (max_lag[i] - min_lag[i]))
    
    return dynamic_lag


@njit(fastmath=True)
def calculate_dynamic_threshold(er: np.ndarray, max_threshold: float, min_threshold: float) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する
    
    Args:
        er: 効率比の配列
        max_threshold: 最大しきい値
        min_threshold: 最小しきい値
    
    Returns:
        動的なしきい値の配列
    """
    size = len(er)
    dynamic_threshold = np.zeros(size)
    
    for i in range(size):
        if np.isnan(er[i]):
            dynamic_threshold[i] = min_threshold
        else:
            # ERが高い（トレンドが強い）ほどしきい値は高く、
            # ERが低い（トレンドが弱い）ほどしきい値は低くなる
            dynamic_threshold[i] = min_threshold + abs(er[i]) * (max_threshold - min_threshold)
    
    return dynamic_threshold


@njit(fastmath=True, parallel=True)
def calculate_z_hurst_exponent(
    data: np.ndarray,
    er: np.ndarray,
    adaptive_windows: np.ndarray,
    adaptive_min_lags: np.ndarray,
    adaptive_max_lags: np.ndarray,
    adaptive_steps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    動的パラメータを使用してZハースト指数を計算する
    
    Args:
        data: 価格データの配列
        er: 効率比の配列
        adaptive_windows: 適応的な分析ウィンドウの配列
        adaptive_min_lags: 適応的な最小ラグの配列
        adaptive_max_lags: 適応的な最大ラグの配列
        adaptive_steps: 適応的なステップの配列
    
    Returns:
        (Zハースト指数の配列, R/S統計量の配列, トレンド強度の配列)のタプル
    """
    size = len(data)
    hurst_values = np.zeros(size)
    rs_values = np.zeros(size)
    trend_strength = np.zeros(size)
    
    # 初期部分はNaNで初期化
    max_window = int(np.max(adaptive_windows))  # 明示的に整数に変換
    hurst_values[:max_window] = np.nan
    rs_values[:max_window] = np.nan
    trend_strength[:max_window] = np.nan
    
    for i in prange(max_window, size):
        # その時点での適応的なパラメータを取得
        window = int(adaptive_windows[i])
        min_lag = int(adaptive_min_lags[i])
        max_lag = int(adaptive_max_lags[i])
        step = int(adaptive_steps[i])
        
        # パラメータの妥当性チェック
        if window < min_lag * 2:
            window = min_lag * 2  # ウィンドウは最小ラグの2倍以上
        
        if max_lag > window:
            max_lag = window  # 最大ラグはウィンドウを超えない
        
        if min_lag >= max_lag:
            min_lag = max_lag // 2  # 最小ラグは最大ラグの半分以下
        
        if step > (max_lag - min_lag) // 2:
            step = max(1, (max_lag - min_lag) // 2)  # ステップは範囲の半分以下
        
        # データ長チェック
        if i < window:
            continue  # データが不足している場合はスキップ
        
        # ウィンドウデータを取得
        window_data = data[i-window:i]
        
        # ハースト指数を計算
        h = calculate_hurst_for_point(window_data, min_lag, max_lag, step)
        
        hurst_values[i] = h
        rs_values[i] = calculate_rs(window_data, min_lag)
        
        # トレンド強度：0.5からの距離（絶対値）
        if not np.isnan(h):
            trend_strength[i] = abs(h - 0.5)
    
    return hurst_values, rs_values, trend_strength


class ZHurstExponent(Indicator):
    """
    Zハースト指数インジケーター
    
    ハースト指数を動的に適応させた拡張版。サイクル効率比（CER）とドミナントサイクルを用いて
    ウィンドウサイズやラグ期間などのパラメータを市場状況に応じて動的に調整します。
    
    特徴:
    - ドミナントサイクルを使用して分析ウィンドウを動的に調整
    - サイクル効率比を使用してラグパラメータを適応的に変更
    - 各時点で最適なパラメータを使用してハースト指数を計算
    - 適応的なしきい値による正確なシグナル生成
    """
    
    def __init__(
        self,
        # 分析ウィンドウパラメータ
        max_window_dc_cycle_part: float = 0.75,
        max_window_dc_max_cycle: int = 144,
        max_window_dc_min_cycle: int = 8,
        max_window_dc_max_output: int = 200,
        max_window_dc_min_output: int = 50,
        
        min_window_dc_cycle_part: float = 0.5,
        min_window_dc_max_cycle: int = 55,
        min_window_dc_min_cycle: int = 5,
        min_window_dc_max_output: int = 50,
        min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,  # 最大ラグはウィンドウの何%か
        min_lag_ratio: float = 0.1,  # 最小ラグはウィンドウの何%か
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'dudi_dce',
        lp_period: int = 10,
        hp_period: int = 48,
        cycle_part: float = 0.5,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.6,
        min_threshold: float = 0.5,
        
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            max_window_dc_cycle_part: 最大ウィンドウ用DCのサイクル部分（デフォルト: 0.75）
            max_window_dc_max_cycle: 最大ウィンドウ用DCの最大サイクル（デフォルト: 144）
            max_window_dc_min_cycle: 最大ウィンドウ用DCの最小サイクル（デフォルト: 8）
            max_window_dc_max_output: 最大ウィンドウ用DCの最大出力（デフォルト: 200）
            max_window_dc_min_output: 最大ウィンドウ用DCの最小出力（デフォルト: 50）
            
            min_window_dc_cycle_part: 最小ウィンドウ用DCのサイクル部分（デフォルト: 0.5）
            min_window_dc_max_cycle: 最小ウィンドウ用DCの最大サイクル（デフォルト: 55）
            min_window_dc_min_cycle: 最小ウィンドウ用DCの最小サイクル（デフォルト: 5）
            min_window_dc_max_output: 最小ウィンドウ用DCの最大出力（デフォルト: 50）
            min_window_dc_min_output: 最小ウィンドウ用DCの最小出力（デフォルト: 20）
            
            max_lag_ratio: 最大ラグとウィンドウの比率（デフォルト: 0.5）
            min_lag_ratio: 最小ラグとウィンドウの比率（デフォルト: 0.1）
            
            max_step: 最大ステップ（デフォルト: 10）
            min_step: 最小ステップ（デフォルト: 2）
            
            cycle_detector_type: サイクル検出器タイプ（デフォルト: 'dudi_dce'）
            lp_period: ローパスフィルターの期間（デフォルト: 10）
            hp_period: ハイパスフィルターの期間（デフォルト: 48）
            cycle_part: サイクル部分（デフォルト: 0.5）
            
            max_threshold: 最大しきい値（デフォルト: 0.7）
            min_threshold: 最小しきい値（デフォルト: 0.55）
            
            src_type: ソースタイプ（デフォルト: 'close'）
        """
        super().__init__(
            f"ZHurstExponent({max_window_dc_max_output}-{min_window_dc_min_output}, {cycle_detector_type})"
        )
        
        # ウィンドウパラメータの保存
        self.max_window_dc_cycle_part = max_window_dc_cycle_part
        self.max_window_dc_max_cycle = max_window_dc_max_cycle
        self.max_window_dc_min_cycle = max_window_dc_min_cycle
        self.max_window_dc_max_output = max_window_dc_max_output
        self.max_window_dc_min_output = max_window_dc_min_output
        
        self.min_window_dc_cycle_part = min_window_dc_cycle_part
        self.min_window_dc_max_cycle = min_window_dc_max_cycle
        self.min_window_dc_min_cycle = min_window_dc_min_cycle
        self.min_window_dc_max_output = min_window_dc_max_output
        self.min_window_dc_min_output = min_window_dc_min_output
        
        # ラグとステップのパラメータ
        self.max_lag_ratio = max_lag_ratio
        self.min_lag_ratio = min_lag_ratio
        self.max_step = max_step
        self.min_step = min_step
        
        # しきい値パラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        
        # ソースタイプ
        self.src_type = src_type
        
        # ドミナントサイクル検出器の初期化
        # 最大ウィンドウ用
        self.max_window_dc = EhlersDuDiDCE(
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=max_window_dc_cycle_part,
            max_output=max_window_dc_max_output,
            min_output=max_window_dc_min_output,
            src_type=src_type
        )
        
        # 最小ウィンドウ用
        self.min_window_dc = EhlersDuDiDCE(
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=min_window_dc_cycle_part,
            max_output=min_window_dc_max_output,
            min_output=min_window_dc_min_output,
            src_type=src_type
        )
        
        # サイクル効率比の初期化
        self.cycle_efficiency_ratio = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type
        )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            if data.ndim == 2 and data.shape[1] >= 4:
                data_hash = hash(tuple(data[:, 3]))  # close column
            else:
                data_hash = hash(tuple(data.flatten()))
        
        # パラメータ値を含める
        param_str = (
            f"{self.max_window_dc_cycle_part}_{self.max_window_dc_max_cycle}_{self.max_window_dc_min_cycle}_"
            f"{self.max_window_dc_max_output}_{self.max_window_dc_min_output}_"
            f"{self.min_window_dc_cycle_part}_{self.min_window_dc_max_cycle}_{self.min_window_dc_min_cycle}_"
            f"{self.min_window_dc_max_output}_{self.min_window_dc_min_output}_"
            f"{self.max_lag_ratio}_{self.min_lag_ratio}_{self.max_step}_{self.min_step}_"
            f"{self.max_threshold}_{self.min_threshold}_{self.src_type}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zハースト指数を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            Zハースト指数値の配列（0-1の範囲）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    raise ValueError("DataFrameには'close'カラムが必要です")
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    prices = data[:, 3]  # close column
                elif data.ndim == 1:
                    prices = data
                else:
                    raise ValueError("サポートされていないデータ形式です")
            
            # データ長の検証
            data_length = len(prices)
            
            # ドミナントサイクルの計算
            max_window_cycles = self.max_window_dc.calculate(data)
            min_window_cycles = self.min_window_dc.calculate(data)
            
            # サイクル効率比の計算
            er = self.cycle_efficiency_ratio.calculate(data)
            
            # 動的パラメータの計算
            adaptive_windows = calculate_dynamic_window(er, max_window_cycles, min_window_cycles)
            
            # 動的な最大ラグと最小ラグの計算 - 明示的に整数型に変換
            adaptive_max_lags = np.zeros(data_length, dtype=np.int32)
            adaptive_min_lags = np.zeros(data_length, dtype=np.int32)
            
            for i in range(data_length):
                adaptive_max_lags[i] = int(adaptive_windows[i] * self.max_lag_ratio)
                adaptive_min_lags[i] = int(adaptive_windows[i] * self.min_lag_ratio)
            
            # 動的なステップの計算（ERが高いほど小さく、低いほど大きく）
            adaptive_steps = np.zeros(data_length, dtype=np.int32)  # 整数型として初期化
            for i in range(data_length):
                if np.isnan(er[i]):
                    adaptive_steps[i] = self.min_step
                else:
                    adaptive_steps[i] = int(self.min_step + (1 - abs(er[i])) * (self.max_step - self.min_step))
            
            # 動的なしきい値の計算
            adaptive_thresholds = calculate_dynamic_threshold(er, self.max_threshold, self.min_threshold)
            
            # Zハースト指数の計算
            hurst_values, rs_values, trend_strength = calculate_z_hurst_exponent(
                prices,
                er,
                adaptive_windows,
                adaptive_min_lags,
                adaptive_max_lags,
                adaptive_steps
            )
            
            # 結果を保存
            self._result = ZHurstExponentResult(
                values=hurst_values,
                rs_values=rs_values,
                trend_strength=trend_strength,
                er=er,
                adaptive_windows=adaptive_windows,
                adaptive_min_lags=adaptive_min_lags,
                adaptive_max_lags=adaptive_max_lags,
                adaptive_steps=adaptive_steps,
                dc_values=max_window_cycles,
                adaptive_thresholds=adaptive_thresholds
            )
            
            self._values = hurst_values  # 基底クラスの要件を満たすため
            
            return hurst_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Zハースト指数計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_rs_values(self) -> np.ndarray:
        """
        R/S統計量を取得する
        
        Returns:
            R/S統計量の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.rs_values
    
    def get_trend_strength(self) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Returns:
            トレンド強度の配列 (0-0.5の範囲、大きいほど強いトレンド性)
        """
        if self._result is None:
            return np.array([])
        return self._result.trend_strength
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比を取得する
        
        Returns:
            サイクル効率比の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_adaptive_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        適応的なパラメータを取得する
        
        Returns:
            (適応的なウィンドウ, 適応的な最小ラグ, 適応的な最大ラグ, 適応的なステップ)のタプル
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty, empty, empty
        return (
            self._result.adaptive_windows,
            self._result.adaptive_min_lags,
            self._result.adaptive_max_lags,
            self._result.adaptive_steps
        )
    
    def get_trend_type(self) -> np.ndarray:
        """
        トレンドタイプを取得する
        
        Returns:
            トレンドタイプの配列:
            1: 持続的トレンド (H > 0.5)
            0: ランダムウォーク (H = 0.5)
            -1: 反持続的トレンド (H < 0.5)
        """
        if self._result is None:
            return np.array([])
        
        values = self._result.values
        trend_type = np.zeros_like(values)
        
        # トレンドタイプを判定
        # 0.5より大きい: 持続的トレンド
        # 0.5未満: 反持続的トレンド
        # 誤差を考慮して0.5±0.01の範囲はランダムウォークとする
        trend_type = np.where(np.isnan(values), np.nan, 
                     np.where(values > 0.51, 1, 
                     np.where(values < 0.49, -1, 0)))
        
        return trend_type
    
    def get_adaptive_thresholds(self) -> np.ndarray:
        """
        適応的なしきい値を取得する
        
        Returns:
            適応的なしきい値の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.adaptive_thresholds
    
    def get_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        適応的なしきい値を使用してトレンドシグナルを取得する
        
        Returns:
            (上昇トレンドシグナル, 下降トレンドシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        
        # 前回値との差分を計算（傾き）
        prev_values = np.roll(self._values, 1)
        prev_values[0] = self._values[0]
        
        slope = self._values - prev_values
        
        # 適応的なしきい値を使用
        thresholds = self._result.adaptive_thresholds
        
        # ハースト指数が適応的な閾値を超え、傾きが正なら上昇トレンド
        uptrend = np.where(
            (self._values > thresholds) & (slope > 0),
            1, 0
        )
        
        # ハースト指数が適応的な閾値を超え、傾きが負なら下降トレンド
        downtrend = np.where(
            (self._values > thresholds) & (slope < 0),
            1, 0
        )
        
        return uptrend, downtrend
    
    def get_mean_reversion_signals(self) -> np.ndarray:
        """
        適応的なしきい値を使用して平均回帰シグナルを取得する
        
        Returns:
            平均回帰シグナル（値が閾値未満の場合は1、そうでない場合は0）
        """
        if self._result is None:
            return np.array([])
        
        # 適応的なしきい値の逆数を平均回帰しきい値として使用（例：0.7 → 0.3）
        mean_reversion_thresholds = 1.0 - self._result.adaptive_thresholds
        
        # ハースト指数が平均回帰しきい値未満なら平均回帰傾向
        mean_reversion = np.where(self._values < mean_reversion_thresholds, 1, 0)
        
        return mean_reversion
    
    def get_dominant_cycle(self) -> np.ndarray:
        """
        ドミナントサイクル値を取得する
        
        Returns:
            ドミナントサイクル値の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.max_window_dc.reset()
        self.min_window_dc.reset()
        self.cycle_efficiency_ratio.reset() 