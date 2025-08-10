#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit
from dataclasses import dataclass

from .indicator import Indicator
from .cycle.ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource
from .alma import ALMA


@dataclass
class DubucHurstResult:
    """Dubucハースト指数の計算結果"""
    values: np.ndarray          # ハースト指数の値
    raw_values: np.ndarray      # 生のハースト指数の値（スムージング前）
    dynamic_threshold: np.ndarray  # 動的しきい値
    trend_state: np.ndarray     # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)
    cycle_periods: np.ndarray   # 計算に使用された動的サイクル期間
    log_x_values: np.ndarray    # ログスケールのX値（デバッグ用）
    log_y_values: np.ndarray    # ログスケールのY値（デバッグ用）


@njit
def calculate_area_for_epsilon(high: np.ndarray, low: np.ndarray, epsilon: int, length: int) -> float:
    """
    指定されたイプシロン値での面積を計算する最適化関数
    
    Args:
        high: 高値配列
        low: 安値配列
        epsilon: イプシロン値
        length: 分析ウィンドウの長さ
    
    Returns:
        float: 正規化された面積
    """
    n = len(high)
    if n < 2 * epsilon + 1 or n < length:
        return np.nan
    
    # 近傍のtopとbottomを計算
    neighborhood_size = 2 * epsilon + 1
    area = 0.0
    count = 0
    base_count = 0
    
    # lengthの範囲で面積を計算
    start_idx = max(0, n - length)
    for i in range(start_idx, n):
        # 近傍の範囲を計算
        start = max(0, i - epsilon)
        end = min(n, i + epsilon + 1)
        
        if end - start > 0:
            # 近傍での最高値と最安値
            top = np.max(high[start:end])
            bottom = np.min(low[start:end])
            
            if not np.isnan(top) and not np.isnan(bottom):
                area += (top - bottom)
                count += 1
                
                # 最初のイプシロン（最小）でのベースカウントを記録
                if epsilon == 1 and base_count == 0:
                    base_count = 1
    
    # 正規化
    if count > 0:
        # ベースカウントが設定されていない場合は現在のカウントを使用
        if base_count == 0:
            base_count = count
        return area * base_count / count
    
    return np.nan


@njit
def calculate_hurst_for_window(high: np.ndarray, low: np.ndarray, length: int, samples: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    指定されたウィンドウでのDubucハースト指数を計算する最適化関数
    
    Args:
        high: 高値配列
        low: 安値配列
        length: 分析ウィンドウの長さ
        samples: サンプル数
    
    Returns:
        Tuple[float, np.ndarray, np.ndarray]: (ハースト指数, ログX値, ログY値)
    """
    n = len(high)
    if n < length or length < 2 or samples < 2:
        return np.nan, np.full(samples, np.nan), np.full(samples, np.nan)
    
    # X（ログ1/epsilon）とY（ログ（面積/epsilon^2））の配列
    log_x = np.zeros(samples)
    log_y = np.zeros(samples)
    
    # 最大と最小のイプシロン
    max_iep = np.log(length)
    min_iep = np.log(1)  # イプシロン=1から開始
    
    base_count = 0
    
    for s in range(samples):
        # イプシロンを対数スケールで計算
        if samples == 1:
            iep = min_iep
        else:
            iep = min_iep + (max_iep - min_iep) * s / (samples - 1)
        
        epsilon = max(1, int(np.exp(iep)))
        
        # ウィンドウの最後の部分を取得
        window_high = high[-length:]
        window_low = low[-length:]
        
        # 面積を計算
        area = calculate_area_for_epsilon(window_high, window_low, epsilon, length)
        
        if not np.isnan(area) and area > 0:
            # ログX値：log(1/epsilon)
            log_x[s] = np.log(1.0 / epsilon)
            
            # ログY値：log(area/epsilon^2)
            log_y[s] = np.log(area / (epsilon * epsilon))
        else:
            log_x[s] = np.nan
            log_y[s] = np.nan
    
    # 有効なデータポイントのフィルタリング
    valid_mask = ~(np.isnan(log_x) | np.isnan(log_y))
    
    if np.sum(valid_mask) < 2:
        return np.nan, log_x, log_y
    
    valid_x = log_x[valid_mask]
    valid_y = log_y[valid_mask]
    
    # 線形回帰でスロープを計算（最小二乗法）
    n_valid = len(valid_x)
    sum_x = np.sum(valid_x)
    sum_y = np.sum(valid_y)
    sum_xy = np.sum(valid_x * valid_y)
    sum_x2 = np.sum(valid_x * valid_x)
    
    # スロープ計算
    denominator = n_valid * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return np.nan, log_x, log_y
    
    slope = (n_valid * sum_xy - sum_x * sum_y) / denominator
    
    # ハースト指数：スロープ - 1
    hurst = slope - 1.0
    
    return hurst, log_x, log_y


@njit
def calculate_hurst_array(high: np.ndarray, low: np.ndarray, lengths: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    動的ウィンドウでハースト指数の配列を計算する
    
    Args:
        high: 高値配列
        low: 安値配列
        lengths: 各時点での分析ウィンドウサイズ配列
        samples: サンプル数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ハースト指数配列, ログX値配列, ログY値配列)
    """
    n = len(high)
    result = np.full(n, np.nan)
    log_x_array = np.full((n, samples), np.nan)
    log_y_array = np.full((n, samples), np.nan)
    
    # 最小必要データポイント
    min_points = 5
    
    for i in range(min_points, n):
        length = int(lengths[i])
        if length < min_points:
            length = min_points
            
        # 十分なデータがある場合のみ計算
        if i >= length - 1:
            # 計算に必要な部分配列
            h_slice = high[:i+1]
            l_slice = low[:i+1]
            
            hurst, log_x, log_y = calculate_hurst_for_window(h_slice, l_slice, length, samples)
            
            result[i] = hurst
            log_x_array[i, :] = log_x
            log_y_array[i, :] = log_y
    
    return result, log_x_array, log_y_array


@njit(fastmath=True)
def calculate_dynamic_threshold(
    hurst_values: np.ndarray,
    base_threshold: float = 0.5
) -> np.ndarray:
    """
    ハースト指数に基づいて動的なしきい値を計算する
    
    Args:
        hurst_values: ハースト指数の配列
        base_threshold: ベースとなるしきい値（デフォルトは0.5）
    
    Returns:
        動的なしきい値の配列
    """
    length = len(hurst_values)
    threshold = np.full(length, base_threshold, dtype=np.float64)
    
    return threshold


@njit(fastmath=True)
def calculate_trend_state(
    hurst_values: np.ndarray,
    dynamic_threshold: np.ndarray
) -> np.ndarray:
    """
    ハースト指数と動的しきい値に基づいてトレンド状態を計算する
    
    Args:
        hurst_values: ハースト指数値の配列
        dynamic_threshold: 動的しきい値の配列
    
    Returns:
        トレンド状態の配列 (1=トレンド、0=レンジ、NaN=不明)
    """
    length = len(hurst_values)
    trend_state = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(hurst_values[i]) or np.isnan(dynamic_threshold[i]):
            continue
        trend_state[i] = 1.0 if hurst_values[i] > dynamic_threshold[i] else 0.0
    
    return trend_state


class DubucHurstExponent(Indicator):
    """
    Dubucハースト指数（Dubuc Hurst Exponent）インディケーター
    
    Dubucの変動法を使用してハースト指数を計算します。
    ドミナントサイクルを使用して動的なウィンドウサイズでの計算が可能で、
    ALMAによるスムージングも提供します。
    
    ハースト指数：
    - H > 0.5: トレンド性（持続性）
    - H = 0.5: ランダムウォーク
    - H < 0.5: 平均回帰性（反転傾向）
    
    参考文献:
    Dubuc B, Quiniou JF, Roques-Carmes C, Tricot C. 
    Evaluating the fractal dimension of profiles. 
    Physical Review A. 1989;39(3):1500-1512.
    """
    
    def __init__(
        self,
        length: int = 100,
        samples: int = 5,
        detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 200,
        min_output: int = 50,
        src_type: str = 'hl2',  # 高値・安値を使用するのでhl2がデフォルト
        smooth_hurst: bool = True,
        hurst_alma_period: int = 3,
        hurst_alma_offset: float = 0.85,
        hurst_alma_sigma: float = 6,
        base_threshold: float = 0.5,
        use_dynamic_window: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            length: 分析ウィンドウの基本サイズ
            samples: スケールサンプル数（最低2、推奨3以上）
            detector_type: ドミナントサイクル検出器タイプ
            lp_period: ドミナントサイクル用ローパスフィルター期間
            hp_period: ドミナントサイクル用ハイパスフィルター期間
            cycle_part: ドミナントサイクル計算用サイクル部分
            max_cycle: ドミナントサイクル最大期間
            min_cycle: ドミナントサイクル最小期間
            max_output: ドミナントサイクル最大出力値
            min_output: ドミナントサイクル最小出力値
            src_type: 価格ソース ('hl2', 'hlc3', etc.)
            smooth_hurst: ハースト指数値にALMAスムージングを適用するかどうか
            hurst_alma_period: ALMAスムージングの期間
            hurst_alma_offset: ALMAスムージングのオフセット
            hurst_alma_sigma: ALMAスムージングのシグマ
            base_threshold: トレンド判定のベースしきい値
            use_dynamic_window: 動的ウィンドウを使用するかどうか
        """
        smooth_str = f"_smooth={'Y' if smooth_hurst else 'N'}" if smooth_hurst else ""
        indicator_name = f"DubucHurst(len={length},smp={samples},det={detector_type},src={src_type}{smooth_str})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.length = length
        self.samples = samples
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        self.base_threshold = base_threshold
        self.use_dynamic_window = use_dynamic_window
        
        # ALMAスムージングパラメータ
        self.smooth_hurst = smooth_hurst
        self.hurst_alma_period = hurst_alma_period
        self.hurst_alma_offset = hurst_alma_offset
        self.hurst_alma_sigma = hurst_alma_sigma
        
        # 入力検証
        if self.length < 2:
            raise ValueError("分析ウィンドウの長さは最低2である必要があります")
        if self.samples < 2:
            raise ValueError("サンプル数は最低2である必要があります")
        
        # PriceSourceの初期化
        self.price_source_extractor = PriceSource()
        
        # ドミナントサイクル検出器（動的ウィンドウ使用時）
        self.dc_detector = None
        if self.use_dynamic_window:
            self.dc_detector = EhlersUnifiedDC(
                detector_type=detector_type,
                cycle_part=cycle_part,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output,
                lp_period=lp_period,
                hp_period=hp_period
            )
        
        # ALMAスムーザーの初期化（有効な場合）
        self.hurst_alma_smoother = None
        if self.smooth_hurst:
            self.hurst_alma_smoother = ALMA(
                period=self.hurst_alma_period,
                offset=self.hurst_alma_offset,
                sigma=self.hurst_alma_sigma,
                src_type='close',  # 直接値を渡すので、ソースタイプは関係ない
                use_kalman_filter=False  # すでに計算が済んでいるのでカルマンは不要
            )
        
        # 結果キャッシュ
        self._values = None  # 生のハースト指数値
        self._smoothed_values = None  # スムージングされたハースト指数値
        self._data_hash = None
        self._cycle_periods = None  # 動的サイクル期間配列
        self._dynamic_threshold = None  # 動的しきい値
        self._trend_state = None  # トレンド状態
        self._result = None  # 計算結果オブジェクト
        self._log_x_values = None  # ログX値（デバッグ用）
        self._log_y_values = None  # ログY値（デバッグ用）
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # 必要なカラムを決定
        required_cols = set(['high', 'low'])
        
        if isinstance(data, pd.DataFrame):
            present_cols = [col for col in data.columns if col.lower() in required_cols]
            if not present_cols:
                # 必要なカラムがない場合、基本的な情報でハッシュ
                try:
                    shape_tuple = data.shape
                    first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row, last_row)
                    data_hash_val = hash(data_repr_tuple)
                except Exception:
                    data_hash_val = hash(str(data)) # フォールバック
            else:
                # 関連するカラムの値でハッシュ
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            # NumPy配列の場合、形状や一部の値でハッシュ (簡略化)
            try:
                 shape_tuple = data.shape
                 first_row = tuple(data[0]) if len(data) > 0 else ()
                 last_row = tuple(data[-1]) if len(data) > 0 else ()
                 mean_val = np.mean(data) if data.size > 0 else 0.0
                 data_repr_tuple = (shape_tuple, first_row, last_row, mean_val)
                 data_hash_val = hash(data_repr_tuple)
            except Exception:
                 data_hash_val = hash(data.tobytes()) # フォールバック
        else:
            data_hash_val = hash(str(data)) # その他の型

        # パラメータ文字列を作成
        param_str = (
            f"len={self.length}_smp={self.samples}_det={self.detector_type}_"
            f"lp={self.lp_period}_hp={self.hp_period}_part={self.cycle_part}_"
            f"maxC={self.max_cycle}_minC={self.min_cycle}_maxO={self.max_output}_"
            f"minO={self.min_output}_src={self.src_type}_smooth={self.smooth_hurst}_"
            f"dyn={self.use_dynamic_window}_{self.hurst_alma_period}_{self.hurst_alma_offset}_{self.hurst_alma_sigma}"
        )
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dubucハースト指数を計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: ハースト指数の値（スムージングが有効な場合はスムージングされた値）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._smoothed_values if self.smooth_hurst else self._values
            
            # データフレームに変換
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # 必要なカラムを確認
            high = df['high'].values
            low = df['low'].values
            
            # 動的ウィンドウサイズの計算
            if self.use_dynamic_window and self.dc_detector is not None:
                # ドミナントサイクルの計算
                dc_values = self.dc_detector.calculate(data)
                # サイクル期間の制限とキャッシュ
                self._cycle_periods = np.clip(dc_values, self.min_output, self.max_output)
                lengths = self._cycle_periods.copy()
            else:
                # 固定ウィンドウサイズを使用
                data_len = len(high)
                lengths = np.full(data_len, self.length)
                self._cycle_periods = lengths.copy()
            
            # Numba最適化関数でハースト指数計算
            hurst_values, log_x_array, log_y_array = calculate_hurst_array(high, low, lengths, self.samples)
            
            # ログ値の保存（デバッグ用）
            self._log_x_values = log_x_array
            self._log_y_values = log_y_array
            
            # ALMAによるスムージング（有効な場合）
            smoothed_hurst_values = hurst_values.copy()  # デフォルトはスムージングなし
            if self.smooth_hurst:
                try:
                    if self.hurst_alma_smoother is None:
                        self.hurst_alma_smoother = ALMA(
                            period=self.hurst_alma_period,
                            offset=self.hurst_alma_offset,
                            sigma=self.hurst_alma_sigma,
                            src_type='close',
                            use_kalman_filter=False
                        )
                        
                    smoothed_values = self.hurst_alma_smoother.calculate(hurst_values)
                    
                    # NaNの処理（最初の数ポイントはNaNになるため、元の値で埋める）
                    nan_indices = np.isnan(smoothed_values)
                    smoothed_hurst_values = smoothed_values.copy()
                    smoothed_hurst_values[nan_indices] = hurst_values[nan_indices]
                except Exception as e:
                    self.logger.error(f"ハースト指数値のスムージング中にエラー: {str(e)}。生の値を使用します。")
                    smoothed_hurst_values = hurst_values.copy()  # エラー時は元の値を使用
            
            # 動的しきい値の計算
            final_values = smoothed_hurst_values if self.smooth_hurst else hurst_values
            self._dynamic_threshold = calculate_dynamic_threshold(
                final_values, 
                self.base_threshold
            )
            
            # トレンド状態の計算
            self._trend_state = calculate_trend_state(final_values, self._dynamic_threshold)

            # 結果を保存してキャッシュ
            self._values = hurst_values
            self._smoothed_values = smoothed_hurst_values
            self._data_hash = data_hash
            
            # 結果オブジェクトを作成
            self._result = DubucHurstResult(
                values=final_values,
                raw_values=hurst_values,
                dynamic_threshold=self._dynamic_threshold,
                trend_state=self._trend_state,
                cycle_periods=self._cycle_periods,
                log_x_values=self._log_x_values,
                log_y_values=self._log_y_values
            )

            # スムージングが有効な場合はスムージングされた値を返す
            return final_values
            
        except Exception as e:
            import traceback
            error_msg = f"Dubucハースト指数計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            
            # エラー時はNaN配列
            self._values = np.full(data_len, np.nan)
            self._smoothed_values = np.full(data_len, np.nan)
            self._dynamic_threshold = np.full(data_len, np.nan)
            self._trend_state = np.full(data_len, np.nan)
            self._cycle_periods = np.full(data_len, np.nan)
            self._log_x_values = np.full((data_len, self.samples), np.nan)
            self._log_y_values = np.full((data_len, self.samples), np.nan)
            self._data_hash = None # エラー時はキャッシュクリア
            
            # エラー時の結果オブジェクト
            self._result = DubucHurstResult(
                values=np.full(data_len, np.nan),
                raw_values=np.full(data_len, np.nan),
                dynamic_threshold=np.full(data_len, np.nan),
                trend_state=np.full(data_len, np.nan),
                cycle_periods=np.full(data_len, np.nan),
                log_x_values=np.full((data_len, self.samples), np.nan),
                log_y_values=np.full((data_len, self.samples), np.nan)
            )
            
            return self._smoothed_values if self.smooth_hurst else self._values
    
    def get_result(self) -> Optional[DubucHurstResult]:
        """
        計算結果全体を取得する
        
        Returns:
            DubucHurstResult: 計算結果オブジェクト
        """
        return self._result
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: 動的しきい値の配列
        """
        return self._dynamic_threshold if self._dynamic_threshold is not None else np.array([])
    
    def get_trend_state(self) -> np.ndarray:
        """
        トレンド状態を取得する (1=トレンド、0=レンジ、NaN=不明)
        
        Returns:
            np.ndarray: トレンド状態の配列
        """
        return self._trend_state if self._trend_state is not None else np.array([])
    
    def get_cycle_periods(self) -> np.ndarray:
        """
        動的サイクル期間を取得する
        
        Returns:
            np.ndarray: サイクル期間の配列
        """
        return self._cycle_periods if self._cycle_periods is not None else np.array([])
    
    def get_log_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ログスケールのX,Y値を取得する（デバッグ用）
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ログX値配列, ログY値配列)
        """
        log_x = self._log_x_values if self._log_x_values is not None else np.array([])
        log_y = self._log_y_values if self._log_y_values is not None else np.array([])
        return log_x, log_y
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._smoothed_values = None
        self._cycle_periods = None
        self._dynamic_threshold = None
        self._trend_state = None
        self._log_x_values = None
        self._log_y_values = None
        self._result = None
        self._data_hash = None
        if self.hurst_alma_smoother and hasattr(self.hurst_alma_smoother, 'reset'):
            self.hurst_alma_smoother.reset()
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        smooth_str = f", smooth={self.smooth_hurst}" if self.smooth_hurst else ""
        dyn_str = f", dynamic={self.use_dynamic_window}" if self.use_dynamic_window else ""
        return f"DubucHurst(len={self.length}, smp={self.samples}, src={self.src_type}{smooth_str}{dyn_str})" 