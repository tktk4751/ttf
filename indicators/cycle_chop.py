#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit
from dataclasses import dataclass

from .indicator import Indicator
from .cycle.ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource
from .alma import ALMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class CycleChoppinessResult:
    """サイクルチョピネスの計算結果"""
    values: np.ndarray          # サイクルチョピネスの値
    raw_values: np.ndarray      # 生のサイクルチョピネスの値（スムージング前）
    dynamic_threshold: np.ndarray  # 動的しきい値
    chop_state: np.ndarray      # チョピネス状態 (1=高チョピネス、0=低チョピネス、NaN=不明)
    cycle_periods: np.ndarray   # 計算に使用された動的サイクル期間
    er_values: np.ndarray       # サイクル効率比（CER）の値


@njit
def calculate_choppiness_for_period(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """
    指定された期間でのチョピネス値を計算する最適化関数
    
    Args:
        high: 高値配列
        low: 安値配列
        close: 終値配列
        period: 計算期間
    
    Returns:
        float: チョピネス値
    """
    # 配列長チェック
    n = len(high)
    if n < period + 1:
        return np.nan
        
    # True Range計算
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # 初日はH-L
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)
    
    # 期間内の最大値と最小値
    highest_high = high[-period:]
    lowest_low = low[-period:]
    max_high = np.max(highest_high)
    min_low = np.min(lowest_low)
    
    # ATRの合計
    tr_sum = np.sum(tr[-period:])
    
    # チョピネス計算
    range_val = max_high - min_low
    if range_val == 0:
        return 1.0  # 完全なチョッピー状態
        
    chop = 1- (np.log10(tr_sum / range_val) / np.log10(period))
    return chop


@njit
def calculate_choppiness_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                              periods: np.ndarray) -> np.ndarray:
    """
    動的期間でチョピネス値の配列を計算する
    
    Args:
        high: 高値配列
        low: 安値配列  
        close: 終値配列
        periods: 各時点での計算期間配列
    
    Returns:
        np.ndarray: チョピネス値の配列
    """
    n = len(high)
    result = np.full(n, np.nan)
    
    # 最小必要データポイント
    min_points = 5
    
    for i in range(min_points, n):
        period = int(periods[i])
        if period < min_points:
            period = min_points
            
        # 十分なデータがある場合のみ計算
        if i >= period:
            # 計算に必要な部分配列
            h_slice = high[i-period:i+1]
            l_slice = low[i-period:i+1]
            c_slice = close[i-period:i+1]
            
            result[i] = calculate_choppiness_for_period(h_slice, l_slice, c_slice, period)
    
    return result


@njit(fastmath=True)
def calculate_dynamic_threshold(
    er_values: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する
    
    Args:
        er_values: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        動的なしきい値の配列
    """
    length = len(er_values)
    threshold = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er_values[i]):
            threshold[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er_values[i])
        # ERが高いほど（トレンドが強いほど）しきい値は高く
        # ERが低いほど（レンジ相場ほど）しきい値は低く
        threshold[i] = min_threshold + er_abs * (max_threshold - min_threshold)
    
    return threshold


@njit(fastmath=True)
def calculate_chop_state(
    chop_values: np.ndarray,
    dynamic_threshold: np.ndarray
) -> np.ndarray:
    """
    サイクルチョピネス値と動的しきい値に基づいてチョピネス状態を計算する
    
    Args:
        chop_values: サイクルチョピネス値の配列
        dynamic_threshold: 動的しきい値の配列
    
    Returns:
        チョピネス状態の配列 (1=高チョピネス、0=低チョピネス、NaN=不明)
    """
    length = len(chop_values)
    chop_state = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(chop_values[i]) or np.isnan(dynamic_threshold[i]):
            continue
        chop_state[i] = 1.0 if chop_values[i] >= dynamic_threshold[i] else 0.0
    
    return chop_state


class CycleChoppiness(Indicator):
    """
    サイクルチョピネス（Cycle Choppiness）インディケーター
    
    ドミナントサイクルを使用して動的なウィンドウサイズでのチョピネスインデックスを計算します。
    計算に使用する価格ソースを選択可能で、ALMAによるスムージングも可能です。
    動的しきい値を用いて高/低チョピネス状態を判定します。
    サイクル効率比（CER）に基づいて動的しきい値を計算します。
    """
    
    def __init__(
        self,
        detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        smooth_chop: bool = True,
        chop_alma_period: int = 3,
        chop_alma_offset: float = 0.85,
        chop_alma_sigma: float = 6,
        max_threshold: float = 0.6,  # 動的しきい値の最大値
        min_threshold: float = 0.4,   # 動的しきい値の最小値
        # CER固有パラメータ
        er_detector_type: Optional[str] = None,
        er_cycle_part: Optional[float] = None,
        er_max_cycle: Optional[int] = None,
        er_min_cycle: Optional[int] = None,
        er_max_output: Optional[int] = None,
        er_min_output: Optional[int] = None,
        er_src_type: Optional[str] = None,
        er_lp_period: Optional[int] = None,
        er_hp_period: Optional[int] = None,
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        smooth_er: bool = True,
        er_alma_period: int = 3,
        er_alma_offset: float = 0.85,
        er_alma_sigma: float = 6,
        self_adaptive: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
            lp_period: ドミナントサイクル用ローパスフィルター期間
            hp_period: ドミナントサイクル用ハイパスフィルター期間
            cycle_part: ドミナントサイクル計算用サイクル部分
            max_cycle: ドミナントサイクル最大期間
            min_cycle: ドミナントサイクル最小期間
            max_output: ドミナントサイクル最大出力値
            min_output: ドミナントサイクル最小出力値
            src_type: 価格ソース ('close', 'hlc3', etc.)
            smooth_chop: チョピネス値にALMAスムージングを適用するかどうか
            chop_alma_period: ALMAスムージングの期間
            chop_alma_offset: ALMAスムージングのオフセット
            chop_alma_sigma: ALMAスムージングのシグマ
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
            er_detector_type: CER固有のドミナントサイクル検出器タイプ
            er_cycle_part: CER固有のサイクル部分
            er_max_cycle: CER固有の最大サイクル期間
            er_min_cycle: CER固有の最小サイクル期間
            er_max_output: CER固有の最大出力値
            er_min_output: CER固有の最小出力値
            er_src_type: CER固有の価格ソース
            er_lp_period: CER固有のローパスフィルター期間
            er_hp_period: CER固有のハイパスフィルター期間
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            smooth_er: 効率比にALMAスムージングを適用するかどうか
            er_alma_period: ALMAスムージングの期間
            er_alma_offset: ALMAスムージングのオフセット
            er_alma_sigma: ALMAスムージングのシグマ
            self_adaptive: セルフアダプティブモードを有効にするかどうか
        """
        smooth_str = f"_smooth={'Y' if smooth_chop else 'N'}" if smooth_chop else ""
        indicator_name = f"CycleChop(det={detector_type},part={cycle_part},src={src_type}{smooth_str})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        
        # ALMAスムージングパラメータ
        self.smooth_chop = smooth_chop
        self.chop_alma_period = chop_alma_period
        self.chop_alma_offset = chop_alma_offset
        self.chop_alma_sigma = chop_alma_sigma
        
        # 動的しきい値パラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

        # CER固有パラメータ
        self.er_detector_type = er_detector_type if er_detector_type is not None else detector_type
        self.er_cycle_part = er_cycle_part if er_cycle_part is not None else cycle_part
        self.er_max_cycle = er_max_cycle if er_max_cycle is not None else max_cycle
        self.er_min_cycle = er_min_cycle if er_min_cycle is not None else min_cycle
        self.er_max_output = er_max_output if er_max_output is not None else max_output
        self.er_min_output = er_min_output if er_min_output is not None else min_output
        self.er_src_type = er_src_type if er_src_type is not None else src_type
        self.er_lp_period = er_lp_period if er_lp_period is not None else lp_period
        self.er_hp_period = er_hp_period if er_hp_period is not None else hp_period
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.smooth_er = smooth_er
        self.er_alma_period = er_alma_period
        self.er_alma_offset = er_alma_offset
        self.er_alma_sigma = er_alma_sigma
        self.self_adaptive = self_adaptive
        
        # PriceSourceの初期化
        self.price_source_extractor = PriceSource()
        
        # ドミナントサイクル検出器
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
        self.chop_alma_smoother = None
        if self.smooth_chop:
            self.chop_alma_smoother = ALMA(
                period=self.chop_alma_period,
                offset=self.chop_alma_offset,
                sigma=self.chop_alma_sigma,
                src_type='close',  # 直接値を渡すので、ソースタイプは関係ない
                use_kalman_filter=False  # すでに計算が済んでいるのでカルマンは不要
            )
        
        # サイクル効率比（CER）インジケーターの初期化
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=self.er_detector_type,
            cycle_part=self.er_cycle_part,
            lp_period=self.er_lp_period,
            hp_period=self.er_hp_period,
            max_cycle=self.er_max_cycle,
            min_cycle=self.er_min_cycle,
            max_output=self.er_max_output,
            min_output=self.er_min_output,
            src_type=self.er_src_type,
            use_kalman_filter=self.use_kalman_filter,
            kalman_measurement_noise=self.kalman_measurement_noise,
            kalman_process_noise=self.kalman_process_noise,
            kalman_n_states=self.kalman_n_states,
            smooth_er=self.smooth_er,
            er_alma_period=self.er_alma_period,
            er_alma_offset=self.er_alma_offset,
            er_alma_sigma=self.er_alma_sigma,
            self_adaptive=self.self_adaptive
        )
        
        # 結果キャッシュ
        self._values = None  # 生のチョピネス値
        self._smoothed_values = None  # スムージングされたチョピネス値
        self._data_hash = None
        self._cycle_periods = None  # 動的サイクル期間配列
        self._dynamic_threshold = None  # 動的しきい値
        self._chop_state = None  # チョピネス状態
        self._result = None  # 計算結果オブジェクト
        self._er_values = None  # 効率比の値
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # 必要なカラムを決定
        required_cols = set(['high', 'low', 'close'])
        
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
            f"det={self.detector_type}_lp={self.lp_period}_hp={self.hp_period}_"
            f"part={self.cycle_part}_maxC={self.max_cycle}_minC={self.min_cycle}_"
            f"maxO={self.max_output}_minO={self.min_output}_src={self.src_type}_"
            f"smooth={self.smooth_chop}_{self.chop_alma_period}_{self.chop_alma_offset}_{self.chop_alma_sigma}"
        )
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクルチョピネスを計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: サイクルチョピネスの値（スムージングが有効な場合はスムージングされた値）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._smoothed_values if self.smooth_chop else self._values
            
            # データフレームに変換
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # 必要なカラムを確認
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # ドミナントサイクルの計算
            dc_values = self.dc_detector.calculate(data)
            
            # サイクル期間の制限とキャッシュ
            self._cycle_periods = np.clip(dc_values, self.min_cycle, self.max_cycle)
            
            # サイクル効率比（CER）の計算
            self._er_values = self.cycle_er.calculate(data)
            
            # Numba最適化関数でチョピネス計算
            chop_values = calculate_choppiness_array(high, low, close, self._cycle_periods)
            
            # ALMAによるスムージング（有効な場合）
            smoothed_chop_values = chop_values.copy()  # デフォルトはスムージングなし
            if self.smooth_chop:
                try:
                    if self.chop_alma_smoother is None:
                        self.chop_alma_smoother = ALMA(
                            period=self.chop_alma_period,
                            offset=self.chop_alma_offset,
                            sigma=self.chop_alma_sigma,
                            src_type='close',
                            use_kalman_filter=False
                        )
                        
                    smoothed_values = self.chop_alma_smoother.calculate(chop_values)
                    
                    # NaNの処理（最初の数ポイントはNaNになるため、元の値で埋める）
                    nan_indices = np.isnan(smoothed_values)
                    smoothed_chop_values = smoothed_values.copy()
                    smoothed_chop_values[nan_indices] = chop_values[nan_indices]
                except Exception as e:
                    self.logger.error(f"チョピネス値のスムージング中にエラー: {str(e)}。生の値を使用します。")
                    smoothed_chop_values = chop_values.copy()  # エラー時は元の値を使用
            
            # 動的しきい値の計算 (サイクル効率比を使用)
            final_values = smoothed_chop_values if self.smooth_chop else chop_values
            self._dynamic_threshold = calculate_dynamic_threshold(
                self._er_values, 
                self.max_threshold,
                self.min_threshold
            )
            
            # チョピネス状態の計算
            self._chop_state = calculate_chop_state(final_values, self._dynamic_threshold)

            # 結果を保存してキャッシュ
            self._values = chop_values
            self._smoothed_values = smoothed_chop_values
            self._data_hash = data_hash
            
            # 結果オブジェクトを作成
            self._result = CycleChoppinessResult(
                values=final_values,
                raw_values=chop_values,
                dynamic_threshold=self._dynamic_threshold,
                chop_state=self._chop_state,
                cycle_periods=self._cycle_periods,
                er_values=self._er_values
            )

            # スムージングが有効な場合はスムージングされた値を返す
            return final_values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクルチョピネス計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            
            # エラー時はNaN配列
            self._values = np.full(data_len, np.nan)
            self._smoothed_values = np.full(data_len, np.nan)
            self._dynamic_threshold = np.full(data_len, np.nan)
            self._chop_state = np.full(data_len, np.nan)
            self._cycle_periods = np.full(data_len, np.nan)
            self._er_values = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュクリア
            
            # エラー時の結果オブジェクト
            self._result = CycleChoppinessResult(
                values=np.full(data_len, np.nan),
                raw_values=np.full(data_len, np.nan),
                dynamic_threshold=np.full(data_len, np.nan),
                chop_state=np.full(data_len, np.nan),
                cycle_periods=np.full(data_len, np.nan),
                er_values=np.full(data_len, np.nan)
            )
            
            return self._smoothed_values if self.smooth_chop else self._values
    
    def get_result(self) -> Optional[CycleChoppinessResult]:
        """
        計算結果全体を取得する
        
        Returns:
            CycleChoppinessResult: 計算結果オブジェクト
        """
        return self._result
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: 動的しきい値の配列
        """
        return self._dynamic_threshold if self._dynamic_threshold is not None else np.array([])
    
    def get_chop_state(self) -> np.ndarray:
        """
        チョピネス状態を取得する (1=高チョピネス、0=低チョピネス、NaN=不明)
        
        Returns:
            np.ndarray: チョピネス状態の配列
        """
        return self._chop_state if self._chop_state is not None else np.array([])
    
    def get_er_values(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is not None:
            return self._result.er_values
        return np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._smoothed_values = None
        self._cycle_periods = None
        self._dynamic_threshold = None
        self._chop_state = None
        self._er_values = None
        self._result = None
        self._data_hash = None
        if self.chop_alma_smoother and hasattr(self.chop_alma_smoother, 'reset'):
            self.chop_alma_smoother.reset()
        if hasattr(self.cycle_er, 'reset'):
            self.cycle_er.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        smooth_str = f", smooth={self.smooth_chop}" if self.smooth_chop else ""
        return f"CycleChop(det={self.detector_type}, part={self.cycle_part}, src={self.src_type}{smooth_str})"
