#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from .indicator import Indicator
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .cycle_chop import CycleChoppiness


@dataclass
class CycleTrendIndexResult:
    """サイクルトレンドインデックスの計算結果"""
    values: np.ndarray          # サイクルトレンドインデックスの値
    er_values: np.ndarray       # サイクル効率比（CER）の値
    chop_values: np.ndarray     # サイクルチョピネスの値
    dynamic_threshold: np.ndarray  # 動的しきい値
    trend_state: np.ndarray     # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)
    er_cycle: np.ndarray        # CERのサイクル期間
    chop_cycle: np.ndarray      # チョピネスのサイクル期間


@njit(fastmath=True)
def calculate_cycle_trend_index(er_values: np.ndarray, chop_values: np.ndarray) -> np.ndarray:
    """
    サイクル効率比とサイクルチョピネスからサイクルトレンドインデックスを計算する
    
    Args:
        er_values: サイクル効率比の配列
        chop_values: サイクルチョピネスの配列
    
    Returns:
        サイクルトレンドインデックスの配列
    """
    length = len(er_values)
    result = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er_values[i]) or np.isnan(chop_values[i]):
            result[i] = np.nan
        else:
            # CER値を-1〜1から0〜1に正規化（絶対値を取る）
            normalized_er = (abs(er_values[i]) + 1) / 2 if abs(er_values[i]) <= 1 else 1.0
            # サイクルチョピネスは既に0〜1のため正規化不要
            
            # 2つの値の平均を計算
            result[i] = (normalized_er + chop_values[i]) / 2
    
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
def calculate_trend_state(
    cti_values: np.ndarray,
    dynamic_threshold: np.ndarray
) -> np.ndarray:
    """
    サイクルトレンドインデックスと動的しきい値に基づいてトレンド状態を計算する
    
    Args:
        cti_values: サイクルトレンドインデックスの配列
        dynamic_threshold: 動的しきい値の配列
    
    Returns:
        トレンド状態の配列 (1=トレンド、0=レンジ、NaN=不明)
    """
    length = len(cti_values)
    trend_state = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(cti_values[i]) or np.isnan(dynamic_threshold[i]):
            continue
        trend_state[i] = 1.0 if cti_values[i] >= dynamic_threshold[i] else 0.0
    
    return trend_state


class CycleTrendIndex(Indicator):
    """
    サイクルトレンドインデックス（Cycle Trend Index）インディケーター
    
    サイクル効率比（CER）とサイクルチョピネスを組み合わせて
    トレンド/レンジを判断するための指標。
    2つの指標の平均値を取り、動的しきい値との比較で市場状態を判断する。
    
    特徴:
    - サイクル効率比（CER）を使用してトレンド測定
    - サイクルチョピネスを使用してレンジ測定
    - 2つの指標を組み合わせて総合的なトレンド評価
    - サイクル効率比に基づく動的しきい値でトレンド/レンジ状態を判定
    """
    
    def __init__(
        self,
        # 共通パラメータ
        detector_type: str = 'phac_e',
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        lp_period: int = 5,
        hp_period: int = 144,
        
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
        self_adaptive: bool = False,
        
        # チョピネス固有パラメータ
        chop_detector_type: Optional[str] = None,
        chop_cycle_part: Optional[float] = None,
        chop_max_cycle: Optional[int] = None,
        chop_min_cycle: Optional[int] = None,
        chop_max_output: Optional[int] = None, 
        chop_min_output: Optional[int] = None,
        chop_src_type: Optional[str] = None,
        chop_lp_period: Optional[int] = None,
        chop_hp_period: Optional[int] = None,
        smooth_chop: bool = True,
        chop_alma_period: int = 3,
        chop_alma_offset: float = 0.85,
        chop_alma_sigma: float = 6,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 共通のドミナントサイクル検出器タイプ（個別指定がない場合の既定値）
            cycle_part: 共通のサイクル部分（個別指定がない場合の既定値）
            max_cycle: 共通の最大サイクル期間（個別指定がない場合の既定値）
            min_cycle: 共通の最小サイクル期間（個別指定がない場合の既定値）
            max_output: 共通の最大出力値（個別指定がない場合の既定値）
            min_output: 共通の最小出力値（個別指定がない場合の既定値）
            src_type: 共通の価格ソース（個別指定がない場合の既定値）
            lp_period: 共通のローパスフィルター期間（個別指定がない場合の既定値）
            hp_period: 共通のハイパスフィルター期間（個別指定がない場合の既定値）
            
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
            
            chop_detector_type: チョピネス固有のドミナントサイクル検出器タイプ
            chop_cycle_part: チョピネス固有のサイクル部分
            chop_max_cycle: チョピネス固有の最大サイクル期間 
            chop_min_cycle: チョピネス固有の最小サイクル期間
            chop_max_output: チョピネス固有の最大出力値
            chop_min_output: チョピネス固有の最小出力値
            chop_src_type: チョピネス固有の価格ソース
            chop_lp_period: チョピネス固有のローパスフィルター期間
            chop_hp_period: チョピネス固有のハイパスフィルター期間
            smooth_chop: チョピネス値にALMAスムージングを適用するかどうか
            chop_alma_period: ALMAスムージングの期間
            chop_alma_offset: ALMAスムージングのオフセット
            chop_alma_sigma: ALMAスムージングのシグマ
            
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
        """
        super().__init__(
            f"CycleTrendIndex(det={detector_type}, er_smooth={'Y' if smooth_er else 'N'}, chop_smooth={'Y' if smooth_chop else 'N'})"
        )
        
        # 共通パラメータを保存
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        
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
        
        # チョピネス固有パラメータ
        self.chop_detector_type = chop_detector_type if chop_detector_type is not None else detector_type
        self.chop_cycle_part = chop_cycle_part if chop_cycle_part is not None else cycle_part
        self.chop_max_cycle = chop_max_cycle if chop_max_cycle is not None else max_cycle
        self.chop_min_cycle = chop_min_cycle if chop_min_cycle is not None else min_cycle
        self.chop_max_output = chop_max_output if chop_max_output is not None else max_output
        self.chop_min_output = chop_min_output if chop_min_output is not None else min_output
        self.chop_src_type = chop_src_type if chop_src_type is not None else src_type
        self.chop_lp_period = chop_lp_period if chop_lp_period is not None else lp_period
        self.chop_hp_period = chop_hp_period if chop_hp_period is not None else hp_period
        self.smooth_chop = smooth_chop
        self.chop_alma_period = chop_alma_period
        self.chop_alma_offset = chop_alma_offset
        self.chop_alma_sigma = chop_alma_sigma
        
        # 動的しきい値パラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        
        # 子インディケーターの初期化
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
        
        self.cycle_chop = CycleChoppiness(
            detector_type=self.chop_detector_type,
            cycle_part=self.chop_cycle_part,
            lp_period=self.chop_lp_period,
            hp_period=self.chop_hp_period,
            max_cycle=self.chop_max_cycle,
            min_cycle=self.chop_min_cycle,
            max_output=self.chop_max_output,
            min_output=self.chop_min_output,
            src_type=self.chop_src_type,
            smooth_chop=self.smooth_chop,
            chop_alma_period=self.chop_alma_period,
            chop_alma_offset=self.chop_alma_offset,
            chop_alma_sigma=self.chop_alma_sigma
        )
        
        # キャッシュ用変数
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        if isinstance(data, pd.DataFrame):
            cols = ['open', 'high', 'low', 'close']
            data_hash_part = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            data_hash_part = hash(tuple(map(tuple, data)))
        
        # パラメータ文字列を生成
        param_str = (
            f"{self.detector_type}_{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_"
            f"{self.max_output}_{self.min_output}_{self.src_type}_{self.lp_period}_{self.hp_period}_"
            f"{self.er_detector_type}_{self.er_cycle_part}_{self.er_max_cycle}_{self.er_min_cycle}_"
            f"{self.er_max_output}_{self.er_min_output}_{self.er_src_type}_{self.er_lp_period}_{self.er_hp_period}_"
            f"{self.use_kalman_filter}_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}_"
            f"{self.smooth_er}_{self.er_alma_period}_{self.er_alma_offset}_{self.er_alma_sigma}_{self.self_adaptive}_"
            f"{self.chop_detector_type}_{self.chop_cycle_part}_{self.chop_max_cycle}_{self.chop_min_cycle}_"
            f"{self.chop_max_output}_{self.chop_min_output}_{self.chop_src_type}_{self.chop_lp_period}_{self.chop_hp_period}_"
            f"{self.smooth_chop}_{self.chop_alma_period}_{self.chop_alma_offset}_{self.chop_alma_sigma}_"
            f"{self.max_threshold}_{self.min_threshold}"
        )
        return f"{data_hash_part}_{hash(param_str)}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CycleTrendIndexResult:
        """
        サイクルトレンドインデックスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要
        
        Returns:
            CycleTrendIndexResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result
            
            self._data_hash = data_hash
            
            # サイクル効率比（CER）の計算
            er_values = self.cycle_er.calculate(data)
            er_cycle_periods = self.cycle_er.get_cycle_periods() if hasattr(self.cycle_er, 'get_cycle_periods') else np.array([])
            
            # サイクルチョピネスの計算
            chop_values = self.cycle_chop.calculate(data)
            chop_cycle_periods = self.cycle_chop.get_cycle_periods() if hasattr(self.cycle_chop, 'get_cycle_periods') else np.array([])
            
            # サイクルトレンドインデックスの計算
            cti_values = calculate_cycle_trend_index(er_values, chop_values)
            
            # 動的しきい値の計算
            dynamic_threshold = calculate_dynamic_threshold(er_values, self.max_threshold, self.min_threshold)
            
            # トレンド状態の計算
            trend_state = calculate_trend_state(cti_values, dynamic_threshold)
            
            # 結果オブジェクトを作成
            result = CycleTrendIndexResult(
                values=cti_values,
                er_values=er_values,
                chop_values=chop_values,
                dynamic_threshold=dynamic_threshold,
                trend_state=trend_state,
                er_cycle=er_cycle_periods,
                chop_cycle=chop_cycle_periods
            )
            
            self._result = result
            self._values = cti_values  # Indicatorクラスの標準出力
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"サイクルトレンドインデックス計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はNaN配列を返す
            n = len(data) if hasattr(data, '__len__') else 0
            empty_result = CycleTrendIndexResult(
                values=np.full(n, np.nan),
                er_values=np.full(n, np.nan),
                chop_values=np.full(n, np.nan),
                dynamic_threshold=np.full(n, np.nan),
                trend_state=np.full(n, np.nan),
                er_cycle=np.full(n, np.nan),
                chop_cycle=np.full(n, np.nan)
            )
            self._result = None
            self._values = np.full(n, np.nan)
            self._data_hash = None  # ハッシュもクリアして次回再計算を強制
            return empty_result
    
    # --- Getter Methods ---
    def get_result(self) -> Optional[CycleTrendIndexResult]:
        """計算結果全体を取得する"""
        return self._result
    
    def get_er_values(self) -> np.ndarray:
        """サイクル効率比（CER）の値を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.er_values
    
    def get_chop_values(self) -> np.ndarray:
        """サイクルチョピネスの値を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.chop_values
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """動的しきい値を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.dynamic_threshold
    
    def get_trend_state(self) -> np.ndarray:
        """トレンド状態を取得する（1=トレンド、0=レンジ、NaN=不明）"""
        if self._result is None:
            return np.array([])
        return self._result.trend_state
    
    def get_er_cycle(self) -> np.ndarray:
        """サイクル効率比（CER）のサイクル期間を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.er_cycle
    
    def get_chop_cycle(self) -> np.ndarray:
        """サイクルチョピネスのサイクル期間を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.chop_cycle
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self.cycle_er.reset()
        self.cycle_chop.reset()
        self._result = None
        self._data_hash = None 