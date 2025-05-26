#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from dataclasses import dataclass

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource
from .alma import ALMA


@dataclass
class CycleROCResult:
    """サイクルROCの計算結果"""
    values: np.ndarray          # サイクルROCの値
    raw_values: np.ndarray      # 生のサイクルROCの値（スムージング前）
    cycle_periods: np.ndarray   # 計算に使用された動的サイクル期間
    roc_signals: np.ndarray     # ROCシグナル (1=上昇、-1=下降、0=中立、NaN=不明)


@njit
def calculate_roc_for_period(prices: np.ndarray, period: int) -> float:
    """
    指定された期間でのROC値を計算する最適化関数
    
    Args:
        prices: 価格配列
        period: 計算期間
    
    Returns:
        float: ROC値（変化率パーセンテージ）
    """
    # 配列長チェック
    n = len(prices)
    if n < period + 1:
        return np.nan
        
    # ROC計算: ((現在値 - n期前の値) / n期前の値) * 100
    current_price = prices[-1]
    past_price = prices[-(period + 1)]
    
    if past_price == 0.0:
        return np.nan
        
    roc = ((current_price - past_price) / past_price) * 100.0
    return roc


@njit
def calculate_roc_array(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    動的期間でROC値の配列を計算する
    
    Args:
        prices: 価格配列
        periods: 各時点での計算期間配列
    
    Returns:
        np.ndarray: ROC値の配列
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    # 最小必要データポイント
    min_points = 2
    
    for i in range(min_points, n):
        period = int(periods[i])
        if period < min_points:
            period = min_points
            
        # 十分なデータがある場合のみ計算
        if i >= period:
            # 計算に必要な部分配列
            price_slice = prices[i-period:i+1]
            result[i] = calculate_roc_for_period(price_slice, period)
    
    return result


@njit
def calculate_roc_signals(roc_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    ROC値に基づいてシグナルを計算する
    
    Args:
        roc_values: ROC値の配列
        threshold: シグナル判定のしきい値
    
    Returns:
        シグナルの配列 (1=上昇、-1=下降、0=中立、NaN=不明)
    """
    length = len(roc_values)
    signals = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(roc_values[i]):
            continue
        
        if roc_values[i] > threshold:
            signals[i] = 1.0  # 上昇
        elif roc_values[i] < -threshold:
            signals[i] = -1.0  # 下降
        else:
            signals[i] = 0.0  # 中立
    
    return signals


class CycleROC(Indicator):
    """
    サイクルROC（Rate of Change）インディケーター
    
    ドミナントサイクルを使用して動的な期間でのROC（変化率）を計算します。
    計算に使用する価格ソースを選択可能で、ALMAによるスムージングも可能です。
    """
    
    def __init__(
        self,
        detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.7,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 144,
        min_output: int = 13,
        src_type: str = 'hlc3',
        smooth_roc: bool = True,
        roc_alma_period: int = 5,
        roc_alma_offset: float = 0.85,
        roc_alma_sigma: float = 6,
        signal_threshold: float = 0.0  # ROCシグナル判定のしきい値
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
            smooth_roc: ROC値にALMAスムージングを適用するかどうか
            roc_alma_period: ALMAスムージングの期間
            roc_alma_offset: ALMAスムージングのオフセット
            roc_alma_sigma: ALMAスムージングのシグマ
            signal_threshold: ROCシグナル判定のしきい値
        """
        smooth_str = f"_smooth={'Y' if smooth_roc else 'N'}" if smooth_roc else ""
        indicator_name = f"CycleROC(det={detector_type},part={cycle_part},src={src_type}{smooth_str})"
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
        self.smooth_roc = smooth_roc
        self.roc_alma_period = roc_alma_period
        self.roc_alma_offset = roc_alma_offset
        self.roc_alma_sigma = roc_alma_sigma
        
        # シグナル関連パラメータ
        self.signal_threshold = signal_threshold
        
        # ドミナントサイクル検出器
        self.dc_detector = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period,
            src_type=src_type
        )
        
        # ALMAスムーザーの初期化（有効な場合）
        self.roc_alma_smoother = None
        if self.smooth_roc:
            self.roc_alma_smoother = ALMA(
                period=self.roc_alma_period,
                offset=self.roc_alma_offset,
                sigma=self.roc_alma_sigma,
                src_type='close',  # 直接値を渡すので、ソースタイプは関係ない
                use_kalman_filter=False  # すでに計算が済んでいるのでカルマンは不要
            )
        
        # 結果キャッシュ
        self._values = None  # 生のROC値
        self._smoothed_values = None  # スムージングされたROC値
        self._data_hash = None
        self._cycle_periods = None  # 動的サイクル期間配列
        self._roc_signals = None  # ROCシグナル
        self._result = None  # 計算結果オブジェクト
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # 必要なカラムを決定
        required_cols = set()
        if self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        else:
            required_cols.add('close')  # デフォルト

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
            f"smooth={self.smooth_roc}_{self.roc_alma_period}_{self.roc_alma_offset}_{self.roc_alma_sigma}_"
            f"threshold={self.signal_threshold}"
        )
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクルROCを計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: サイクルROCの値（スムージングが有効な場合はスムージングされた値）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._smoothed_values if self.smooth_roc else self._values
            
            # データフレームに変換
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # 価格ソースを抽出
            source_prices = PriceSource.calculate_source(df, self.src_type)
            
            # ドミナントサイクルの計算
            dc_values = self.dc_detector.calculate(data)
            
            # サイクル期間の制限とキャッシュ
            self._cycle_periods = np.clip(dc_values, self.min_cycle, self.max_cycle)
            
            # Numba最適化関数でROC計算
            roc_values = calculate_roc_array(source_prices, self._cycle_periods)
            
            # ALMAによるスムージング（有効な場合）
            smoothed_roc_values = roc_values.copy()  # デフォルトはスムージングなし
            if self.smooth_roc:
                try:
                    if self.roc_alma_smoother is None:
                        self.roc_alma_smoother = ALMA(
                            period=self.roc_alma_period,
                            offset=self.roc_alma_offset,
                            sigma=self.roc_alma_sigma,
                            src_type='close',
                            use_kalman_filter=False
                        )
                        
                    smoothed_values = self.roc_alma_smoother.calculate(roc_values)
                    
                    # NaNの処理（最初の数ポイントはNaNになるため、元の値で埋める）
                    nan_indices = np.isnan(smoothed_values)
                    smoothed_roc_values = smoothed_values.copy()
                    smoothed_roc_values[nan_indices] = roc_values[nan_indices]
                except Exception as e:
                    self.logger.error(f"ROC値のスムージング中にエラー: {str(e)}。生の値を使用します。")
                    smoothed_roc_values = roc_values.copy()  # エラー時は元の値を使用
            
            # ROCシグナルの計算
            final_values = smoothed_roc_values if self.smooth_roc else roc_values
            self._roc_signals = calculate_roc_signals(final_values, self.signal_threshold)

            # 結果を保存してキャッシュ
            self._values = roc_values
            self._smoothed_values = smoothed_roc_values
            self._data_hash = data_hash
            
            # 結果オブジェクトを作成
            self._result = CycleROCResult(
                values=final_values,
                raw_values=roc_values,
                cycle_periods=self._cycle_periods,
                roc_signals=self._roc_signals
            )

            # スムージングが有効な場合はスムージングされた値を返す
            return final_values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクルROC計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            
            # エラー時はNaN配列
            self._values = np.full(data_len, np.nan)
            self._smoothed_values = np.full(data_len, np.nan)
            self._roc_signals = np.full(data_len, np.nan)
            self._cycle_periods = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュクリア
            
            # エラー時の結果オブジェクト
            self._result = CycleROCResult(
                values=np.full(data_len, np.nan),
                raw_values=np.full(data_len, np.nan),
                cycle_periods=np.full(data_len, np.nan),
                roc_signals=np.full(data_len, np.nan)
            )
            
            return self._smoothed_values if self.smooth_roc else self._values
    
    def get_result(self) -> Optional[CycleROCResult]:
        """
        計算結果全体を取得する
        
        Returns:
            CycleROCResult: 計算結果オブジェクト
        """
        return self._result
    
    def get_roc_signals(self) -> np.ndarray:
        """
        ROCシグナルを取得する (1=上昇、-1=下降、0=中立、NaN=不明)
        
        Returns:
            np.ndarray: ROCシグナルの配列
        """
        return self._roc_signals if self._roc_signals is not None else np.array([])
    
    def get_cycle_periods(self) -> np.ndarray:
        """
        動的サイクル期間を取得する
        
        Returns:
            np.ndarray: 動的サイクル期間の配列
        """
        return self._cycle_periods if self._cycle_periods is not None else np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._smoothed_values = None
        self._cycle_periods = None
        self._roc_signals = None
        self._result = None
        self._data_hash = None
        if self.roc_alma_smoother and hasattr(self.roc_alma_smoother, 'reset'):
            self.roc_alma_smoother.reset()
        if hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        smooth_str = f", smooth={self.smooth_roc}" if self.smooth_roc else ""
        return f"CycleROC(det={self.detector_type}, part={self.cycle_part}, src={self.src_type}{smooth_str})" 