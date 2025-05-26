#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
from dataclasses import dataclass

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource


@dataclass
class CycleRSXResult:
    """サイクルRSXの計算結果"""
    values: np.ndarray          # サイクルRSXの値
    cycle_periods: np.ndarray   # 計算に使用された動的サイクル期間
    overbought_signals: np.ndarray  # 買われすぎシグナル
    oversold_signals: np.ndarray    # 売られすぎシグナル


@njit
def calculate_rsx_for_dynamic_period(
    prices: np.ndarray,
    start_idx: int,
    period: int
) -> float:
    """
    指定された期間での単一RSX値を計算する最適化関数
    
    Args:
        prices: 価格配列
        start_idx: 計算開始インデックス
        period: 計算期間
    
    Returns:
        float: RSX値（0-100の範囲）
    """
    if start_idx < period or period < 2:
        return np.nan
    
    # 必要な期間のデータを取得
    data_slice = prices[start_idx-period+1:start_idx+1]
    length = len(data_slice)
    
    if length < period:
        return np.nan
    
    # RSX計算のための初期化
    f8 = data_slice * 100.0  # 元の値を100倍
    
    # 初期値の設定
    f10 = np.zeros(length)
    v8 = np.zeros(length)
    f28 = np.zeros(length)
    f30 = np.zeros(length)
    vC = np.zeros(length)
    f38 = np.zeros(length)
    f40 = np.zeros(length)
    v10 = np.zeros(length)
    f48 = np.zeros(length)
    f50 = np.zeros(length)
    v14 = np.zeros(length)
    f58 = np.zeros(length)
    f60 = np.zeros(length)
    v18 = np.zeros(length)
    f68 = np.zeros(length)
    f70 = np.zeros(length)
    v1C = np.zeros(length)
    f78 = np.zeros(length)
    f80 = np.zeros(length)
    v20 = np.zeros(length)
    
    f88 = np.zeros(length)
    f90 = np.zeros(length)
    f0 = np.zeros(length)
    v4 = np.zeros(length)
    
    # パラメータの計算
    f18 = 3.0 / (period + 2.0)
    f20 = 1.0 - f18
    
    for i in range(1, length):
        # 価格変化の計算
        f10[i] = f8[i-1]
        v8[i] = f8[i] - f10[i]
        
        # フィルタリング（1段階目）
        f28[i] = f20 * f28[i-1] + f18 * v8[i]
        f30[i] = f18 * f28[i] + f20 * f30[i-1]
        vC[i] = f28[i] * 1.5 - f30[i] * 0.5
        
        # フィルタリング（2段階目）
        f38[i] = f20 * f38[i-1] + f18 * vC[i]
        f40[i] = f18 * f38[i] + f20 * f40[i-1]
        v10[i] = f38[i] * 1.5 - f40[i] * 0.5
        
        # フィルタリング（3段階目）
        f48[i] = f20 * f48[i-1] + f18 * v10[i]
        f50[i] = f18 * f48[i] + f20 * f50[i-1]
        v14[i] = f48[i] * 1.5 - f50[i] * 0.5
        
        # 絶対値のフィルタリング（1段階目）
        f58[i] = f20 * f58[i-1] + f18 * abs(v8[i])
        f60[i] = f18 * f58[i] + f20 * f60[i-1]
        v18[i] = f58[i] * 1.5 - f60[i] * 0.5
        
        # 絶対値のフィルタリング（2段階目）
        f68[i] = f20 * f68[i-1] + f18 * v18[i]
        f70[i] = f18 * f68[i] + f20 * f70[i-1]
        v1C[i] = f68[i] * 1.5 - f70[i] * 0.5
        
        # 絶対値のフィルタリング（3段階目）
        f78[i] = f20 * f78[i-1] + f18 * v1C[i]
        f80[i] = f18 * f78[i] + f20 * f80[i-1]
        v20[i] = f78[i] * 1.5 - f80[i] * 0.5
        
        # カウンタの計算
        if f90[i-1] == 0:
            if period - 1 >= 5:
                f88[i] = period - 1
            else:
                f88[i] = 5
        else:
            if f88[i-1] <= f90[i-1]:
                f88[i] = f88[i-1] + 1
            else:
                f88[i] = f90[i-1] + 1
        
        # フラグの計算
        if f88[i] >= f90[i-1] and f8[i] != f10[i]:
            f0[i] = 1
        else:
            f0[i] = 0
        
        if f88[i] == f90[i-1] and f0[i] == 0:
            f90[i] = 0
        else:
            f90[i] = f90[i-1]
        
        # RSXの計算
        if f88[i] < f90[i] and v20[i] > 0:
            v4[i] = (v14[i] / v20[i] + 1) * 50
        else:
            v4[i] = 50
        
        # 0-100の範囲にクリップ
        if v4[i] > 100:
            v4[i] = 100
        elif v4[i] < 0:
            v4[i] = 0
    
    # 最後の値を返す
    return v4[-1]


@njit
def calculate_cycle_rsx_array(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    動的期間でサイクルRSX値の配列を計算する
    
    Args:
        prices: 価格配列
        periods: 各時点での計算期間配列
    
    Returns:
        np.ndarray: サイクルRSX値の配列
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    # 最小必要データポイント
    min_points = 5
    
    for i in range(min_points, n):
        period = int(periods[i])
        if period < min_points:
            period = min_points
            
        # 十分なデータがある場合のみ計算
        if i >= period:
            rsx_value = calculate_rsx_for_dynamic_period(prices, i, period)
            result[i] = rsx_value
    
    return result


@njit
def calculate_overbought_oversold_signals(
    rsx_values: np.ndarray,
    overbought_level: float = 70.0,
    oversold_level: float = 30.0
) -> tuple:
    """
    買われすぎ・売られすぎのシグナルを計算する
    
    Args:
        rsx_values: RSX値の配列
        overbought_level: 買われすぎレベル
        oversold_level: 売られすぎレベル
    
    Returns:
        (買われすぎシグナル, 売られすぎシグナル)のタプル
    """
    length = len(rsx_values)
    overbought_signals = np.zeros(length, dtype=np.float64)
    oversold_signals = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(rsx_values[i]):
            if rsx_values[i] >= overbought_level:
                overbought_signals[i] = 1.0
            if rsx_values[i] <= oversold_level:
                oversold_signals[i] = 1.0
    
    return overbought_signals, oversold_signals


class CycleRSX(Indicator):
    """
    サイクルRSX（Relative Strength eXtended）インディケーター
    
    ドミナントサイクルを使用して動的な期間でのRSXを計算します。
    RSXはRSIの改良版で、よりスムーズで反応が速い特徴があります。
    RSX自体に多段階のフィルタリングが組み込まれているため、追加のスムージングは不要です。
    """
    
    def __init__(
        self,
        detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 89,
        cycle_part: float = 0.4,
        max_cycle: int = 55,
        min_cycle: int = 5,
        max_output: int = 34,
        min_output: int = 3,
        src_type: str = 'hlc3',
        overbought_level: float = 70.0,
        oversold_level: float = 30.0
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分（デフォルト）
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
            overbought_level: 買われすぎレベル（デフォルト: 70）
            oversold_level: 売られすぎレベル（デフォルト: 30）
        """
        indicator_name = f"CycleRSX(det={detector_type},part={cycle_part},src={src_type})"
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
        
        # シグナル関連パラメータ
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
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
        
        # 結果キャッシュ
        self._values = None  # RSX値
        self._data_hash = None
        self._cycle_periods = None  # 動的サイクル期間配列
        self._overbought_signals = None  # 買われすぎシグナル
        self._oversold_signals = None  # 売られすぎシグナル
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
            f"ob={self.overbought_level}_os={self.oversold_level}"
        )
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクルRSXを計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: サイクルRSXの値
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._values
            
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
            
            # Numba最適化関数でサイクルRSX計算
            rsx_values = calculate_cycle_rsx_array(source_prices, self._cycle_periods)
            
            # 買われすぎ・売られすぎシグナルの計算
            self._overbought_signals, self._oversold_signals = calculate_overbought_oversold_signals(
                rsx_values, self.overbought_level, self.oversold_level
            )

            # 結果を保存してキャッシュ
            self._values = rsx_values
            self._data_hash = data_hash
            
            # 結果オブジェクトを作成
            self._result = CycleRSXResult(
                values=rsx_values,
                cycle_periods=self._cycle_periods,
                overbought_signals=self._overbought_signals,
                oversold_signals=self._oversold_signals
            )

            return rsx_values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクルRSX計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            
            # エラー時はNaN配列
            self._values = np.full(data_len, np.nan)
            self._overbought_signals = np.full(data_len, np.nan)
            self._oversold_signals = np.full(data_len, np.nan)
            self._cycle_periods = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュクリア
            
            # エラー時の結果オブジェクト
            self._result = CycleRSXResult(
                values=np.full(data_len, np.nan),
                cycle_periods=np.full(data_len, np.nan),
                overbought_signals=np.full(data_len, np.nan),
                oversold_signals=np.full(data_len, np.nan)
            )
            
            return self._values
    
    def get_result(self) -> Optional[CycleRSXResult]:
        """
        計算結果全体を取得する
        
        Returns:
            CycleRSXResult: 計算結果オブジェクト
        """
        return self._result
    
    def get_overbought_oversold(
        self,
        ob_level: float = None,
        os_level: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        買われすぎ・売られすぎのシグナルを取得
        
        Args:
            ob_level: 買われすぎのレベル（Noneの場合はコンストラクタで設定した値を使用）
            os_level: 売られすぎのレベル（Noneの場合はコンストラクタで設定した値を使用）
        
        Returns:
            (買われすぎシグナル, 売られすぎシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        if ob_level is None and os_level is None:
            # 既に計算済みのシグナルを返す
            return self._overbought_signals, self._oversold_signals
        else:
            # 新しいレベルで再計算
            ob = self.overbought_level if ob_level is None else ob_level
            os = self.oversold_level if os_level is None else os_level
            return calculate_overbought_oversold_signals(self._result.values, ob, os)
    
    def get_crossover_signals(self, level: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        クロスオーバー・クロスアンダーのシグナルを取得
        
        Args:
            level: クロスするレベル（デフォルト：50）
        
        Returns:
            (クロスオーバーシグナル, クロスアンダーシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        values = self._result.values
        
        # 1つ前の値を取得（最初の要素には前の値がないので同じ値を使用）
        prev_values = np.roll(values, 1)
        prev_values[0] = values[0]
        
        # クロスオーバー: 前の値がレベル未満で、現在の値がレベル以上
        crossover = np.where(
            (prev_values < level) & (values >= level),
            1, 0
        )
        
        # クロスアンダー: 前の値がレベル以上で、現在の値がレベル未満
        crossunder = np.where(
            (prev_values >= level) & (values < level),
            1, 0
        )
        
        return crossover, crossunder
    
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
        self._cycle_periods = None
        self._overbought_signals = None
        self._oversold_signals = None
        self._result = None
        self._data_hash = None
        if hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"CycleRSX(det={self.detector_type}, part={self.cycle_part}, src={self.src_type})" 