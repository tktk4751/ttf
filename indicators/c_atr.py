#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .alma import calculate_alma
from .hyper_smoother import hyper_smoother, calculate_hyper_smoother_numba
from .ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class CATRResult:
    """CATRの計算結果"""
    values: np.ndarray        # CATRの値（%ベース）
    absolute_values: np.ndarray  # CATRの値（金額ベース）
    tr: np.ndarray           # True Range
    er: np.ndarray           # サイクル効率比（CER）
    atr_period: np.ndarray  # ドミナントサイクルから決定されたATR期間
    dc_values: np.ndarray    # ドミナントサイクル値


@vectorize(['float64(float64, float64)'], nopython=True, fastmath=True, cache=True)
def max_vec(a: float, b: float) -> float:
    """aとbの最大値を返す（ベクトル化版）"""
    return max(a, b)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def max3_vec(a: float, b: float, c: float) -> float:
    """a, b, cの最大値を返す（ベクトル化版）"""
    return max(a, max(b, c))


@njit(fastmath=True, parallel=True, cache=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Rangeを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    if length > 0:
        tr[0] = high[0] - low[0]
    
    # 一時配列を使用して計算効率化
    tr1 = np.zeros(length, dtype=np.float64)
    tr2 = np.zeros(length, dtype=np.float64)
    tr3 = np.zeros(length, dtype=np.float64)
    
    # 各要素のTRの計算を分解して並列化
    for i in prange(1, length):
        # 当日の高値 - 当日の安値
        tr1[i] = high[i] - low[i]
        # |当日の高値 - 前日の終値|
        tr2[i] = abs(high[i] - close[i-1])
        # |当日の安値 - 前日の終値|
        tr3[i] = abs(low[i] - close[i-1])
    
    # 最大値を計算（並列処理）
    for i in prange(1, length):
        tr[i] = max(tr1[i], max(tr2[i], tr3[i]))
    
    return tr


@njit(fastmath=True, parallel=True, cache=True)
def calculate_c_atr(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    er: np.ndarray,
    atr_period: np.ndarray,
    max_period: int,
    smoother_type: str = 'alma'  # 'alma'または'hyper'
) -> np.ndarray:
    """
    CATRを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列（ERまたはCER）
        atr_period: サイクル検出器から直接決定されたATR期間の配列
        max_period: 最大期間（計算開始位置用）
        smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
    
    Returns:
        CATRの値を返す
    """
    length = len(high)
    c_atr = np.zeros(length, dtype=np.float64)
    
    # True Rangeの計算
    tr = calculate_true_range(high, low, close)
    
    # 各時点での平滑化を計算
    for i in prange(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(atr_period[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置までのTRデータを取得（効率化のためウィンドウサイズを制限）
        start_idx = max(0, i-curr_period*2)
        window = tr[start_idx:i+1]
        
        # 選択された平滑化アルゴリズムを適用
        if smoother_type == 'alma':
            # ALMAを使用して平滑化（固定パラメータ：offset=0.85, sigma=6）
            smoothed_values = calculate_alma(window, curr_period, 0.85, 6.0)
        else:  # 'hyper'
            # ハイパースムーサーを使用して平滑化
            smoothed_values = calculate_hyper_smoother_numba(window, curr_period)
        
        # 最後の値をATRとして使用
        if len(smoothed_values) > 0 and not np.isnan(smoothed_values[-1]):
            c_atr[i] = smoothed_values[-1]
    
    return c_atr


@njit(fastmath=True, parallel=True, cache=True)
def calculate_percent_atr(absolute_atr: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    金額ベースのATRから%ベースのATRを計算する（並列高速化版）
    
    Args:
        absolute_atr: 金額ベースのATR配列
        close: 終値の配列
    
    Returns:
        %ベースのATR配列
    """
    length = len(absolute_atr)
    percent_atr = np.zeros_like(absolute_atr, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(length):
        if not np.isnan(absolute_atr[i]) and close[i] > 0:
            percent_atr[i] = absolute_atr[i] / close[i]
    
    return percent_atr


class CATR(Indicator):
    """
    CATR（Cycle Average True Range）インジケーター
    
    特徴:
    - ドミナントサイクル検出器から直接ATR期間を決定
    - サイクル効率比（CER）を利用してトレンド強度に応じたボラティリティの測定
    - ALMAまたはハイパースムーサーによる平滑化
    - 金額ベースと%ベースの両方の値を提供
    
    使用方法:
    - ボラティリティに基づいた利益確定・損切りレベルの設定
    - ATRチャネルやボラティリティストップの構築
    - トレンドの強さに適応したポジションサイジング
    - 異なる価格帯の銘柄間でのボラティリティ比較（%ベース）
    """
    
    def __init__(
        self,
        detector_type: str = 'hody',             # 検出器タイプ
        cycle_part: float = 0.5,                 # サイクル部分の倍率
        lp_period: int = 5,
        hp_period: int = 55,
        max_cycle: int = 55,                     # 最大サイクル期間
        min_cycle: int = 5,                      # 最小サイクル期間
        max_output: int = 34,                    # 最大出力値
        min_output: int = 5,                     # 最小出力値
        smoother_type: str = 'alma'              # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 55）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 5）
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー
        """
        super().__init__(f"CATR({detector_type}, {max_output}, {min_output}, {smoother_type})")
        
        # パラメータの保存
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.smoother_type = smoother_type
        
        # ドミナントサイクル検出器を初期化
        self.dc_detector = EhlersUnifiedDC(
            detector_type=self.detector_type,
            cycle_part=self.cycle_part,
            lp_period=self.lp_period,
            hp_period=self.hp_period,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output
        )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        # np.ndarrayの場合はバイト文字列に変換
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        # pd.DataFrameの場合は必要な列をバイト文字列に変換
        else:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            data_bytes = high.tobytes() + low.tobytes() + close.tobytes()
        
        # external_erがある場合は、そのハッシュ値も含める
        er_bytes = b'' if external_er is None else external_er.tobytes()
        
        # パラメータから文字列を作成
        param_str = f"{self.detector_type}_{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_" \
                    f"{self.max_output}_{self.min_output}_{self.smoother_type}"
        
        # データとパラメータのハッシュ値を組み合わせて返す
        return str(hash(data_bytes + er_bytes + param_str.encode()))
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        CATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            CATRの値（%ベース）
        """
        try:
            # サイクル効率比（CER）の検証
            if external_er is None:
                raise ValueError("サイクル効率比（CER）は必須です。external_erパラメータを指定してください")
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換（効率化）
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                # NumPyに一度で変換して計算を効率化
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
                close = np.asarray(data['close'].values, dtype=np.float64)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    # 一度にNumPy配列に変換して計算効率化
                    high = np.asarray(data[:, 1], dtype=np.float64)  # high
                    low = np.asarray(data[:, 2], dtype=np.float64)   # low
                    close = np.asarray(data[:, 3], dtype=np.float64) # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # ドミナントサイクルの計算 - 直接ATR期間として使用
            dc_values = self.dc_detector.calculate(data)
            atr_period = np.asarray(dc_values, dtype=np.float64)
            
            # 最大ATR期間の最大値を取得（計算開始位置用）
            max_period_value = int(np.nanmax(atr_period))
            if np.isnan(max_period_value) or max_period_value < 10:
                max_period_value = 34  # デフォルト値
            
            # データ長の検証
            data_length = len(high)
            if data_length < max_period_value:
                raise ValueError(f"データ長({data_length})が必要な期間よりも短いです")
            
            # サイクル効率比（CER）を使用（高速化）
            er = np.asarray(external_er, dtype=np.float64)
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # CATRの計算（並列版 - 高速化）
            c_atr_values = calculate_c_atr(
                high,
                low,
                close,
                er,
                atr_period,
                max_period_value,
                self.smoother_type
            )
            
            # 金額ベースのATR値を保存
            absolute_atr_values = c_atr_values
            
            # %ベースのATR値に変換（終値に対する比率）（並列版 - 高速化）
            percent_atr_values = calculate_percent_atr(absolute_atr_values, close)
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = CATRResult(
                values=np.copy(percent_atr_values),  # %ベースのATR
                absolute_values=np.copy(absolute_atr_values),  # 金額ベースのATR
                tr=np.copy(calculate_true_range(high, low, close)),  # TRを再計算（高速版）
                er=np.copy(er),
                atr_period=np.copy(atr_period),
                dc_values=np.copy(dc_values)  # ドミナントサイクル値を保存
            )
            
            self._values = percent_atr_values  # 標準インジケーターインターフェース用
            return percent_atr_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CATR計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時は前回の結果を維持
            if self._result is None:
                return np.array([])
            return self._result.values
    
    def get_dc_values(self) -> np.ndarray:
        """
        ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range (TR)の値を取得する
        
        Returns:
            np.ndarray: TRの値
        """
        if self._result is None:
            return np.array([])
        return self._result.tr
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_atr_period(self) -> np.ndarray:
        """
        ATR期間の値を取得する
        
        Returns:
            np.ndarray: ATR期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.atr_period
    
    def get_percent_atr(self) -> np.ndarray:
        """
        %ベースのATRを取得する
        
        Returns:
            np.ndarray: %ベースのATR値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            return np.array([])
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_atr(self) -> np.ndarray:
        """
        金額ベースのATRを取得する
        
        Returns:
            np.ndarray: 金額ベースのATR値
        """
        if self._result is None:
            return np.array([])
        return self._result.absolute_values
    
    def get_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのATR × 倍数
        """
        atr = self.get_percent_atr()
        return atr * multiplier
    
    def get_absolute_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        金額ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 金額ベースのATR × 倍数
        """
        abs_atr = self.get_absolute_atr()
        return abs_atr * multiplier
        
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.dc_detector.reset() 