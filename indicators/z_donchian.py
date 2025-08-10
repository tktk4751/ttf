#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .cycle.ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZDonchianResult:
    """Z_Donchianの計算結果"""
    upper: np.ndarray        # 上限バンド
    lower: np.ndarray        # 下限バンド
    middle: np.ndarray       # 中央線
    er: np.ndarray           # サイクル効率比（CER）
    dynamic_period: np.ndarray  # 動的ドンチャン期間
    dc_values: np.ndarray    # ドミナントサイクル値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なドンチャン期間を計算する（ベクトル化版）
    
    Args:
        er: 効率比の値（ERまたはCER）
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の値
    """
    if np.isnan(er):
        return np.nan
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = abs(er)
    return np.round(min_period + (1.0 - er_abs) * (max_period - min_period))


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: np.ndarray, min_period: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的なドンチャン期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_period: 最大期間の配列
        min_period: 最小期間の配列
    
    Returns:
        動的な期間の配列
    """
    length = len(er)
    periods = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er[i]):
            periods[i] = np.nan
        else:
            # ERが高い（トレンドが強い）ほど期間は短く、
            # ERが低い（トレンドが弱い）ほど期間は長くなる
            er_abs = abs(er[i])
            periods[i] = min_period[i] + (1.0 - er_abs) * (max_period[i] - min_period[i])
    
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True)
def calculate_z_donchian(
    high: np.ndarray, 
    low: np.ndarray, 
    er: np.ndarray,
    dynamic_period: np.ndarray,
    max_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z_Donchianを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        er: 効率比の配列（ERまたはCER）
        dynamic_period: 動的期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上限バンド、下限バンド、中央線の配列
    """
    length = len(high)
    upper = np.full(length, np.nan, dtype=np.float64)
    lower = np.full(length, np.nan, dtype=np.float64)
    middle = np.full(length, np.nan, dtype=np.float64)
    
    # 各時点でのドンチャンチャネルを計算
    for i in prange(max_period, length):
        # その時点での動的な期間を取得（最小3に制限）
        curr_period = max(3, int(dynamic_period[i]))
        
        # トレンドの方向によって期間を調整（オプション）
        if er[i] > 0:  # 上昇トレンド
            # 上限バンドの期間を短くして反応性を高め、下限バンドの期間を長くして安定性を確保
            upper_period = max(3, int(curr_period * 0.7))
            lower_period = curr_period
        elif er[i] < 0:  # 下降トレンド
            # 下限バンドの期間を短くして反応性を高め、上限バンドの期間を長くして安定性を確保
            upper_period = curr_period
            lower_period = max(3, int(curr_period * 0.7))
        else:
            # トレンドなしの場合は両方同じ期間を使用
            upper_period = curr_period
            lower_period = curr_period
        
        # 上限バンドの計算（過去upper_period期間の最高値）
        if i >= upper_period:
            window_high = high[i-upper_period+1:i+1]
            if len(window_high) > 0:
                upper[i] = np.max(window_high)
                
        # 下限バンドの計算（過去lower_period期間の最安値）
        if i >= lower_period:
            window_low = low[i-lower_period+1:i+1]
            if len(window_low) > 0:
                lower[i] = np.min(window_low)
                
        # 中央線の計算
        if not np.isnan(upper[i]) and not np.isnan(lower[i]):
            middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


class ZDonchian(Indicator):
    """
    Z_Donchian（Z Donchian Channel）インジケーター
    
    特徴:
    - サイクル効率比（CER）に基づいて期間を動的に調整
    - 最大・最小期間それぞれに独立したEhlersHoDyDCを使用
    - トレンドの強さと方向に応じた適応性
    
    使用方法:
    - トレンドの強さに応じた動的なサポート/レジスタンスレベルの識別
    - ブレイクアウト/ブレイクダウンのエントリーシグナル
    - トレンド相場での損切り/利益確定レベルの設定
    - チャネルの幅から市場のボラティリティを評価
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,          # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 144,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 13,              # 最大期間用ドミナントサイクル計算用
        max_dc_max_output: int = 89,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_output: int = 21,             # 最大期間用ドミナントサイクル計算用
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 55,               # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 5,                # 最小期間用ドミナントサイクル計算用
        min_dc_max_output: int = 21,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_output: int = 8                # 最小期間用ドミナントサイクル計算用
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_dc_min_cycle: 最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 13）
            max_dc_max_output: 最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 89）
            max_dc_min_output: 最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 21）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_dc_min_cycle: 最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_dc_max_output: 最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 21）
            min_dc_min_output: 最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 8）
        """
        super().__init__(
            f"ZDonchian({max_dc_max_output}, {min_dc_max_output})"
        )
        # 最大期間用パラメータ
        self.max_dc_cycle_part = max_dc_cycle_part
        self.max_dc_max_cycle = max_dc_max_cycle
        self.max_dc_min_cycle = max_dc_min_cycle
        self.max_dc_max_output = max_dc_max_output
        self.max_dc_min_output = max_dc_min_output
        
        # 最小期間用パラメータ
        self.min_dc_cycle_part = min_dc_cycle_part
        self.min_dc_max_cycle = min_dc_max_cycle
        self.min_dc_min_cycle = min_dc_min_cycle
        self.min_dc_max_output = min_dc_max_output
        self.min_dc_min_output = min_dc_min_output
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
        # 最大期間用ドミナントサイクル検出器を初期化
        self.max_dc_detector = EhlersHoDyDC(
            cycle_part=self.max_dc_cycle_part,
            max_cycle=self.max_dc_max_cycle,
            min_cycle=self.max_dc_min_cycle,
            max_output=self.max_dc_max_output,
            min_output=self.max_dc_min_output,
            src_type='hlc3'
        )
        
        # 最小期間用ドミナントサイクル検出器を初期化
        self.min_dc_detector = EhlersHoDyDC(
            cycle_part=self.min_dc_cycle_part,
            max_cycle=self.min_dc_max_cycle,
            min_cycle=self.min_dc_min_cycle,
            max_output=self.min_dc_max_output,
            min_output=self.min_dc_min_output,
            src_type='hlc3'
        )
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = (
            f"{self.max_dc_cycle_part}_{self.max_dc_max_cycle}_{self.max_dc_min_cycle}_"
            f"{self.max_dc_max_output}_{self.max_dc_min_output}_"
            f"{self.min_dc_cycle_part}_{self.min_dc_max_cycle}_{self.min_dc_min_cycle}_"
            f"{self.min_dc_max_output}_{self.min_dc_min_output}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z_Donchianを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'カラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            中央線（ミドルバンド）の値
        """
        try:
            # サイクル効率比（CER）の検証
            if external_er is None:
                raise ValueError("サイクル効率比（CER）は必須です。external_erパラメータを指定してください")
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.middle
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low']):
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = np.asarray(data[:, 1], dtype=np.float64)  # high
                    low = np.asarray(data[:, 2], dtype=np.float64)   # low
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # 最大期間用ドミナントサイクルの計算
            max_dc_values = self.max_dc_detector.calculate(data)
            
            # 最小期間用ドミナントサイクルの計算
            min_dc_values = self.min_dc_detector.calculate(data)
            
            # 最大ドンチャン期間の最大値を取得（計算開始位置用）
            max_period_value = int(np.nanmax(max_dc_values))
            if np.isnan(max_period_value) or max_period_value < 10:
                max_period_value = 89  # デフォルト値
            
            # データ長の検証
            data_length = len(high)
            if data_length < max_period_value:
                raise ValueError(f"データ長({data_length})が必要な期間よりも短いです")
            
            # サイクル効率比（CER）を使用
            er = np.asarray(external_er, dtype=np.float64)
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # 動的なドンチャン期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                max_dc_values,
                min_dc_values
            )
            
            # Z_Donchianの計算（並列版）
            upper, lower, middle = calculate_z_donchian(
                high,
                low,
                er,
                dynamic_period,
                max_period_value
            )
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = ZDonchianResult(
                upper=np.copy(upper),
                lower=np.copy(lower),
                middle=np.copy(middle),
                er=np.copy(er),
                dynamic_period=np.copy(dynamic_period),
                dc_values=np.copy(max_dc_values)  # 最大期間用DCを保存
            )
            
            self._values = middle  # 標準インジケーターインターフェース用
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Z_Donchian計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時は前回の結果を維持
            if self._result is None:
                return np.array([])
            return self._result.middle
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        上限、下限、中央線の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 上限（アッパーバンド）、下限（ロワーバンド）、中央線（ミドルバンド）
        """
        if self._result is None:
            return np.array([]), np.array([]), np.array([])
        return self._result.upper, self._result.lower, self._result.middle
    
    def get_max_dc_values(self) -> np.ndarray:
        """
        最大期間用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 最大期間用ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_min_dc_values(self) -> np.ndarray:
        """
        最小期間用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 最小期間用ドミナントサイクルの値
        """
        # 最小期間用DCは結果に保存していないため、再計算する必要がある
        return self.min_dc_detector.get_values()
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的ドンチャン期間の値を取得する
        
        Returns:
            np.ndarray: 動的ドンチャン期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_period
    
    def get_channel_width(self) -> np.ndarray:
        """
        チャネル幅を取得する
        
        Returns:
            np.ndarray: 上限と下限の差（チャネル幅）
        """
        if self._result is None:
            return np.array([])
        return self._result.upper - self._result.lower
    
    def get_percent_channel_width(self, reference_price: np.ndarray) -> np.ndarray:
        """
        参照価格に対する相対的なチャネル幅を取得する
        
        Args:
            reference_price: 参照価格（通常は終値）
            
        Returns:
            np.ndarray: 相対的なチャネル幅（%）
        """
        if self._result is None or reference_price is None or len(reference_price) == 0:
            return np.array([])
        
        channel_width = self.get_channel_width()
        if len(channel_width) != len(reference_price):
            raise ValueError("参照価格の長さがチャネル幅と一致しません")
        
        # ゼロ除算を避ける
        result = np.zeros_like(channel_width)
        mask = reference_price != 0
        result[mask] = (channel_width[mask] / reference_price[mask]) * 100
        
        return result
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.max_dc_detector.reset()
        self.min_dc_detector.reset() 