#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_ma import AlphaMA
from .alpha_volatility import AlphaVolatility
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaVolBandResult:
    """AlphaVolBandの計算結果"""
    middle: np.ndarray        # 中心線（AlphaMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    er: np.ndarray            # Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ボラティリティ乗数
    alpha_vol: np.ndarray     # AlphaVolatility値（%ベース）
    alpha_vol_abs: np.ndarray # AlphaVolatility値（金額ベース）


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    効率比に基づいて動的なボラティリティ乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # ERが高い（トレンドが強い）ほど乗数は小さく、
    # ERが低い（トレンドが弱い）ほど乗数は大きくなる
    multipliers = max_mult - er * (max_mult - min_mult)
    return multipliers


@jit(nopython=True)
def calculate_alpha_vol_band(
    alpha_ma: np.ndarray,
    alpha_vol: np.ndarray,
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファボラティリティバンドを計算する（高速化版）
    
    Args:
        alpha_ma: AlphaMA値の配列
        alpha_vol: AlphaVolatility値の配列（use_percentに応じて%または金額ベース）
        dynamic_multiplier: 動的乗数の配列
        use_percent: パーセンテージベースのボラティリティを使用するかどうか
                    True: 中心値に対する割合として使用（バンド幅 = MA * Vol% * 乗数）
                    False: 金額値をそのまま使用（バンド幅 = Vol金額 * 乗数）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(alpha_ma)
    middle = np.copy(alpha_ma)
    upper = np.full_like(alpha_ma, np.nan)
    lower = np.full_like(alpha_ma, np.nan)
    
    # 単純なループで処理（Numba対応）
    for i in range(length):
        # NaN以外のときだけ計算
        if not (np.isnan(alpha_ma[i]) or np.isnan(alpha_vol[i]) or np.isnan(dynamic_multiplier[i])):
            if use_percent:
                # %ベースのボラティリティを使用する場合、中心値に対する割合で計算
                # ボラティリティは小数点表記（0.01 = 1%）なので、そのまま掛ける
                band_width = alpha_ma[i] * alpha_vol[i] * dynamic_multiplier[i]
            else:
                # 金額ベースのボラティリティをそのまま使用
                band_width = alpha_vol[i] * dynamic_multiplier[i]
            
            upper[i] = middle[i] + band_width
            lower[i] = middle[i] - band_width
    
    return middle, upper, lower


class AlphaVolBand(Indicator):
    """
    アルファボラティリティバンド（Alpha Volatility Band）インジケーター
    
    特徴:
    - 中心線にAlphaMA（動的適応型移動平均）を使用
    - バンド幅の計算にAlphaVolatility（動的適応型ボラティリティ）を使用
    - ボラティリティ乗数が効率比（ER）に基づいて動的に調整
    - ハイパースムーサーによる平滑化アルゴリズムをAlphaVolatilityで使用
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（ER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    
    AlphaKeltnerChannelとの違い:
    - ATR（Average True Range）の代わりに標準偏差ベースのボラティリティを使用
    - より長期的な価格変動の振れ幅を捉えることができる
    - HLCデータだけでなく、closeデータのみでも機能する
    
    ボラティリティの使用方法:
    - 金額ベース（デフォルト）: 実際の価格単位でのボラティリティを使用
      ボラティリティが価格変動の絶対的な大きさを直接反映するため、
      大きな価格変動があるときにバンド幅が適切に拡大する
    
    - パーセントベース（オプション）: 価格に対する相対的なボラティリティを使用
      異なる価格レベルでの比較や、長期間のバックテストに有用だが、
      価格が大きく上昇/下落した場合にバンド幅が不自然に拡大/縮小する可能性がある
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        max_vol_period: int = 89,
        min_vol_period: int = 13,
        smoothing_period: int = 14,
        max_multiplier: float = 6.0,
        min_multiplier: float = 3.0,
        use_percent_vol: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 55）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 8）
            max_vol_period: AlphaVolatilityの最大期間（デフォルト: 89）
            min_vol_period: AlphaVolatilityの最小期間（デフォルト: 13）
            smoothing_period: ボラティリティの平滑化期間（デフォルト: 14）
            max_multiplier: ボラティリティ乗数の最大値（デフォルト: 6.0）
            min_multiplier: ボラティリティ乗数の最小値（デフォルト: 3.0）
            use_percent_vol: パーセンテージベースのボラティリティを使用するかどうか（デフォルト: False）
                            True: 中心値に対する割合のボラティリティを使用
                            False: 金額ベースのボラティリティを使用（デフォルト）
        
        注意:
            金額ベースのボラティリティ（use_percent_vol=False）を使用する場合、
            max_multiplierとmin_multiplierの値は通常、パーセントベースよりも
            小さい値（例: 1.0〜3.0程度）を使用することが推奨されます。
            これは、金額ベースのボラティリティがすでに価格スケールを持っているためです。
        """
        super().__init__(
            f"AlphaVolBand({er_period}, {max_multiplier}, {min_multiplier})"
        )
        self.er_period = er_period
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.max_vol_period = max_vol_period
        self.min_vol_period = min_vol_period
        self.smoothing_period = smoothing_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.use_percent_vol = use_percent_vol
        
        # AlphaMAとAlphaVolatilityのインスタンス化
        self.alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period
        )
        
        self.alpha_volatility = AlphaVolatility(
            er_period=er_period,
            max_vol_period=max_vol_period,
            min_vol_period=min_vol_period,
            smoothing_period=smoothing_period
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファボラティリティバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            中心線の値（AlphaMA）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    close = data[:, 3]  # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            
            # AlphaMAの計算
            alpha_ma_values = self.alpha_ma.calculate(data)
            if alpha_ma_values is None:
                raise ValueError("AlphaMAの計算に失敗しました")
            
            # AlphaVolatilityの計算
            alpha_vol_values = self.alpha_volatility.calculate(data)
            if alpha_vol_values is None:
                raise ValueError("AlphaVolatilityの計算に失敗しました")
            
            # 効率比（ER）の取得（AlphaMAから）
            er = self.alpha_ma.get_efficiency_ratio()
            
            # 動的ボラティリティ乗数の計算
            dynamic_multiplier = calculate_dynamic_multiplier(
                er,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # 使用するボラティリティ値を決定
            if self.use_percent_vol:
                # ここではそのままのVolatility値（小数表記）を使用
                # 表示用に100倍するのはget_metricsメソッドで行う
                vol_for_calc = alpha_vol_values
                vol_abs_for_result = self.alpha_volatility.get_absolute_volatility()
            else:
                # 金額ベースのボラティリティ値
                vol_for_calc = self.alpha_volatility.get_absolute_volatility()
                vol_abs_for_result = vol_for_calc
            
            # アルファボラティリティバンドの計算
            middle, upper, lower = calculate_alpha_vol_band(
                alpha_ma_values,
                vol_for_calc,
                dynamic_multiplier,
                self.use_percent_vol
            )
            
            # 結果の保存
            self._result = AlphaVolBandResult(
                middle=middle,
                upper=upper,
                lower=lower,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                alpha_vol=alpha_vol_values,  # 元のスケール（小数表記）
                alpha_vol_abs=vol_abs_for_result  # 金額ベース
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            self.logger.error(f"AlphaVolBand計算中にエラー: {str(e)}")
            
            # エラー時でも最低限の結果を初期化
            if 'close' in locals():
                length = len(close)
                empty_array = np.full(length, np.nan)
                self._result = AlphaVolBandResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    er=empty_array,
                    dynamic_multiplier=empty_array,
                    alpha_vol=empty_array,
                    alpha_vol_abs=empty_array
                )
                self._values = empty_array
                return empty_array
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ボラティリティバンドの各バンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
    def get_dynamic_multiplier(self) -> np.ndarray:
        """
        動的ボラティリティ乗数の値を取得する
        
        Returns:
            np.ndarray: 動的ボラティリティ乗数の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.dynamic_multiplier
    
    def get_percent_volatility(self) -> np.ndarray:
        """
        %ベースのボラティリティ値を取得する
        
        Returns:
            np.ndarray: %ベースのボラティリティ値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        # 表示用に100倍する
        return self._result.alpha_vol * 100
    
    def get_absolute_volatility(self) -> np.ndarray:
        """
        金額ベースのボラティリティ値を取得する
        
        Returns:
            np.ndarray: 金額ベースのボラティリティ値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.alpha_vol_abs
    
    def get_all_metrics(self) -> dict:
        """
        すべての計算結果を辞書形式で取得する
        
        Returns:
            dict: 各種指標の値
        """
        if self._result is None:
            return {}
        
        return {
            'middle': self._result.middle,
            'upper': self._result.upper,
            'lower': self._result.lower,
            'er': self._result.er,
            'dynamic_multiplier': self._result.dynamic_multiplier,
            'percent_volatility': self.get_percent_volatility(),
            'absolute_volatility': self._result.alpha_vol_abs
        } 