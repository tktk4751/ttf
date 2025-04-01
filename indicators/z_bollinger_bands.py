#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .z_ma import ZMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZBollingerBandsResult:
    """ZBollingerBandsの計算結果"""
    middle: np.ndarray        # 中心線（ZMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cer: np.ndarray           # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的標準偏差乗数
    std_dev: np.ndarray       # 標準偏差
    max_dc_values: np.ndarray  # 最大サイクル期間値
    min_dc_values: np.ndarray  # 最小サイクル期間値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_multiplier_vec(cer: float, max_mult: float, min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な標準偏差乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    return max_mult - cer_abs * (max_mult - min_mult)


@njit(fastmath=True)
def calculate_dynamic_multiplier(cer: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的な標準偏差乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    cer_abs = np.abs(cer)
    multipliers = max_mult - cer_abs * (max_mult - min_mult)
    return multipliers


@njit(fastmath=True)
def calculate_rolling(prices: np.ndarray, period: np.ndarray) -> np.ndarray:
    """
    動的期間を用いたローリング標準偏差を計算する（高速化版）
    
    Args:
        prices: 価格の配列
        period: 動的な計算期間の配列（整数値）
    
    Returns:
        ローリング標準偏差の配列
    """
    length = len(prices)
    std_dev = np.full(length, np.nan, dtype=np.float64)
    
    # 各位置で動的な期間を使用して標準偏差を計算
    for i in range(length):
        curr_period = int(period[i])
        if np.isnan(curr_period) or curr_period < 2 or i < curr_period:
            continue
            
        # 現在のウィンドウを取得
        window = prices[i-curr_period+1:i+1]
        
        # 平均を計算
        window_sum = 0.0
        for j in range(curr_period):
            window_sum += window[j]
        window_mean = window_sum / curr_period
        
        # 二乗差の合計を計算
        sum_sq_diff = 0.0
        for j in range(curr_period):
            diff = window[j] - window_mean
            sum_sq_diff += diff * diff
        
        # 標準偏差を計算
        std_dev[i] = np.sqrt(sum_sq_diff / curr_period)
    
    return std_dev


@njit(fastmath=True, parallel=True)
def calculate_z_bollinger_bands(
    z_ma: np.ndarray,
    std_dev: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zボリンジャーバンドを計算する（パラレル高速化版）
    
    Args:
        z_ma: ZMA値の配列
        std_dev: 標準偏差の配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(z_ma)
    middle = np.copy(z_ma)
    upper = np.full_like(z_ma, np.nan, dtype=np.float64)
    lower = np.full_like(z_ma, np.nan, dtype=np.float64)
    
    # 有効なデータ判別用マスク
    valid_mask = ~(np.isnan(z_ma) | np.isnan(std_dev) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列
    band_width = np.zeros_like(z_ma, dtype=np.float64)
    
    # 並列計算のためバンド幅を先に計算
    for i in prange(length):
        if valid_mask[i]:
            band_width[i] = std_dev[i] * dynamic_multiplier[i]
    
    # 上下バンドを計算
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class ZBollingerBands(Indicator):
    """
    Zボリンジャーバンド（Z Bollinger Bands）インジケーター
    
    特徴:
    - 中心線にZMA（Z Moving Average）を使用
    - 標準偏差の計算期間をドミナントサイクルに基づいて動的に調整
    - 標準偏差の乗数がサイクル効率比（CER）に基づいて動的に調整
    
    市場状態に応じた最適な挙動:
    - トレンド強い（CER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（CER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 2.5,
        min_multiplier: float = 1.0,
        max_cycle_part: float = 0.5,    
        max_max_cycle: int = 144,       
        max_min_cycle: int = 10,        
        max_max_output: int = 89,       
        max_min_output: int = 13,       
        min_cycle_part: float = 0.25,   
        min_max_cycle: int = 55,        
        min_min_cycle: int = 5,         
        min_max_output: int = 21,       
        min_min_output: int = 5,        
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機（デフォルト）
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_multiplier: 標準偏差乗数の最大値（デフォルト: 2.5）
            min_multiplier: 標準偏差乗数の最小値（デフォルト: 1.0）
            max_cycle_part: 標準偏差最大期間用サイクル部分（デフォルト: 0.5）
            max_max_cycle: 標準偏差最大期間用最大サイクル（デフォルト: 144）
            max_min_cycle: 標準偏差最大期間用最小サイクル（デフォルト: 10）
            max_max_output: 標準偏差最大期間用最大出力値（デフォルト: 34）
            max_min_output: 標準偏差最大期間用最小出力値（デフォルト: 10）
            min_cycle_part: 標準偏差最小期間用サイクル部分（デフォルト: 0.25）
            min_max_cycle: 標準偏差最小期間用最大サイクル（デフォルト: 55）
            min_min_cycle: 標準偏差最小期間用最小サイクル（デフォルト: 5）
            min_max_output: 標準偏差最小期間用最大出力値（デフォルト: 14）
            min_min_output: 標準偏差最小期間用最小出力値（デフォルト: 5）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        super().__init__(
            f"ZBB({cycle_detector_type}, {max_multiplier}, {min_multiplier}, {src_type})"
        )
        
        # 基本パラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.src_type = src_type
        
        # 標準偏差計算用パラメータ
        self.max_cycle_part = max_cycle_part
        self.max_max_cycle = max_max_cycle
        self.max_min_cycle = max_min_cycle
        self.max_max_output = max_max_output
        self.max_min_output = max_min_output
        
        self.min_cycle_part = min_cycle_part
        self.min_max_cycle = min_max_cycle
        self.min_min_cycle = min_min_cycle
        self.min_max_output = min_max_output
        self.min_min_output = min_min_output
        
        # サイクル効率比（CER）のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type
        )
        
        # ZMAのインスタンス化
        self.z_ma = ZMA(
            # 最大サイクル設定
            max_dc_cycle_part=max_cycle_part,
            max_dc_max_cycle=max_max_cycle,
            max_dc_min_cycle=max_min_cycle,
            max_dc_max_output=max_max_output,
            max_dc_min_output=max_min_output,
            
            # 最小サイクル設定
            min_dc_cycle_part=min_cycle_part,
            min_dc_max_cycle=min_max_cycle,
            min_dc_min_cycle=min_min_cycle,
            min_dc_max_output=min_max_output,
            min_dc_min_output=min_min_output,
            
            src_type=src_type
        )
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low', 'close'] if self.src_type in ['hlc3', 'hl2', 'ohlc4'] else ['close']
            cols += ['open'] if self.src_type == 'ohlc4' else []
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = (
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.max_multiplier}_{self.min_multiplier}_{self.src_type}_"
            f"{self.max_cycle_part}_{self.max_max_output}_{self.max_min_output}_"
            f"{self.min_cycle_part}_{self.min_max_output}_{self.min_min_output}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zボリンジャーバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            中心線の値（ZMA）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.middle
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # サイクル効率比（CER）の計算
            cer = self.cycle_er.calculate(data)
            if cer is None or len(cer) == 0:
                raise ValueError("サイクル効率比（CER）の計算に失敗しました")
            
            # ZMAの計算（サイクル効率比を使用）
            z_ma_values = self.z_ma.calculate(data, external_er=cer)
            if z_ma_values is None or len(z_ma_values) == 0:
                raise ValueError("ZMAの計算に失敗しました")
            
            # ソースデータの取得
            prices = self.calculate_source_values(data, self.src_type)
            
            # ドミナントサイクル値の取得（ZMAの内部検出器から）
            max_dc_values = self.z_ma.get_max_dc_values()
            min_dc_values = self.z_ma.get_min_dc_values()
            
            # 動的な標準偏差期間を計算（ドミナントサイクルを使用）
            # 最大値と最小値の間で調整
            dynamic_period = np.round(max_dc_values).astype(np.int32)
            
            # 標準偏差の計算（動的な期間を使用）
            std_dev = calculate_rolling(prices, dynamic_period)
            
            # 動的標準偏差乗数の計算（ベクトル化関数を使用）
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # Zボリンジャーバンドの計算（パラレル高速版）
            middle, upper, lower = calculate_z_bollinger_bands(
                z_ma_values,
                std_dev,
                dynamic_multiplier
            )
            
            # 結果の保存（参照コピーを避けるためnp.copyを使用）
            self._result = ZBollingerBandsResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                cer=np.copy(cer),
                dynamic_multiplier=np.copy(dynamic_multiplier),
                std_dev=np.copy(std_dev),
                max_dc_values=np.copy(max_dc_values),
                min_dc_values=np.copy(min_dc_values)
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZBollingerBands計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は前回の結果を維持する（nullではなく）
            if self._result is None:
                # 初回エラー時は空の結果を作成
                empty_array = np.array([])
                self._result = ZBollingerBandsResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    cer=empty_array,
                    dynamic_multiplier=empty_array,
                    std_dev=empty_array,
                    max_dc_values=empty_array,
                    min_dc_values=empty_array
                )
                self._values = empty_array
            
            return self._values
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ボリンジャーバンドのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_cycle_er(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.cer
    
    def get_dynamic_multiplier(self) -> np.ndarray:
        """
        動的標準偏差乗数の値を取得する
        
        Returns:
            np.ndarray: 動的標準偏差乗数の値
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_multiplier
    
    def get_standard_deviation(self) -> np.ndarray:
        """
        標準偏差の値を取得する
        
        Returns:
            np.ndarray: 標準偏差の値
        """
        if self._result is None:
            return np.array([])
        return self._result.std_dev
    
    def get_dominant_cycle_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ドミナントサイクルの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (最大サイクル値, 最小サイクル値)
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        return self._result.max_dc_values, self._result.min_dc_values
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.cycle_er.reset() if hasattr(self.cycle_er, 'reset') else None
        self.z_ma.reset() if hasattr(self.z_ma, 'reset') else None 
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
        """
        与えられたソースタイプに基づいて価格値を計算する
        
        Args:
            data: 価格データ
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        
        Returns:
            np.ndarray: 選択されたソースタイプに基づく価格値
        """
        if isinstance(data, pd.DataFrame):
            if src_type == 'close' and 'close' in data.columns:
                return data['close'].values
            elif src_type == 'hlc3' and all(col in data.columns for col in ['high', 'low', 'close']):
                return (data['high'].values + data['low'].values + data['close'].values) / 3.0
            elif src_type == 'hl2' and all(col in data.columns for col in ['high', 'low']):
                return (data['high'].values + data['low'].values) / 2.0
            elif src_type == 'ohlc4' and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                return (data['open'].values + data['high'].values + data['low'].values + data['close'].values) / 4.0
            else:
                raise ValueError(f"選択されたソースタイプ ({src_type}) に必要なカラムがDataFrameにありません")
        else:
            # NumPy配列の場合はカラムの位置に基づいて選択
            if data.ndim != 2:
                raise ValueError("NumPy配列は2次元である必要があります")
            if src_type == 'close':
                return data[:, 3] if data.shape[1] >= 4 else data[:, 0]
            elif src_type == 'hlc3':
                return (data[:, 1] + data[:, 2] + data[:, 3]) / 3.0 if data.shape[1] >= 4 else data[:, 0]
            elif src_type == 'hl2':
                return (data[:, 1] + data[:, 2]) / 2.0 if data.shape[1] >= 3 else data[:, 0]
            elif src_type == 'ohlc4':
                return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4.0 if data.shape[1] >= 4 else data[:, 0]
            else:
                raise ValueError(f"無効なソースタイプ: {src_type}")

    def calculate_dc_values(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        cycle_part: float,
        max_cycle: int,
        min_cycle: int,
        max_output: int,
        min_output: int
    ) -> np.ndarray:
        """
        ドミナントサイクル値を計算する
        
        Args:
            data: 価格データ
            cycle_part: サイクル部分
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
            max_output: 最大出力値
            min_output: 最小出力値
        
        Returns:
            np.ndarray: ドミナントサイクル値
        """
        # サイクル検出器とパラメータを使用してサイクル値を計算
        from .ehlers_hody_dc import EhlersHoDyDC
        
        dc = EhlersHoDyDC(
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=self.src_type
        )
        
        # データからソース値を取得
        prices = self.calculate_source_values(data, self.src_type)
        
        # サイクル値を計算
        return dc.calculate(data)

    def calculate_rolling(self, prices: np.ndarray, period: np.ndarray) -> np.ndarray:
        """
        動的期間を用いたローリング標準偏差を計算する
        
        Args:
            prices: 価格の配列
            period: 動的な計算期間の配列（整数値）
        
        Returns:
            np.ndarray: ローリング標準偏差の配列
        """
        # クラス内の標準偏差計算関数をラップ
        return calculate_rolling(prices, period)

    def calculate_dynamic_multiplier(self, cer: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
        """
        動的な乗数を計算する
        
        Args:
            cer: サイクル効率比の配列
            max_mult: 最大乗数
            min_mult: 最小乗数
        
        Returns:
            np.ndarray: 動的乗数の配列
        """
        # クラス内の乗数計算関数をラップ
        return calculate_dynamic_multiplier(cer, max_mult, min_mult)