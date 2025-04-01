#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import calculate_hyper_smoother_numba


@dataclass
class AlphaMAResult:
    """AlphaMAの計算結果"""
    values: np.ndarray        # AlphaMAの値（スムージング済み）
    raw_values: np.ndarray    # 生のAlphaMA値（スムージング前）
    er: np.ndarray           # サイクル効率比（CER）
    dynamic_kama_period: np.ndarray  # 動的KAMAピリオド
    dynamic_fast_period: np.ndarray  # 動的Fast期間
    dynamic_slow_period: np.ndarray  # 動的Slow期間


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_kama_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なKAMAピリオドを計算する（ベクトル化版）
    
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
    return np.round(min_period + (1.0 - abs(er)) * (max_period - min_period))


@njit(fastmath=True, parallel=True)
def calculate_dynamic_kama_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なKAMAピリオドを計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = np.abs(er)
    periods = min_period + (1.0 - er_abs) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True)
def calculate_dynamic_kama_constants(er: np.ndarray, 
                                    max_slow: int, min_slow: int,
                                    max_fast: int, min_fast: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    効率比に基づいて動的なKAMAのfast/slow期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_slow: 遅い移動平均の最大期間
        min_slow: 遅い移動平均の最小期間
        max_fast: 速い移動平均の最大期間
        min_fast: 速い移動平均の最小期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            動的なfast期間の配列、動的なslow期間の配列、fastの定数、slowの定数
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = np.abs(er)
    fast_periods = min_fast + (1.0 - er_abs) * (max_fast - min_fast)
    slow_periods = min_slow + (1.0 - er_abs) * (max_slow - min_slow)
    
    fast_periods_rounded = np.round(fast_periods).astype(np.int32)
    slow_periods_rounded = np.round(slow_periods).astype(np.int32)
    
    # 定数の計算
    fast_constants = 2.0 / (fast_periods + 1.0)
    slow_constants = 2.0 / (slow_periods + 1.0)
    
    return fast_periods_rounded, slow_periods_rounded, fast_constants, slow_constants


@njit(fastmath=True)
def calculate_alpha_ma(prices: np.ndarray, er: np.ndarray, er_period: int,
                      kama_period: np.ndarray,
                      fast_constants: np.ndarray, slow_constants: np.ndarray) -> np.ndarray:
    """
    AlphaMAを計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        er: 効率比の配列（ERまたはCER）
        er_period: 効率比の計算期間
        kama_period: 動的なKAMAピリオドの配列
        fast_constants: 速い移動平均の定数配列
        slow_constants: 遅い移動平均の定数配列
    
    Returns:
        AlphaMAの配列
    """
    length = len(prices)
    alpha_ma = np.full(length, np.nan, dtype=np.float64)
    
    # 最初のAlphaMAは最初の価格
    if length > 0:
        alpha_ma[0] = prices[0]
    
    # 各時点でのAlphaMAを計算
    for i in range(1, length):
        if np.isnan(er[i]) or i < er_period:
            alpha_ma[i] = alpha_ma[i-1]
            continue
        
        # 現在の動的パラメータを取得
        curr_kama_period = int(kama_period[i])
        if curr_kama_period < 1:
            curr_kama_period = 1
        
        # 現在の時点での実際のER値を計算
        if i >= curr_kama_period:
            # 変化とボラティリティの計算
            change = prices[i] - prices[i-curr_kama_period]
            volatility = np.sum(np.abs(np.diff(prices[max(0, i-curr_kama_period):i+1])))
            
            # ゼロ除算を防止
            current_er = 0.0 if volatility < 1e-10 else abs(change) / volatility
            
            # スムージング定数の計算
            smoothing_constant = (current_er * (fast_constants[i] - slow_constants[i]) + slow_constants[i]) ** 2
            
            # 0-1の範囲に制限
            smoothing_constant = max(0.0, min(1.0, smoothing_constant))
            
            # AlphaMAの計算
            alpha_ma[i] = alpha_ma[i-1] + smoothing_constant * (prices[i] - alpha_ma[i-1])
        else:
            alpha_ma[i] = alpha_ma[i-1]
    
    return alpha_ma


class AlphaMA(Indicator):
    """
    AlphaMA (Alpha Moving Average) インジケーター
    
    サイクル効率比（CER）に基づいて以下のパラメータを動的に調整する適応型移動平均線：
    - KAMAピリオド自体
    - KAMAのfast期間
    - KAMAのslow期間
    
    特徴:
    - すべてのパラメータがサイクル効率比に応じて動的に調整される
    - トレンドが強い時：短いピリオドと速い反応
    - レンジ相場時：長いピリオドとノイズ除去
    - オプションのハイパースムーサーによる追加平滑化
    """
    
    def __init__(
        self,
        max_kama_period: int = 144,
        min_kama_period: int = 10,
        max_slow_period: int = 34,
        min_slow_period: int = 9,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        hyper_smooth_period: int = 0,
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            max_kama_period: KAMAピリオドの最大値（デフォルト: 144）
            min_kama_period: KAMAピリオドの最小値（デフォルト: 10）
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 34）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 9）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 8）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
            hyper_smooth_period: ハイパースムーサーの平滑化期間（デフォルト: 0、0以下の場合は平滑化しない）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(
            f"AlphaMA({max_kama_period}, {min_kama_period}, "
            f"{max_slow_period}, {min_slow_period}, {max_fast_period}, {min_fast_period}, {src_type})"
        )
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        self.hyper_smooth_period = hyper_smooth_period
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # closeカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合はcloseだけハッシュ
                data_hash = hash(tuple(data[:, 3]))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = f"{self.max_kama_period}_{self.min_kama_period}_{self.max_slow_period}_{self.min_slow_period}_{self.max_fast_period}_{self.min_fast_period}_{self.hyper_smooth_period}_{self.src_type}_{external_er_hash}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        AlphaMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            AlphaMAの値（ハイパースムーサーで平滑化されている場合はその結果、そうでなければ生のAlphaMA値）
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
            
            # 指定されたソースタイプの価格データを取得
            prices = self.calculate_source_values(data, self.src_type)
            
            # データ長の検証（簡略化）
            data_length = len(prices)
            
            # サイクル効率比（CER）を使用
            er = external_er
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # 動的なKAMAピリオドの計算（ベクトル化関数を使用）
            dynamic_kama_period = calculate_dynamic_kama_period_vec(
                er,
                self.max_kama_period,
                self.min_kama_period
            ).astype(np.int32)
            
            # 動的なfast/slow期間の計算（パラレル処理版）
            fast_periods, slow_periods, fast_constants, slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            # AlphaMAの計算（高速化版）
            alpha_ma_raw = calculate_alpha_ma(
                prices,
                er,
                len(er) // 10,  # 近似値として使用
                dynamic_kama_period,
                fast_constants,
                slow_constants
            )
            
            # ハイパースムーサーによる平滑化（オプション）
            if self.hyper_smooth_period > 0:
                # ハイパースムーサーを適用
                alpha_ma_smoothed = calculate_hyper_smoother_numba(alpha_ma_raw, self.hyper_smooth_period)
                
                # 出力用に平滑化済み値を使用
                alpha_ma_values = alpha_ma_smoothed
            else:
                # 平滑化しない場合は生の値をそのまま使用
                alpha_ma_values = alpha_ma_raw
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = AlphaMAResult(
                values=np.copy(alpha_ma_values),
                raw_values=np.copy(alpha_ma_raw),
                er=np.copy(er),
                dynamic_kama_period=np.copy(dynamic_kama_period),
                dynamic_fast_period=np.copy(fast_periods),
                dynamic_slow_period=np.copy(slow_periods)
            )
            
            self._values = alpha_ma_values
            return alpha_ma_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は前回の結果を維持する
            if self._result is None:
                # 初回エラー時は空の配列を返す
                return np.array([])
            return self._result.values
    
    def get_raw_values(self) -> np.ndarray:
        """
        ハイパースムーサーによる平滑化前の生AlphaMA値を取得する
        
        Returns:
            np.ndarray: 生のAlphaMA値（平滑化前）
        """
        if self._result is None:
            return np.array([])
        return self._result.raw_values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的な期間の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return (
            self._result.dynamic_kama_period,
            self._result.dynamic_fast_period,
            self._result.dynamic_slow_period
        )
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None 