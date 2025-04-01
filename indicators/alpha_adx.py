#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import hyper_smoother


@dataclass
class AlphaADXResult:
    """AlphaADXの計算結果"""
    values: np.ndarray          # AlphaADXの値（0-1の範囲で正規化）
    er: np.ndarray              # Efficiency Ratio
    plus_di: np.ndarray         # +DI値
    minus_di: np.ndarray        # -DI値
    dynamic_period: np.ndarray  # 動的ADX期間
    tr: np.ndarray              # True Range


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@jit(nopython=True)
def calculate_dm(high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Directional Movement（+DM, -DM）を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if (up > down) and (up > 0):
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        
        if (down > up) and (down > 0):
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    return plus_dm, minus_dm


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なADX期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_period + (1.0 - er) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_alpha_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    rsx_period: int,
    max_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファADXを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列
        dynamic_period: 動的期間の配列
        rsx_period: RSX平滑化期間
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ADX, +DI, -DI)の配列
    """
    length = len(high)
    adx = np.full(length, np.nan)
    plus_di = np.full(length, np.nan)
    minus_di = np.full(length, np.nan)
    
    # True Rangeの計算
    tr = calculate_tr(high, low, close)
    
    # +DM, -DMの計算
    plus_dm, minus_dm = calculate_dm(high, low)
    
    # 各時点での平滑化とDI/ADX計算
    for i in range(max_period, length):
        # その時点での動的な期間を取得
        curr_period = int(dynamic_period[i])
        if curr_period < 2:
            curr_period = 2
        
        # データ抽出範囲の決定（最低でもcurr_period*3要素を確保）
        data_start = max(0, i - curr_period * 3)
        
        # TR, +DM, -DMのサブセット抽出
        tr_subset = tr[data_start:i+1]
        plus_dm_subset = plus_dm[data_start:i+1]
        minus_dm_subset = minus_dm[data_start:i+1]
        
        # ハイパースムーサーによる平滑化
        if len(tr_subset) > 0:
            tr_smoothed = hyper_smoother(tr_subset, curr_period)
            plus_dm_smoothed = hyper_smoother(plus_dm_subset, curr_period)
            minus_dm_smoothed = hyper_smoother(minus_dm_subset, curr_period)
            
            # +DI, -DIの計算（ゼロ除算回避）
            if tr_smoothed[-1] > 0:
                plus_di[i] = 100.0 * plus_dm_smoothed[-1] / tr_smoothed[-1]
                minus_di[i] = 100.0 * minus_dm_smoothed[-1] / tr_smoothed[-1]
            else:
                plus_di[i] = 0.0
                minus_di[i] = 0.0
            
            # DXの計算
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
            else:
                dx = 0.0
            
            # 過去のDX値を収集
            dx_window = []
            for j in range(min(curr_period, i+1)):
                idx = i - j
                if idx >= 0 and not np.isnan(plus_di[idx]) and not np.isnan(minus_di[idx]):
                    di_sum_j = plus_di[idx] + minus_di[idx]
                    if di_sum_j > 0:
                        dx_window.append(100.0 * abs(plus_di[idx] - minus_di[idx]) / di_sum_j)
                    else:
                        dx_window.append(0.0)
            
            # DX値が十分にある場合のみADXを計算
            if len(dx_window) > 0:
                dx_array = np.array(dx_window)
                adx[i] = hyper_smoother(dx_array, min(curr_period, len(dx_array)))[-1]
    
    # 0-1の範囲に正規化
    adx = adx / 100.0
    plus_di = plus_di / 100.0
    minus_di = minus_di / 100.0
    
    return adx, plus_di, minus_di


class AlphaADX(Indicator):
    """
    アルファADX（Alpha Average Directional Index）インジケーター
    
    特徴:
    - 効率比（ER）に基づいて期間を動的に調整
    - EMAの代わりにALMA（Arnaud Legoux Moving Average）を使用してDI/DXを平滑化
    - 0-1の範囲で正規化
    
    解釈:
    - 0.5以上: 強いトレンド
    - 0.25以下: 弱いトレンド/レンジ相場
    - +DIが-DIを上回る: 上昇トレンド
    - -DIが+DIを上回る: 下降トレンド
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        rsx_period: int = 10
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_adx_period: ADX期間の最大値（デフォルト: 55）
            min_adx_period: ADX期間の最小値（デフォルト: 8）
            rsx_period: RSX平滑化期間（デフォルト: 10）
        """
        super().__init__(
            f"AlphaADX({er_period}, {max_adx_period}, {min_adx_period}, "
            f"{rsx_period})"
        )
        self.er_period = er_period
        self.max_adx_period = max_adx_period
        self.min_adx_period = min_adx_period
        self.rsx_period = rsx_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            アルファADXの値（0-1の範囲）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(high)
            
            # データ長が最小期間より小さい場合は、NaNで埋めた配列を返す
            if data_length < self.min_adx_period:
                self.logger.warning(f"データ長（{data_length}）が最小期間（{self.min_adx_period}）より小さいため、NaN配列を返します")
                result = np.full(data_length, np.nan)
                self._values = result
                
                # 結果の保存（すべてNaN）
                self._result = AlphaADXResult(
                    values=result,
                    er=np.full(data_length, np.nan),
                    plus_di=np.full(data_length, np.nan),
                    minus_di=np.full(data_length, np.nan),
                    dynamic_period=np.full(data_length, self.min_adx_period),
                    tr=np.full(data_length, np.nan)
                )
                
                return result
            
            # 期間の検証
            self._validate_period(self.er_period, data_length)
            
            # True Range (TR)の計算
            tr = calculate_tr(high, low, close)
            
            # +DM, -DMの計算
            plus_dm, minus_dm = calculate_dm(high, low)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, min(self.er_period, data_length - 1))
            
            # 動的なADX期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                min(self.max_adx_period, data_length - 1),
                min(self.min_adx_period, data_length - 1)
            )
            
            # 結果配列の初期化
            adx = np.full(data_length, np.nan)
            plus_di = np.full(data_length, np.nan)
            minus_di = np.full(data_length, np.nan)
            
            # 各時点での平滑化とDI/ADX計算
            for i in range(min(self.max_adx_period, data_length), data_length):
                # その時点での動的な期間を取得
                curr_period = int(dynamic_period[i])
                curr_period = max(2, min(curr_period, i))  # 期間が2未満またはiより大きくならないように制限
                
                # データ抽出範囲の決定（最低でもcurr_period*3要素を確保）
                data_start = max(0, i - curr_period * 3)
                
                # TR, +DM, -DMのサブセット抽出
                tr_subset = tr[data_start:i+1]
                plus_dm_subset = plus_dm[data_start:i+1]
                minus_dm_subset = minus_dm[data_start:i+1]
                
                # ハイパースムーサーによる平滑化
                if len(tr_subset) > 0:
                    tr_smoothed = hyper_smoother(tr_subset, min(curr_period, len(tr_subset)))
                    plus_dm_smoothed = hyper_smoother(plus_dm_subset, min(curr_period, len(plus_dm_subset)))
                    minus_dm_smoothed = hyper_smoother(minus_dm_subset, min(curr_period, len(minus_dm_subset)))
                    
                    # +DI, -DIの計算（ゼロ除算回避）
                    if tr_smoothed[-1] > 0:
                        plus_di[i] = 100.0 * plus_dm_smoothed[-1] / tr_smoothed[-1]
                        minus_di[i] = 100.0 * minus_dm_smoothed[-1] / tr_smoothed[-1]
                    else:
                        plus_di[i] = 0.0
                        minus_di[i] = 0.0
                    
                    # DXの計算
                    di_sum = plus_di[i] + minus_di[i]
                    if di_sum > 0:
                        dx = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
                    else:
                        dx = 0.0
                    
                    # 過去のDX値を収集
                    dx_window = []
                    for j in range(min(curr_period, i+1)):
                        idx = i - j
                        if idx >= 0 and not np.isnan(plus_di[idx]) and not np.isnan(minus_di[idx]):
                            di_sum_j = plus_di[idx] + minus_di[idx]
                            if di_sum_j > 0:
                                dx_window.append(100.0 * abs(plus_di[idx] - minus_di[idx]) / di_sum_j)
                            else:
                                dx_window.append(0.0)
                    
                    # DX値が十分にある場合のみADXを計算
                    if len(dx_window) > 0:
                        dx_array = np.array(dx_window)
                        adx[i] = hyper_smoother(dx_array, min(curr_period, len(dx_array)))[-1]
            
            # 0-1の範囲に正規化
            adx = adx / 100.0
            plus_di = plus_di / 100.0
            minus_di = minus_di / 100.0
            
            # 結果の保存
            self._result = AlphaADXResult(
                values=adx,
                er=er,
                plus_di=plus_di,
                minus_di=minus_di,
                dynamic_period=dynamic_period,
                tr=tr
            )
            
            self._values = adx
            return adx
            
        except Exception as e:
            self.logger.error(f"AlphaADX計算中にエラー: {str(e)}")
            return np.full(len(data) if hasattr(data, '__len__') else 0, np.nan)
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_directional_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        +DIと-DIの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (+DI, -DI)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.plus_di, self._result.minus_di
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的ADX期間の値を取得する
        
        Returns:
            np.ndarray: 動的ADX期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range (TR)の値を取得する
        
        Returns:
            np.ndarray: TRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.tr 