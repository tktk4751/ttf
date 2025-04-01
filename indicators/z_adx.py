#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, vectorize, float64, prange

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import calculate_hyper_smoother_numba
from .alma import calculate_alma
from .ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZADXResult:
    """ZADXの計算結果"""
    values: np.ndarray          # ZADXの値（0-1の範囲で正規化）
    er: np.ndarray              # サイクル効率比（CER）
    plus_di: np.ndarray         # +DI値
    minus_di: np.ndarray        # -DI値
    dynamic_period: np.ndarray  # 動的ADX期間
    tr: np.ndarray              # True Range
    dc_values: np.ndarray       # ドミナントサイクル値


@vectorize(['float64(float64, float64)'], nopython=True, fastmath=True)
def max_vec(a: float, b: float) -> float:
    """ベクトル化された最大値計算"""
    return max(a, b)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def max3_vec(a: float, b: float, c: float) -> float:
    """3つの値の最大値を計算（ベクトル化）"""
    return max(a, max(b, c))


@njit(fastmath=True, parallel=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Rangeの配列
    """
    size = len(high)
    tr = np.zeros(size)
    
    # 最初のTRは単純に高値 - 安値
    if size > 0:
        tr[0] = high[0] - low[0]
    
    # 残りのTRを計算
    for i in prange(1, size):
        # 3つの条件の最大値を計算
        hl = high[i] - low[i]
        hcp = abs(high[i] - close[i-1])
        lcp = abs(low[i] - close[i-1])
        
        tr[i] = max(hl, max(hcp, lcp))
    
    return tr


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なADXピリオドを計算する（ベクトル化版）
    
    Args:
        er: 効率比の値
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


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: np.ndarray, min_period: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的なADXピリオドを計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間の配列（ドミナントサイクルから計算）
        min_period: 最小期間の配列（ドミナントサイクルから計算）
    
    Returns:
        動的な期間の配列
    """
    size = len(er)
    dynamic_period = np.zeros(size)
    
    for i in range(size):
        if np.isnan(er[i]):
            dynamic_period[i] = max_period[i]
        else:
            # ERが高い（トレンドが強い）ほど期間は短く、
            # ERが低い（トレンドが弱い）ほど期間は長くなる
            dynamic_period[i] = min_period[i] + (1.0 - abs(er[i])) * (max_period[i] - min_period[i])
            # 整数に丸める
            dynamic_period[i] = round(dynamic_period[i])
    
    return dynamic_period


@njit(fastmath=True)
def calculate_z_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    max_period: int,
    smoother_type: str = 'alma'  # 'alma'または'hyper'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ZADXを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比（Efficiency Ratio）の配列
        dynamic_period: 動的な期間の配列
        max_period: 最大期間
        smoother_type: スムーザーのタイプ ('alma'または'hyper')
    
    Returns:
        (ZADX, +DI, -DI, TR)のタプル
    """
    size = len(close)
    tr = np.zeros(size)
    plus_dm = np.zeros(size)
    minus_dm = np.zeros(size)
    
    # 最初の値を初期化
    tr[0] = high[0] - low[0]
    
    # True Range、プラスDM、マイナスDMを計算
    for i in range(1, size):
        # True Range計算
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        
        # Directional Movement計算
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > 0 and up_move > down_move:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > 0 and down_move > up_move:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # 平滑化されたDM、TRの計算用配列
    smoothed_plus_dm = np.zeros(size)
    smoothed_minus_dm = np.zeros(size)
    smoothed_tr = np.zeros(size)
    
    # +DI, -DIの計算用配列
    plus_di = np.zeros(size)
    minus_di = np.zeros(size)
    
    # ZADX計算用配列
    zadx = np.zeros(size)
    
    # Wilder方式のスムージングの代わりに各ポイントで適応的な平滑化を使用
    for i in range(1, size):
        period = int(dynamic_period[i])
        if period < 1:
            period = 1
            
        # 最初のmax_period+1ポイントは各時点でのデータが不足しているのでスキップ
        if i < max_period + 1:
            continue
            
        # その時点での適応的な平滑化を行う
        if smoother_type == 'alma':
            # ALMA（アルノルドの移動平均）を使用
            # ALMAの典型的なパラメータ
            sigma = 6.0  # ガウシアンカーブの標準偏差
            offset = 0.85  # 重みの中心オフセット（0-1）
            
            # スムージング計算
            window_plus_dm = plus_dm[max(0, i-period*2):i+1]
            window_minus_dm = minus_dm[max(0, i-period*2):i+1]
            window_tr = tr[max(0, i-period*2):i+1]
            
            if len(window_plus_dm) > 0 and len(window_minus_dm) > 0 and len(window_tr) > 0:
                # ALMAスムージング（各値の平滑化）
                weights = np.zeros(period)
                m = offset * (period - 1)
                s = period / sigma
                
                sum_weights = 0
                for j in range(period):
                    weights[j] = np.exp(-0.5 * ((j - m) / s) ** 2)
                    sum_weights += weights[j]
                
                # 重みの正規化
                weights /= sum_weights
                
                # 直近のperiod個の値に重みを付けて平滑化
                if len(window_plus_dm) >= period:
                    smoothed_plus_dm[i] = np.sum(window_plus_dm[-period:] * weights)
                    smoothed_minus_dm[i] = np.sum(window_minus_dm[-period:] * weights)
                    smoothed_tr[i] = np.sum(window_tr[-period:] * weights)
                else:
                    # データが足りない場合は単純な平均
                    smoothed_plus_dm[i] = np.mean(window_plus_dm)
                    smoothed_minus_dm[i] = np.mean(window_minus_dm)
                    smoothed_tr[i] = np.mean(window_tr)
        
        elif smoother_type == 'hyper':
            # ハイパースムーサーを使用
            alpha = 2.0 / (period + 1.0)
            
            smoothed_plus_dm[i] = alpha * plus_dm[i] + (1 - alpha) * smoothed_plus_dm[i-1]
            smoothed_minus_dm[i] = alpha * minus_dm[i] + (1 - alpha) * smoothed_minus_dm[i-1]
            smoothed_tr[i] = alpha * tr[i] + (1 - alpha) * smoothed_tr[i-1]
        
        else:
            # デフォルトはWilder法に似たEMA
            alpha = 1.0 / period
            
            smoothed_plus_dm[i] = alpha * plus_dm[i] + (1 - alpha) * smoothed_plus_dm[i-1]
            smoothed_minus_dm[i] = alpha * minus_dm[i] + (1 - alpha) * smoothed_minus_dm[i-1]
            smoothed_tr[i] = alpha * tr[i] + (1 - alpha) * smoothed_tr[i-1]
        
        # 0除算を防ぐ
        if smoothed_tr[i] > 0:
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        
        # ZADX計算
        di_diff = abs(plus_di[i] - minus_di[i])
        di_sum = plus_di[i] + minus_di[i]
        
        # 0除算を防ぐ
        if di_sum > 0:
            dx = 100.0 * di_diff / di_sum
        else:
            dx = 0.0
        
        # EMAでZADXを計算
        if i > 0:
            zadx[i] = (dx + (period - 1) * zadx[i-1]) / period
        else:
            zadx[i] = dx
    
    # ZADXを0-1の範囲に正規化
    zadx = zadx / 100.0
    
    return zadx, plus_di, minus_di, tr


class ZADX(Indicator):
    """
    ZADX インジケーター
    
    Alpha ADXの拡張版。サイクル効率比とドミナントサイクルを用いて動的に期間を調整します。
    
    特徴:
    - ドミナントサイクルを用いた動的な期間計算
    - サイクル効率比による細かな調整
    - ALMAまたはハイパースムーサーによる平滑化オプション
    - トレンドの強さと方向の分析に最適化
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,          # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 34,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 5,              # 最大期間用ドミナントサイクル計算用
        max_dc_max_output: int = 21,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_output: int = 8,             # 最大期間用ドミナントサイクル計算用
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 21,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 3,               # 最小期間用ドミナントサイクル計算用
        min_dc_max_output: int = 13,             # 最小期間用ドミナントサイクル計算用
        min_dc_min_output: int = 3,              # 最小期間用ドミナントサイクル計算用
        
        er_period: int = 21,                     # 効率比の計算期間 
        smoother_type: str = 'alma'              # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル検出の最大期間（デフォルト: 55）
            max_dc_min_cycle: 最大期間用ドミナントサイクル検出の最小期間（デフォルト: 5）
            max_dc_max_output: 最大期間用ドミナントサイクル出力の最大値（デフォルト: 55）
            max_dc_min_output: 最大期間用ドミナントサイクル出力の最小値（デフォルト: 5）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル検出の最大期間（デフォルト: 34）
            min_dc_min_cycle: 最小期間用ドミナントサイクル検出の最小期間（デフォルト: 3）
            min_dc_max_output: 最小期間用ドミナントサイクル出力の最大値（デフォルト: 13）
            min_dc_min_output: 最小期間用ドミナントサイクル出力の最小値（デフォルト: 3）
            
            er_period: 効率比の計算期間（デフォルト: 21）
            smoother_type: 平滑化タイプ ('alma'または'hyper')（デフォルト: 'alma'）
        """
        super().__init__(
            f"ZADX({max_dc_max_output}-{min_dc_min_output}, {smoother_type})"
        )
        
        # ドミナントサイクルの最大期間検出器
        self.max_dc = EhlersHoDyDC(
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=max_dc_max_output,
            min_output=max_dc_min_output,
            src_type='hlc3'
        )
        
        # ドミナントサイクルの最小期間検出器
        self.min_dc = EhlersHoDyDC(
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=min_dc_max_output,
            min_output=min_dc_min_output,
            src_type='hlc3'
        )
        
        self.er_period = er_period
        self.smoother_type = smoother_type
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            columns = ['high', 'low', 'close']
            if all(col in data.columns for col in columns):
                data_hash = hash(tuple(map(tuple, data[columns].values)))
            else:
                # 必要なカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合は必要な列だけハッシュ
                data_hash = hash(tuple(map(tuple, data[:, 1:4])))  # high, low, close
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = (
            f"{self.max_dc.cycle_part}_{self.max_dc.max_cycle}_{self.max_dc.min_cycle}_"
            f"{self.max_dc.max_output}_{self.max_dc.min_output}_"
            f"{self.min_dc.cycle_part}_{self.min_dc.max_cycle}_{self.min_dc.min_cycle}_"
            f"{self.min_dc.max_output}_{self.min_dc.min_output}_"
            f"{self.er_period}_{self.smoother_type}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ZADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
            external_er: 外部から提供される効率比（オプション）
        
        Returns:
            ZADX値の配列（0-1の範囲）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                required_columns = ['high', 'low', 'close']
                if not all(col in data.columns for col in required_columns):
                    raise ValueError(f"DataFrameには{required_columns}カラムが必要です")
                
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は(n,4)以上の形状である必要があります")
            
            # データ長の検証
            data_length = len(close)
            
            # ドミナントサイクルを使用して動的な最大/最小期間を計算
            max_periods = self.max_dc.calculate(data)
            min_periods = self.min_dc.calculate(data)
            
            # 効率比の計算（外部から提供されない場合は計算）
            if external_er is None:
                er = calculate_efficiency_ratio_for_period(close, self.er_period)
            else:
                # 外部から提供されるERをそのまま使用
                if len(external_er) != data_length:
                    raise ValueError(f"外部ERの長さ({len(external_er)})がデータ長({data_length})と一致しません")
                er = external_er
            
            # 動的なADX期間の計算
            dynamic_period = calculate_dynamic_period(er, max_periods, min_periods)
            
            # ZADXの計算
            zadx, plus_di, minus_di, tr = calculate_z_adx(
                high, 
                low, 
                close, 
                er, 
                dynamic_period, 
                max_period=int(np.max(max_periods) if len(max_periods) > 0 else 20),
                smoother_type=self.smoother_type
            )
            
            # 結果を保存
            self._result = ZADXResult(
                values=zadx,
                er=er,
                plus_di=plus_di,
                minus_di=minus_di,
                dynamic_period=dynamic_period,
                tr=tr,
                dc_values=max_periods
            )
            
            self._values = zadx  # 基底クラスの要件を満たすため
            
            return zadx
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZADX計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_max_dc_values(self) -> np.ndarray:
        """
        最大ドミナントサイクル値を取得する
        
        Returns:
            np.ndarray: 最大ドミナントサイクル値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_min_dc_values(self) -> np.ndarray:
        """
        最小ドミナントサイクル値を取得する
        
        Returns:
            np.ndarray: 最小ドミナントサイクル値
        """
        if self._result is None:
            return np.array([])
        return self.min_dc._values if hasattr(self.min_dc, '_values') else np.array([])
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range値を取得する
        
        Returns:
            np.ndarray: True Range値
        """
        if self._result is None:
            return np.array([])
        return self._result.tr
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的なADX期間の値を取得する
        
        Returns:
            np.ndarray: 動的なADX期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_period
    
    def get_directional_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        方向性指数（+DIと-DI）を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (+DI, -DI)のタプル
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        return self._result.plus_di, self._result.minus_di
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.max_dc.reset()
        self.min_dc.reset() 