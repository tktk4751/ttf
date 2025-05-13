#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
# 使用するインジケーターをインポート
from .c_ma import CMA 
from .c_atr import CATR
from .cycle_volatility_er import CycleVolatilityER
from .cycle_efficiency_ratio import CycleEfficiencyRatio # CMAとCATR用に必要


@dataclass
class CCChannelResult: # クラス名を変更
    """CCチャネルの計算結果"""
    middle: np.ndarray        # 中心線（CMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cver: np.ndarray          # Cycle Volatility Efficiency Ratio
    cer: np.ndarray           # Cycle Efficiency Ratio (Price based)
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    catr: np.ndarray          # CATR値（絶対値）
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値 (adaptiveメソッドの場合のみ有効)
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値 (adaptiveメソッドの場合のみ有効)


# Numba最適化関数 (ZChannelから流用)
@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_multiplier_vec(cer: float, max_mult: float, min_mult: float) -> float:
    """
    効率比に基づいて動的なATR乗数を計算する（ベクトル化版）- Adaptiveメソッド用
    
    Args:
        cer: 効率比の値 (CVER)
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    if np.isnan(cer):
        return np.nan
    # ERが高いほど乗数は小さく、低いほど乗数は大きくなる
    return max_mult - abs(cer) * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_max_multiplier(cer: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    効率比に基づいて動的な最大ATR乗数を計算する（ベクトル化版）- Adaptiveメソッド用
    
    Args:
        cer: 効率比の値 (CVER)
        max_max_mult: 最大乗数の最大値
        min_max_mult: 最大乗数の最小値
    
    Returns:
        動的な最大乗数の値
    """
    if np.isnan(cer):
        return np.nan
    # ERが低いほど最大乗数は大きく、高いほど小さくなる
    return max_max_mult - abs(cer) * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_min_multiplier(cer: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    効率比に基づいて動的な最小ATR乗数を計算する（ベクトル化版）- Adaptiveメソッド用
    
    Args:
        cer: 効率比の値 (CVER)
        max_min_mult: 最小乗数の最大値
        min_min_mult: 最小乗数の最小値
    
    Returns:
        動的な最小乗数の値
    """
    if np.isnan(cer):
        return np.nan
    # ERが高いほど最小乗数は小さく (min_min_multに近づく)、
    # ERが低いほど最小乗数は大きくなる (max_min_multに近づく)
    return max_min_mult - abs(cer) * (max_min_mult - min_min_mult)


# --- 新しい動的乗数計算関数 ---
@vectorize(['float64(float64)'], nopython=True, fastmath=True, cache=True)
def calculate_new_dynamic_multiplier(er_value: float) -> float:
    """
    新しい動的乗数を計算する (ベクトル化版) - Newメソッド用
    式: round((1 - abs(ER)) * 10, 2) (ERはCVERまたはCER)
    
    Args:
        er_value: 効率比の値 (CVER または CER)
        
    Returns:
        動的な乗数の値
    """
    if np.isnan(er_value):
        return np.nan
    # ERが低い(0に近い)ほど乗数は大きく、高い(1に近い)ほど乗数は小さくなる
    multiplier = (1.0 - abs(er_value)) * 10.0
    # 小数点第二位で丸める
    return np.round(multiplier, 2)
# -----------------------------


# calculate_z_channel 関数は名前と引数名以外は同じロジックなので流用
@njit(fastmath=True, parallel=True, cache=True)
def calculate_cc_channel_bands( # 関数名を変更
    center_line: np.ndarray, # 引数名を変更 (c_ma)
    volatility: np.ndarray,  # 引数名を変更 (catr_absolute)
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False # CC_Channelでは常にFalseで使用
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CCチャネルのバンドを計算する（パラレル高速化版）
    
    Args:
        center_line: 中心線の配列 (CMA)
        volatility: ボラティリティの配列 (CATR絶対値)
        dynamic_multiplier: 動的乗数の配列
        use_percent: パーセントベースボラティリティを使用するかどうか (常にFalse)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(center_line)
    middle = np.copy(center_line)
    upper = np.full_like(center_line, np.nan, dtype=np.float64)
    lower = np.full_like(center_line, np.nan, dtype=np.float64)
    
    # 有効なデータ判別用マスク (dynamic_multiplierもチェック)
    valid_mask = ~(np.isnan(center_line) | np.isnan(volatility) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列 - パラレル計算の準備
    band_width = np.zeros_like(center_line, dtype=np.float64)
    
    # 並列計算で一度にバンド幅を計算
    for i in prange(length):
        if valid_mask[i]:
            # 金額ベースのCATRを使用
            band_width[i] = volatility[i] * dynamic_multiplier[i]
    
    # 一度にバンドを計算（並列処理）
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class CC_Channel(Indicator): # クラス名を変更
    """
    CCチャネル (CMA + CATR Channel) インジケーター
    
    特徴:
    - 中心線にCMA (Cycle Moving Average) を使用
    - バンド幅の計算にCATR (Cycle Average True Range) を使用
    - ATR乗数がCVERに基づいて動的に調整 (adaptiveメソッドの場合)
    - または、(1 - abs(ER)) * 10 に基づく乗数を使用 (newメソッドの場合、ERはCVERかCERを選択可能)
    - 使用するインジケーターのパラメータは外部から設定可能
    
    市場状態に応じた最適な挙動:
    - ボラティリティトレンド強い (CVER/CER高い):
      - 狭いバンド幅（小さい乗数）
    - ボラティリティトレンド弱い (CVER/CER低い):
      - 広いバンド幅（大きい乗数）
    """
    
    def __init__(
        self,
        # --- CCチャネル固有パラメータ ---
        multiplier_method: str = 'adaptive', # 動的乗数計算方法: 'adaptive' or 'new'
        new_method_er_source: str = 'cver', # 'new'メソッドで使用するERソース: 'cver' or 'cer'
        # 'adaptive'メソッド用パラメータ
        cc_max_max_multiplier: float = 8.0,
        cc_min_max_multiplier: float = 3.0,
        cc_max_min_multiplier: float = 1.5,
        cc_min_min_multiplier: float = 0.5,
        
        # --- CMAパラメータ ---
        cma_detector_type: str = 'hody_e', 
        cma_cycle_part: float = 0.618, 
        cma_lp_period: int = 5, 
        cma_hp_period: int = 89, 
        cma_max_cycle: int = 55, 
        cma_min_cycle: int = 5, 
        cma_max_output: int = 34, 
        cma_min_output: int = 8, 
        cma_fast_period: int = 2, 
        cma_slow_period: int = 30, 
        cma_src_type: str = 'hlc3', # CMAで使用するソースタイプ
        
        # --- CATRパラメータ ---
        catr_detector_type: str = 'hody', 
        catr_cycle_part: float = 0.5, 
        catr_lp_period: int = 5, 
        catr_hp_period: int = 55, 
        catr_max_cycle: int = 55, 
        catr_min_cycle: int = 5, 
        catr_max_output: int = 34, 
        catr_min_output: int = 5, 
        catr_smoother_type: str = 'alma',
        
        # --- CVERパラメータ ---
        cver_detector_type: str = 'hody', 
        cver_lp_period: int = 5, 
        cver_hp_period: int = 144, 
        cver_cycle_part: float = 0.5, 
        cver_max_cycle: int = 144, 
        cver_min_cycle: int = 5, 
        cver_max_output: int = 89, 
        cver_min_output: int = 5, 
        cver_src_type: str = 'hlc3', # CVERのドミナントサイクル計算で使用するソースタイプ
        
        # --- CERパラメータ (CMA/CATR用) ---
        cer_detector_type: str = 'hody', 
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,
        cer_max_cycle: int = 144,
        cer_min_cycle: int = 5,
        cer_max_output: int = 89,
        cer_min_output: int = 5,
        cer_src_type: str = 'hlc3' # CERのドミナントサイクル計算で使用するソースタイプ
        
    ): 
        """
        コンストラクタ
        内部で使用するインジケーターのパラメータを設定します。
        """
        # multiplier_method と new_method_er_source の検証
        if multiplier_method not in ['adaptive', 'new']:
            raise ValueError("multiplier_method must be 'adaptive' or 'new'")
        if new_method_er_source not in ['cver', 'cer']:
            raise ValueError("new_method_er_source must be 'cver' or 'cer'")
            
        self.multiplier_method = multiplier_method
        self.new_method_er_source = new_method_er_source

        # パラメータを保存 (キャッシュキー生成用)
        self.src_type = cer_src_type 
        self.max_max_multiplier = cc_max_max_multiplier
        self.min_max_multiplier = cc_min_max_multiplier
        self.max_min_multiplier = cc_max_min_multiplier
        self.min_min_multiplier = cc_min_min_multiplier

        # 全パラメータを保存（キャッシュキーで使用するため）
        self._params = {
            'multiplier_method': multiplier_method, 
            'new_method_er_source': new_method_er_source, # new_method_er_source を追加
            'cc_max_max_multiplier': cc_max_max_multiplier,
            'cc_min_max_multiplier': cc_min_max_multiplier,
            'cc_max_min_multiplier': cc_max_min_multiplier,
            'cc_min_min_multiplier': cc_min_min_multiplier,
            'cma_detector_type': cma_detector_type, 
            'cma_cycle_part': cma_cycle_part, 
            'cma_lp_period': cma_lp_period, 
            'cma_hp_period': cma_hp_period, 
            'cma_max_cycle': cma_max_cycle, 
            'cma_min_cycle': cma_min_cycle, 
            'cma_max_output': cma_max_output, 
            'cma_min_output': cma_min_output, 
            'cma_fast_period': cma_fast_period, 
            'cma_slow_period': cma_slow_period, 
            'cma_src_type': cma_src_type,
            'catr_detector_type': catr_detector_type, 
            'catr_cycle_part': catr_cycle_part, 
            'catr_lp_period': catr_lp_period, 
            'catr_hp_period': catr_hp_period, 
            'catr_max_cycle': catr_max_cycle, 
            'catr_min_cycle': catr_min_cycle, 
            'catr_max_output': catr_max_output, 
            'catr_min_output': catr_min_output, 
            'catr_smoother_type': catr_smoother_type,
            'cver_detector_type': cver_detector_type, 
            'cver_lp_period': cver_lp_period, 
            'cver_hp_period': cver_hp_period, 
            'cver_cycle_part': cver_cycle_part, 
            'cver_max_cycle': cver_max_cycle, 
            'cver_min_cycle': cver_min_cycle, 
            'cver_max_output': cver_max_output, 
            'cver_min_output': cver_min_output, 
            'cver_src_type': cver_src_type,
            'cer_detector_type': cer_detector_type, 
            'cer_lp_period': cer_lp_period,
            'cer_hp_period': cer_hp_period,
            'cer_cycle_part': cer_cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'cer_src_type': cer_src_type
        }
        
        # super().__init__ の呼び出し (識別子を設定)
        method_detail = f"method={self.multiplier_method}"
        if self.multiplier_method == 'adaptive':
            method_detail += f",dyn_mult=[{self.min_min_multiplier}-{self.max_min_multiplier}, {self.min_max_multiplier}-{self.max_max_multiplier}]"
        elif self.multiplier_method == 'new':
            method_detail += f",er_src={self.new_method_er_source}"
            
        super().__init__(
            f"CC_Channel(src={cer_src_type}, {method_detail})" 
        )
        
        # コンポーネントのインスタンス化 (変更なし)
        self.cma = CMA(
            detector_type=cma_detector_type, 
            cycle_part=cma_cycle_part, 
            lp_period=cma_lp_period, 
            hp_period=cma_hp_period, 
            max_cycle=cma_max_cycle, 
            min_cycle=cma_min_cycle, 
            max_output=cma_max_output, 
            min_output=cma_min_output, 
            fast_period=cma_fast_period, 
            slow_period=cma_slow_period, 
            src_type=cma_src_type
        )
        
        self.catr = CATR(
            detector_type=catr_detector_type, 
            cycle_part=catr_cycle_part, 
            lp_period=catr_lp_period, 
            hp_period=catr_hp_period, 
            max_cycle=catr_max_cycle, 
            min_cycle=catr_min_cycle, 
            max_output=catr_max_output, 
            min_output=catr_min_output, 
            smoother_type=catr_smoother_type
        )
        
        self.cver = CycleVolatilityER(
            detector_type=cver_detector_type, 
            lp_period=cver_lp_period, 
            hp_period=cver_hp_period, 
            cycle_part=cver_cycle_part, 
            max_cycle=cver_max_cycle, 
            min_cycle=cver_min_cycle, 
            max_output=cver_max_output, 
            min_output=cver_min_output, 
            src_type=cver_src_type 
        )
        
        self.cer = CycleEfficiencyRatio(
            detector_type=cer_detector_type, 
            lp_period=cer_lp_period,
            hp_period=cer_hp_period,
            cycle_part=cer_cycle_part,
            max_cycle=cer_max_cycle,
            min_cycle=cer_min_cycle,
            max_output=cer_max_output,
            min_output=cer_min_output,
            src_type=cer_src_type
        )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
        self._cache = {} 
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            cols = ['high', 'low', 'close', 'open'] # 必要な可能性のあるカラム
            relevant_cols = [col for col in cols if col in data.columns]
            if not relevant_cols:
                 raise ValueError("DataFrameに必要なカラム (high, low, closeなど) が見つかりません")
            # NumPy配列に変換してハッシュ計算（高速化）
            data_values = np.vstack([data[col].values for col in relevant_cols])
            data_hash = hash(data_values.tobytes())
        else:
             # NumPy配列の場合はバイト列に変換してハッシュ計算（高速化）
            data_hash = hash(data.tobytes() if isinstance(data, np.ndarray) else str(data))
        
        # パラメータのハッシュを生成（順序を固定するためソート）
        param_hash = hash(tuple(sorted(self._params.items()))) # new_method_er_source も含まれる

        return f"{data_hash}_{param_hash}" # 全パラメータを含むハッシュを返す
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        CCチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、src_typeに応じたカラムが必要
        
        Returns:
            中心線の値（CMA）
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache:
                # キャッシュヒット時は、格納されている結果の中心線を返す
                cached_result = self._cache[data_hash]
                if isinstance(cached_result, CCChannelResult):
                    return cached_result.middle
                else: 
                    return cached_result 
            
            # --- 計算開始 --- 
            
            # データの検証 (最低限)
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']): 
                    raise ValueError("DataFrameには少なくとも 'high', 'low', 'close' カラムが必要です")
                if not any(col in data.columns for col in ['open', 'high', 'low', 'close']):
                     raise ValueError("DataFrameに価格データカラム (open, high, low, close) が見つかりません")
            elif data.ndim != 2 or data.shape[1] < 4: # OHLC想定
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")

            # 1. 価格ベースのCERを計算 (CMA/CATR用、および new メソッドで使用する可能性あり)
            cer_values = self.cer.calculate(data)
            if cer_values is None or len(cer_values) == 0:
                raise ValueError("価格ベースのサイクル効率比 (CER) の計算に失敗しました")
            
            # 2. CMAを計算
            c_ma_values = self.cma.calculate(data, external_er=cer_values)
            if c_ma_values is None or len(c_ma_values) == 0:
                raise ValueError("CMAの計算に失敗しました")
                
            # 3. CATRを計算 (%ベースと絶対値の両方を取得)
            catr_percent = self.catr.calculate(data, external_er=cer_values)
            if catr_percent is None or len(catr_percent) == 0:
                 raise ValueError("CATRの計算に失敗しました")
            catr_absolute = self.catr.get_absolute_atr()
            if catr_absolute is None or len(catr_absolute) == 0:
                 raise ValueError("CATR絶対値の取得に失敗しました")

            # 4. CVERを計算 (CATR絶対値が必要、および new/adaptive メソッドで使用する可能性あり)
            cver_values = self.cver.calculate(price_data=data, catr_values=catr_absolute)
            if cver_values is None or len(cver_values) == 0:
                raise ValueError("サイクルボラティリティ効率比 (CVER) の計算に失敗しました")
            
            cver_np = np.asarray(cver_values, dtype=np.float64)
            cer_np = np.asarray(cer_values, dtype=np.float64) # CERもNumPy配列に
            
            # --- 動的乗数計算の分岐 ---
            nan_array = np.full_like(cver_np, np.nan) # NaN配列を事前に準備
            if self.multiplier_method == 'adaptive':
                # 'adaptive' でも ER ソースを選択できるようにする
                if self.new_method_er_source == 'cver':
                    er_source_for_adaptive = cver_np
                elif self.new_method_er_source == 'cer':
                    er_source_for_adaptive = cer_np
                else:
                    # __init__でチェック済みだが念のため
                    raise ValueError(f"Invalid new_method_er_source: {self.new_method_er_source}")
                    
                # 5a. 動的な最大・最小乗数の計算 (Adaptive - 選択されたERソースを使用)
                max_mult_values = calculate_dynamic_max_multiplier(
                    er_source_for_adaptive, # 修正: 選択されたERソースを使用
                    self.max_max_multiplier, 
                    self.min_max_multiplier 
                )
                min_mult_values = calculate_dynamic_min_multiplier(
                    er_source_for_adaptive, # 修正: 選択されたERソースを使用
                    self.max_min_multiplier, 
                    self.min_min_multiplier 
                )
                # 6a. 動的ATR乗数の計算（Adaptive - 選択されたERソースを使用）
                dynamic_multiplier = calculate_dynamic_multiplier_vec(
                    er_source_for_adaptive, # 修正: 選択されたERソースを使用
                    max_mult_values,
                    min_mult_values
                )
            elif self.multiplier_method == 'new':
                # 5b & 6b. 新しい動的乗数の計算 (New - 指定されたERソースを使用)
                if self.new_method_er_source == 'cver':
                    er_source_for_new_method = cver_np
                elif self.new_method_er_source == 'cer':
                    er_source_for_new_method = cer_np # NumPy配列に変換済みのCERを使用
                else:
                    # __init__でチェック済みだが念のため
                    raise ValueError(f"Invalid new_method_er_source: {self.new_method_er_source}")
                dynamic_multiplier = calculate_new_dynamic_multiplier(er_source_for_new_method)
                max_mult_values = nan_array 
                min_mult_values = nan_array 
            else: 
                 # __init__でチェックしているが念のため
                 raise ValueError(f"Invalid multiplier_method: {self.multiplier_method}")
            # --------------------------

            # 7. CCチャネルのバンド計算 (CMAとCATR絶対値を使用)
            middle, upper, lower = calculate_cc_channel_bands( 
                np.asarray(c_ma_values, dtype=np.float64),
                np.asarray(catr_absolute, dtype=np.float64),
                np.asarray(dynamic_multiplier, dtype=np.float64), # 計算された乗数を使用
                use_percent=False 
            )
            
            # 8. 結果の保存
            result = CCChannelResult(
                middle=np.copy(middle),        
                upper=np.copy(upper),
                lower=np.copy(lower),
                cver=np.copy(cver_values),
                cer=np.copy(cer_values), # CERも結果に保存
                dynamic_multiplier=np.copy(dynamic_multiplier),
                catr=np.copy(catr_absolute), 
                # new メソッドの場合は NaN が入る
                max_mult_values=np.copy(max_mult_values), 
                min_mult_values=np.copy(min_mult_values)  
            )
            
            # クラス変数とキャッシュに結果を保存
            self._result = result
            self._cache[data_hash] = result 
            
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CC_Channel計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    # --- ゲッターメソッド --- 
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CCチャネルのバンド値を取得する
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_cycle_volatility_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルボラティリティ効率比 (CVER) の値を取得する
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        return self._result.cver
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        return self._result.dynamic_multiplier
    
    def get_catr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値 (絶対値/金額ベース) を取得する
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        return self._result.catr
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する (multiplier_method='adaptive' の場合のみ有効)
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        # 'new' メソッドの場合は NaN が格納されているはず
        return self._result.max_mult_values
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する (multiplier_method='adaptive' の場合のみ有効)
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        # 'new' メソッドの場合は NaN が格納されているはず
        return self._result.min_mult_values
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        価格ベースのサイクル効率比 (CER) の値を取得する
        """
        if data is not None:
            self.calculate(data)
        if not hasattr(self, '_result') or self._result is None:
            return np.array([])
        return self._result.cer
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self._cache = {}
        # 内部インジケーターのリセット
        self.cma.reset() if hasattr(self.cma, 'reset') else None
        self.catr.reset() if hasattr(self.catr, 'reset') else None
        self.cver.reset() if hasattr(self.cver, 'reset') else None
        self.cer.reset() if hasattr(self.cer, 'reset') else None
