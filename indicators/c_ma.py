#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class CMAResult:
    """CMAの計算結果"""
    values: np.ndarray        # CMAの値
    er: np.ndarray            # サイクル効率比（CER）
    kama_period: np.ndarray   # サイクル検出器から直接決定されたKAMAピリオド
    sc: np.ndarray            # スムージング定数
    dc_values: np.ndarray     # ドミナントサイクル値


@njit(fastmath=True, cache=True)
def calculate_c_ma(prices: np.ndarray, er: np.ndarray, kama_period: np.ndarray,
                  fast_period: int, slow_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    CMAを計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        er: 効率比の配列（ERまたはCER）
        kama_period: サイクル検出器から決定されたKAMAピリオドの配列
        fast_period: 速い移動平均の期間（固定値）
        slow_period: 遅い移動平均の期間（固定値）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: CMAの配列とスムージング定数の配列
    """
    length = len(prices)
    c_ma = np.full(length, np.nan, dtype=np.float64)
    sc_values = np.zeros(length, dtype=np.float64)
    
    # 最初のCMAは最初の価格
    if length > 0:
        c_ma[0] = prices[0]
    
    # 固定のfast/slow定数を計算
    fast_constant = 2.0 / (fast_period + 1.0)
    slow_constant = 2.0 / (slow_period + 1.0)
    
    # 各時点でのCMAを計算
    for i in range(1, length):
        if np.isnan(er[i]):
            c_ma[i] = c_ma[i-1]
            continue
        
        # サイクル効率比に基づいてスムージング定数を計算
        sc = (er[i] * (fast_constant - slow_constant) + slow_constant) ** 2
        
        # 0-1の範囲に制限
        sc_values[i] = max(0.0, min(1.0, sc))
        
        # CMAの計算
        c_ma[i] = c_ma[i-1] + sc_values[i] * (prices[i] - c_ma[i-1])
    
    return c_ma, sc_values


class CMA(Indicator):
    """
    CMA (Cycle Moving Average) インジケーター
    
    Z_MAをシンプル化したバージョン。特徴：
    - KAMAの期間をサイクル検出器で直接決定
    - fast期間とslow期間は固定値を使用
    - サイクル効率比（CER）を使用した適応型移動平均
    """
    
    def __init__(
        self,
        detector_type: str = 'hody_e',
        cycle_part: float = 0.5,
        lp_period: int = 5,
        hp_period: int = 55,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 62,
        min_output: int = 13,
        fast_period: int = 2,
        slow_period: int = 30,
        src_type: str = 'hlc3'
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
            max_cycle: 最大サイクル期間（デフォルト: 144）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 8）
            fast_period: 速い移動平均の期間（デフォルト: 2、固定値）
            slow_period: 遅い移動平均の期間（デフォルト: 30、固定値）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3（デフォルト）
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(
            f"CMA({detector_type}, {max_output}, {min_output}, {fast_period}, {slow_period}, {src_type})"
        )
        
        # パラメータを保存
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
        # ドミナントサイクル検出器を初期化
        self.dc_detector = EhlersUnifiedDC(
            detector_type=self.detector_type,
            cycle_part=self.cycle_part,
            lp_period=self.lp_period,
            hp_period=self.hp_period,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            src_type=self.src_type
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする（高速化）
            if 'close' in data.columns:
                data_hash = hash(data['close'].values.tobytes())
            else:
                # closeカラムがない場合は全カラムのハッシュ（高速化）
                cols = ['high', 'low', 'close', 'open']
                data_values = np.vstack([data[col].values for col in cols if col in data.columns])
                data_hash = hash(data_values.tobytes())
        else:
            # NumPy配列の場合（高速化）
            if isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 4:
                    # OHLCデータの場合はcloseだけハッシュ（高速化）
                    data_hash = hash(data[:, 3].tobytes())
                else:
                    # それ以外は全体をハッシュ（高速化）
                    data_hash = hash(data.tobytes())
            else:
                data_hash = hash(str(data))
        
        # 外部ERがある場合はそのハッシュも含める（高速化）
        external_er_hash = "no_external_er"
        if external_er is not None and isinstance(external_er, np.ndarray):
            external_er_hash = hash(external_er.tobytes())
        
        # パラメータ値を含める
        param_str = (
            f"{self.detector_type}_{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_"
            f"{self.max_output}_{self.min_output}_{self.fast_period}_{self.slow_period}_"
            f"{self.src_type}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        CMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            CMAの値
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
            
            # データ長の検証
            data_length = len(prices)
            
            # NumPy配列に変換して計算を高速化
            prices_np = np.asarray(prices, dtype=np.float64)
            
            # ドミナントサイクルの計算
            dc_values = self.dc_detector.calculate(data)
            
            # サイクル効率比（CER）を使用
            er = np.asarray(external_er, dtype=np.float64)
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # CMAの計算（高速化版）
            c_ma_values, sc_values = calculate_c_ma(
                prices_np,
                er,
                dc_values,
                self.fast_period,
                self.slow_period
            )
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = CMAResult(
                values=np.copy(c_ma_values),
                er=np.copy(er),
                kama_period=np.copy(dc_values),
                sc=np.copy(sc_values),
                dc_values=np.copy(dc_values)
            )
            
            self._values = c_ma_values
            return c_ma_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は前回の結果を維持する
            if self._result is None:
                # 初回エラー時は空の配列を返す
                return np.array([])
            return self._result.values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dc_values(self) -> np.ndarray:
        """
        ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_kama_period(self) -> np.ndarray:
        """
        KAMAピリオドの値を取得する
        
        Returns:
            np.ndarray: KAMAピリオドの値
        """
        if self._result is None:
            return np.array([])
        return self._result.kama_period
    
    def get_smoothing_constants(self) -> np.ndarray:
        """
        スムージング定数の値を取得する
        
        Returns:
            np.ndarray: スムージング定数の値
        """
        if self._result is None:
            return np.array([])
        return self._result.sc
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.dc_detector.reset() 