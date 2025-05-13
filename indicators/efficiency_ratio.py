#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit, njit

from .indicator import Indicator
from .price_source import PriceSource


@njit
def calculate_efficiency_ratio(change: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    効率比（Efficiency Ratio）を計算する（高速化版）
    
    Args:
        change: 価格変化（終値の差分）の配列
        volatility: ボラティリティ（価格変化の絶対値の合計）の配列
    
    Returns:
        効率比の配列
    """
    return np.abs(change) / (volatility + 1e-10)  # ゼロ除算を防ぐ


@njit
def calculate_efficiency_ratio_for_period(prices: np.ndarray, period: int) -> np.ndarray:
    """
    指定された期間の効率比（ER）を計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        period: 計算期間
    
    Returns:
        効率比の配列（0-1の範囲）
        - 1に近いほど効率的な価格変動（強いトレンド）
        - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    """
    length = len(prices)
    er = np.zeros(length)
    
    for i in range(period, length):
        change = prices[i] - prices[i-period]
        volatility = np.sum(np.abs(np.diff(prices[i-period:i+1])))
        er[i] = calculate_efficiency_ratio(
            np.array([change]),
            np.array([volatility])
        )[0]
    
    return er


class EfficiencyRatio(Indicator):
    """
    効率比（Efficiency Ratio）インジケーター
    
    価格変動の効率性を測定する指標
    - 1に近いほど効率的な価格変動（強いトレンド）
    - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    
    使用方法：
    - 0.618以上: 効率的な価格変動（強いトレンド）
    - 0.382以下: 非効率な価格変動（レンジ・ノイズ）
    """
    
    def __init__(self, period: int = 10, src_type: str = 'close'):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 10）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(f"ER({period}, {src_type})")
        self.period = period
        self.src_type = src_type.lower()
        # データキャッシュ用
        self._values = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        src_lower = self.src_type.lower()
        if src_lower == 'open':
            required_cols.add('open')
        elif src_lower == 'high':
            required_cols.add('high')
        elif src_lower == 'low':
            required_cols.add('low')
        elif src_lower == 'close':
            required_cols.add('close')
        elif src_lower == 'hl2':
            required_cols.update(['high', 'low'])
        elif src_lower == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif src_lower == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif src_lower == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif src_lower == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
        else:
            required_cols.add('close')  # デフォルトはclose

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
                    data_hash_val = hash(str(data))  # フォールバック
            else:
                # 関連するカラムの値でハッシュ
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            # NumPy配列の場合、形状や一部の値でハッシュ
            try:
                shape_tuple = data.shape
                first_row = tuple(data[0]) if len(data) > 0 else ()
                last_row = tuple(data[-1]) if len(data) > 0 else ()
                mean_val = np.mean(data) if data.size > 0 else 0.0
                data_repr_tuple = (shape_tuple, first_row, last_row, mean_val)
                data_hash_val = hash(data_repr_tuple)
            except Exception:
                data_hash_val = hash(data.tobytes())  # フォールバック
        else:
            data_hash_val = hash(str(data))  # その他の型

        # パラメータ文字列を作成
        param_str = f"period={self.period}_src={self.src_type}"
        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            効率比の配列（0-1の範囲）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._values
            
            # PriceSourceを使って価格データを取得
            prices = PriceSource.calculate_source(data, self.src_type)
            
            # データ長の検証
            data_length = len(prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                self._values = np.array([])
                self._data_hash = data_hash  # 空でもキャッシュする
                return self._values
            
            self._validate_period(self.period, data_length)
            
            # 効率比の計算（高速化版）
            er_values = calculate_efficiency_ratio_for_period(prices, self.period)
            
            # 結果を保存してキャッシュ
            self._values = er_values
            self._data_hash = data_hash
            
            return er_values
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EfficiencyRatio計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._values = np.full(data_len, np.nan)  # エラー時はNaN配列
            self._data_hash = None  # エラー時はキャッシュクリア
            return self._values
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._data_hash = None
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"ER({self.period}, {self.src_type})" 