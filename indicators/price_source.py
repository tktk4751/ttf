#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from numba import jit, vectorize

from .indicator import Indicator


@jit(nopython=True)
def calculate_hl2(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    HL2（高値と安値の平均）を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
    
    Returns:
        HL2の配列
    """
    return (high + low) / 2


@jit(nopython=True)
def calculate_hlc3(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    HLC3（高値、安値、終値の平均）を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        HLC3の配列
    """
    return (high + low + close) / 3


@jit(nopython=True)
def calculate_ohlc4(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    OHLC4（始値、高値、安値、終値の平均）を計算する
    
    Args:
        open_: 始値の配列
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        OHLC4の配列
    """
    return (open_ + high + low + close) / 4


@jit(nopython=True)
def calculate_hlcc4(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    HLCC4（高値、安値、終値の平均で、終値を2倍重み付け）を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        HLCC4の配列
    """
    return (high + low + close + close) / 4


@jit(nopython=True)
def calculate_weighted_close(high: np.ndarray, low: np.ndarray, close: np.ndarray, weight: float) -> np.ndarray:
    """
    重み付き終値を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        weight: 終値の重み
    
    Returns:
        重み付き終値の配列
    """
    return (high + low + close * weight) / (2 + weight)


class PriceSource(Indicator):
    """
    さまざまな価格ソースを計算するインジケーター
    
    サポートされている価格ソース:
    - open: 始値
    - high: 高値
    - low: 安値
    - close: 終値
    - hl2: (high + low) / 2
    - hlc3: (high + low + close) / 3
    - ohlc4: (open + high + low + close) / 4
    - hlcc4: (high + low + close + close) / 4 = (high + low + 2 * close) / 4
    - weighted_close: (high + low + weight * close) / (2 + weight)
    
    使用例:
        >>> price_source = PriceSource()
        >>> source_data = price_source.calculate(data)
        >>> hl2 = price_source.get_hl2()
        >>> hlc3 = price_source.get_source('hlc3')
    """
    
    SOURCES = {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'hl2': 'hl2',
        'hlc3': 'hlc3',
        'ohlc4': 'ohlc4',
        'hlcc4': 'hlcc4',
        'weighted_close': 'weighted_close'
    }
    
    @staticmethod
    def calculate(data: Union[pd.DataFrame, np.ndarray], source_type: str = 'close') -> np.ndarray:
        """
        静的メソッド: 指定されたデータから特定の価格ソースを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            source_type: 価格ソースのタイプ（デフォルト: 'close'）
                サポートされているタイプ: 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'weighted_close'
                
        Returns:
            選択された価格ソースの配列
        """
        valid_sources = ['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'weighted_close']
        if source_type not in valid_sources:
            raise ValueError(f"無効な価格ソースタイプ: {source_type}。有効なタイプ: {', '.join(valid_sources)}")
        
        try:
            # DataFrameからデータを抽出
            if isinstance(data, pd.DataFrame):
                required_columns = ['open', 'high', 'low', 'close']
                # DataFrameの列名を正規化（大文字小文字を区別しない）
                columns = {}
                df_columns = data.columns.str.lower()
                
                for req_col in required_columns:
                    # 正確な名前またはOHLCのエイリアスをチェック
                    if req_col in df_columns:
                        columns[req_col] = data.columns[df_columns == req_col][0]
                    else:
                        # エイリアスのチェック
                        aliases = {
                            'open': ['o', 'op', 'opening'],
                            'high': ['h', 'hi', 'highest'],
                            'low': ['l', 'lo', 'lowest'],
                            'close': ['c', 'cl', 'closing']
                        }
                        
                        found = False
                        for alias in aliases.get(req_col, []):
                            if alias in df_columns:
                                columns[req_col] = data.columns[df_columns == alias][0]
                                found = True
                                break
                        
                        if not found:
                            raise ValueError(f"必要な列 '{req_col}' がDataFrameに見つかりません")
                
                # データの取得
                open_prices = data[columns['open']].values
                high_prices = data[columns['high']].values
                low_prices = data[columns['low']].values
                close_prices = data[columns['close']].values
            else:
                # NumPy配列形式を想定
                if data.ndim == 2 and data.shape[1] >= 4:
                    open_prices = data[:, 0]
                    high_prices = data[:, 1]
                    low_prices = data[:, 2]
                    close_prices = data[:, 3]
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列必要です")
            
            # 基本価格ソースを返す
            if source_type == 'open':
                return open_prices
            elif source_type == 'high':
                return high_prices
            elif source_type == 'low':
                return low_prices
            elif source_type == 'close':
                return close_prices
            
            # 派生価格ソースを計算して返す
            elif source_type == 'hl2':
                return calculate_hl2(high_prices, low_prices)
            elif source_type == 'hlc3':
                return calculate_hlc3(high_prices, low_prices, close_prices)
            elif source_type == 'ohlc4':
                return calculate_ohlc4(open_prices, high_prices, low_prices, close_prices)
            elif source_type == 'hlcc4':
                return calculate_hlcc4(high_prices, low_prices, close_prices)
            elif source_type == 'weighted_close':
                # デフォルトの重み付け係数を使用
                return calculate_weighted_close(high_prices, low_prices, close_prices, 2.0)
            
            # ここには到達しないはず
            return close_prices
            
        except Exception as e:
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"PriceSource.calculate静的メソッド内でエラー: {error_msg}\n{stack_trace}")
            # エラー時はデータが利用可能であれば終値を返す、そうでなければ空の配列
            if 'close_prices' in locals():
                return close_prices
            return np.array([])
    
    @staticmethod
    def validate_price_source(price_source: str) -> bool:
        """
        価格ソースが有効かどうかを検証する
        
        Args:
            price_source: 検証する価格ソース
            
        Returns:
            bool: 価格ソースが有効な場合はTrue、そうでなければFalse
        """
        valid_sources = [
            'open', 'high', 'low', 'close',
            'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'weighted_close'
        ]
        return price_source in valid_sources
    
    def __init__(self, weighted_close_factor: float = 2.0):
        """
        コンストラクタ
        
        Args:
            weighted_close_factor: 重み付き終値における終値の重み（デフォルト: 2.0）
        """
        super().__init__("PriceSource")
        self.weighted_close_factor = weighted_close_factor
        
        # 各ソースごとのデータキャッシュ
        self._sources: Dict[str, np.ndarray] = {}
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データからハッシュ値を生成する（キャッシュ用）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            データのハッシュ文字列
        """
        if isinstance(data, pd.DataFrame):
            return hash(str(data.head(3)) + str(data.tail(3)) + str(len(data)))
        else:
            sample = np.concatenate([data[:3], data[-3:]])
            return hash(str(sample) + str(len(data)))
    
    def _extract_data(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        データからOHLC価格を抽出する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            開始価格、高値、安値、終値のタプル
        """
        if isinstance(data, pd.DataFrame):
            # カラム名を小文字に変換して検索
            columns = {col.lower(): col for col in data.columns}
            
            # データがすべて存在するか確認
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in columns:
                    raise ValueError(f"必要なカラム '{col}' がデータに含まれていません")
            
            # データの抽出
            open_prices = data[columns['open']].values
            high_prices = data[columns['high']].values
            low_prices = data[columns['low']].values
            close_prices = data[columns['close']].values
        else:
            # NumPy配列形式を想定
            if data.ndim == 2 and data.shape[1] >= 4:
                open_prices = data[:, 0]
                high_prices = data[:, 1]
                low_prices = data[:, 2]
                close_prices = data[:, 3]
            else:
                raise ValueError("NumPy配列は2次元で、少なくとも4列必要です")
        
        return open_prices, high_prices, low_prices, close_prices
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        すべての価格ソースを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            各ソースタイプごとの価格データを含む辞書
        """
        try:
            # データハッシュによるキャッシュ確認
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._sources:
                return self._sources
            
            # ハッシュが異なる場合は再計算
            self._data_hash = data_hash
            
            # データの抽出
            open_prices, high_prices, low_prices, close_prices = self._extract_data(data)
            
            # 基本ソースの保存
            self._sources = {
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices
            }
            
            # 派生ソースの計算
            self._sources['hl2'] = calculate_hl2(high_prices, low_prices)
            self._sources['hlc3'] = calculate_hlc3(high_prices, low_prices, close_prices)
            self._sources['ohlc4'] = calculate_ohlc4(open_prices, high_prices, low_prices, close_prices)
            self._sources['hlcc4'] = calculate_hlcc4(high_prices, low_prices, close_prices)
            self._sources['weighted_close'] = calculate_weighted_close(
                high_prices, low_prices, close_prices, self.weighted_close_factor
            )
            
            return self._sources
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"PriceSource計算中にエラー: {error_msg}\n{stack_trace}")
            return {}
    
    def get_source(self, source_type: str = 'close') -> np.ndarray:
        """
        指定された種類の価格ソースを取得する
        
        Args:
            source_type: 価格ソースのタイプ（デフォルト: 'close'）
                サポートされているタイプ: 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'hlcc4', 'weighted_close'
                
        Returns:
            選択された価格ソースの配列
        """
        if source_type not in self.SOURCES:
            raise ValueError(f"無効な価格ソースタイプ: {source_type}。有効なタイプ: {', '.join(self.SOURCES.keys())}")
        
        if not self._sources or source_type not in self._sources:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        return self._sources[source_type]
    
    def get_open(self) -> np.ndarray:
        """始値を取得する"""
        return self.get_source('open')
    
    def get_high(self) -> np.ndarray:
        """高値を取得する"""
        return self.get_source('high')
    
    def get_low(self) -> np.ndarray:
        """安値を取得する"""
        return self.get_source('low')
    
    def get_close(self) -> np.ndarray:
        """終値を取得する"""
        return self.get_source('close')
    
    def get_hl2(self) -> np.ndarray:
        """HL2を取得する"""
        return self.get_source('hl2')
    
    def get_hlc3(self) -> np.ndarray:
        """HLC3を取得する"""
        return self.get_source('hlc3')
    
    def get_ohlc4(self) -> np.ndarray:
        """OHLC4を取得する"""
        return self.get_source('ohlc4')
    
    def get_hlcc4(self) -> np.ndarray:
        """HLCC4を取得する"""
        return self.get_source('hlcc4')
    
    def get_weighted_close(self) -> np.ndarray:
        """重み付き終値を取得する"""
        return self.get_source('weighted_close')
    
    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        self._sources = {}
        self._data_hash = None 