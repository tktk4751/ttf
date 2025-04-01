#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import logging

from logger import get_logger


class Indicator(ABC):
    """
    インディケーターの基底クラス
    すべてのインディケーターはこのクラスを継承する
    """
    
    # ソースタイプの定義
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: インディケーターの名前
        """
        self.name = name
        self.logger = logging.getLogger(f"indicator.{name}")
        self._values: Optional[np.ndarray] = None
    
    @abstractmethod
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        インディケーターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            計算結果のNumPy配列
        """
        pass
    
    def _validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        入力データを検証し、適切な形式に変換する
        
        Args:
            data: 入力データ
        
        Returns:
            検証済みのNumPy配列
        """
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            return data['close'].values
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("NumPy配列は1次元である必要があります")
            return data
        else:
            raise ValueError("サポートされていないデータ型です")
    
    def _validate_period(self, period: int, data_length: int) -> None:
        """
        期間パラメータを検証する
        
        Args:
            period: 期間
            data_length: データの長さ
        """
        if period < 1:
            raise ValueError("期間は1以上である必要があります")
        if period > data_length:
            raise ValueError("期間がデータ長より大きいです")
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str = 'close') -> np.ndarray:
        """
        指定されたソースタイプの価格データを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        
        Returns:
            ソースタイプに基づく価格データの配列
        """
        # ソースタイプの検証
        src_type = src_type.lower()
        if src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # DataFrameの場合
        if isinstance(data, pd.DataFrame):
            # カラム名のマッピング
            column_mapping = {
                'open': ['open', 'Open'],
                'high': ['high', 'High'],
                'low': ['low', 'Low'],
                'close': ['close', 'Close', 'adj close', 'Adj Close']
            }
            
            # 各OHLCカラムを見つける
            ohlc = {}
            for key, possible_names in column_mapping.items():
                found = False
                for name in possible_names:
                    if name in data.columns:
                        ohlc[key] = data[name].values
                        found = True
                        break
                
                # closeだけは必須、他は特定のsrc_typeで必要な場合のみ必須
                if not found and (key == 'close' or 
                                 (key == 'high' and src_type in ['hlc3', 'hl2', 'ohlc4']) or
                                 (key == 'low' and src_type in ['hlc3', 'hl2', 'ohlc4']) or
                                 (key == 'open' and src_type == 'ohlc4')):
                    raise ValueError(f"DataFrameには'{key}'カラムが必要です。利用可能なカラム: {data.columns.tolist()}")
            
            # ソースタイプに基づいて価格データを計算
            if src_type == 'close':
                return ohlc['close']
            elif src_type == 'hlc3':
                return (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3.0
            elif src_type == 'hl2':
                return (ohlc['high'] + ohlc['low']) / 2.0
            elif src_type == 'ohlc4':
                return (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4.0
        
        # NumPy配列の場合
        else:
            if data.ndim == 1:
                # 1次元配列の場合は、closeとみなす
                if src_type != 'close':
                    raise ValueError(f"1次元配列の場合、ソースタイプは'close'のみサポートされています: {src_type}")
                return data
            elif data.ndim == 2 and data.shape[1] >= 4:
                # 2次元配列の場合はOHLCフォーマットを想定
                # [0]=open, [1]=high, [2]=low, [3]=close
                if src_type == 'close':
                    return data[:, 3]
                elif src_type == 'hlc3':
                    return (data[:, 1] + data[:, 2] + data[:, 3]) / 3.0
                elif src_type == 'hl2':
                    return (data[:, 1] + data[:, 2]) / 2.0
                elif src_type == 'ohlc4':
                    return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4.0
            else:
                raise ValueError(f"NumPy配列は1次元または少なくとも4列（OHLC）の2次元である必要があります")
        
        return np.array([])  # エラー時の保険
    
    def get_values(self) -> np.ndarray:
        """
        計算済みの値を取得する
        
        Returns:
            計算結果のNumPy配列
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values
        
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        派生クラスでオーバーライドして、内部状態やキャッシュを初期化できます
        """
        self._values = None
