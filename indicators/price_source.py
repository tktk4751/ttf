#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価格ソース計算ユーティリティ（シンプル版）
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional


class PriceSource:
    """価格ソースの計算ユーティリティクラス（シンプル版）"""
    
    @staticmethod
    def calculate_source(
        data: Union[pd.DataFrame, np.ndarray], 
        src_type: str = 'close'
    ) -> np.ndarray:
        """
        指定されたソースタイプの価格データを計算
        
        Args:
            data: 価格データ
            src_type: ソースタイプ
                基本: 'close', 'open', 'high', 'low', 'hlc3', 'hl2', 'ohlc4', 'oc2'
        
        Returns:
            計算された価格配列（必ずnp.ndarray）
        """
        src_type = src_type.lower()
        
        # 従来のソースタイプ処理
        if isinstance(data, pd.DataFrame):
            result = PriceSource._calculate_from_dataframe(data, src_type)
        elif isinstance(data, np.ndarray):
            result = PriceSource._calculate_from_array(data, src_type)
        else:
            raise ValueError("サポートされていないデータ型です")
        
        # 結果を確実にnp.ndarrayに変換
        if result is not None:
            if isinstance(result, pd.Series):
                result = result.values
            elif not isinstance(result, np.ndarray):
                result = np.asarray(result)
            
            # 型とサイズの確認
            if result.dtype == np.object_:
                result = result.astype(np.float64)
            elif not np.issubdtype(result.dtype, np.number):
                result = result.astype(np.float64)
            
            return result
        else:
            raise ValueError("価格データの計算に失敗しました")
    
    @staticmethod
    def _calculate_from_dataframe(data: pd.DataFrame, src_type: str) -> np.ndarray:
        """DataFrameから価格を計算"""
        # カラム名のマッピング
        column_mapping = {
            'open': ['open', 'Open'],
            'high': ['high', 'High'], 
            'low': ['low', 'Low'],
            'close': ['close', 'Close', 'adj close', 'Adj Close']
        }
        
        # OHLCカラムを見つける
        ohlc = {}
        for key, possible_names in column_mapping.items():
            found = False
            for name in possible_names:
                if name in data.columns:
                    ohlc[key] = data[name].values
                    found = True
                    break
            
            # closeは常に必須（他のソースタイプでも必要）
            if not found and key == 'close':
                raise ValueError(f"'{key}' カラムが見つかりません")
        
        # ソースタイプに基づいて計算
        if src_type == 'close':
            return ohlc['close']
        elif src_type == 'high':
            if 'high' not in ohlc:
                raise ValueError("high カラムが見つかりません")
            return ohlc['high']
        elif src_type == 'low':
            if 'low' not in ohlc:
                raise ValueError("low カラムが見つかりません")
            return ohlc['low']
        elif src_type == 'open':
            if 'open' not in ohlc:
                raise ValueError("open カラムが見つかりません")
            return ohlc['open']
        elif src_type == 'hlc3':
            if 'high' not in ohlc or 'low' not in ohlc:
                raise ValueError("hlc3には high, low, close が必要です")
            return (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3.0
        elif src_type == 'hl2':
            if 'high' not in ohlc or 'low' not in ohlc:
                raise ValueError("hl2には high, low が必要です")
            return (ohlc['high'] + ohlc['low']) / 2.0
        elif src_type == 'ohlc4':
            if any(k not in ohlc for k in ['open', 'high', 'low']):
                raise ValueError("ohlc4には open, high, low, close が必要です")
            return (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4.0
        elif src_type == 'oc2':
            if 'open' not in ohlc:
                raise ValueError("oc2には open, close が必要です")
            return (ohlc['open'] + ohlc['close']) / 2.0
        else:
            raise ValueError(f"サポートされていないソースタイプ: {src_type}")
    
    @staticmethod
    def _calculate_from_array(data: np.ndarray, src_type: str) -> np.ndarray:
        """NumPy配列から価格を計算"""
        if data.ndim == 1:
            if src_type not in ['close']:
                raise ValueError("1次元配列では'close'のみサポートされています")
            return data
        elif data.ndim == 2 and data.shape[1] >= 4:
            # OHLC形式を想定 [open, high, low, close]
            if src_type == 'close':
                return data[:, 3]
            elif src_type == 'open':
                return data[:, 0]
            elif src_type == 'high':
                return data[:, 1]
            elif src_type == 'low':
                return data[:, 2]
            elif src_type == 'hlc3':
                return (data[:, 1] + data[:, 2] + data[:, 3]) / 3.0
            elif src_type == 'hl2':
                return (data[:, 1] + data[:, 2]) / 2.0
            elif src_type == 'ohlc4':
                return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4.0
            elif src_type == 'oc2':
                return (data[:, 0] + data[:, 3]) / 2.0
            else:
                raise ValueError(f"サポートされていないソースタイプ: {src_type}")
        else:
            raise ValueError("配列は1次元またはOHLC形式の2次元である必要があります")
    
    @staticmethod
    def get_available_sources() -> Dict[str, str]:
        """
        利用可能なソースタイプの一覧を取得
        
        Returns:
            ソースタイプ辞書 {タイプ: 説明}
        """
        sources = {
            'close': '終値',
            'open': '始値',
            'high': '高値',
            'low': '安値',
            'hlc3': '(高値 + 安値 + 終値) / 3',
            'hl2': '(高値 + 安値) / 2',
            'ohlc4': '(始値 + 高値 + 安値 + 終値) / 4',
            'oc2': '(始値 + 終値) / 2'
        }
        
        return sources