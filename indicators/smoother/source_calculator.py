#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価格ソース計算ユーティリティ（循環インポート回避用）
"""

import numpy as np
import pandas as pd
from typing import Union


def calculate_source_simple(data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
    """
    シンプルな価格ソース計算（PriceSourceを使わない）
    
    Args:
        data: 価格データ
        src_type: ソースタイプ
    
    Returns:
        計算された価格配列
    """
    src_type = src_type.lower()
    
    if isinstance(data, pd.DataFrame):
        # DataFrameから価格を計算
        if src_type == 'close':
            return data['close'].values
        elif src_type == 'high':
            return data['high'].values
        elif src_type == 'low':
            return data['low'].values
        elif src_type == 'open':
            return data['open'].values
        elif src_type == 'hlc3':
            return ((data['high'] + data['low'] + data['close']) / 3.0).values
        elif src_type == 'oc2':
            return ((data['open'] + data['close']) / 2.0).values
        elif src_type == 'hl2':
            return ((data['high'] + data['low']) / 2.0).values
        elif src_type == 'ohlc4':
            return ((data['open'] + data['high'] + data['low'] + data['close']) / 4.0).values
        else:
            # デフォルトはclose
            return data['close'].values
    
    elif isinstance(data, np.ndarray):
        # NumPy配列から価格を計算
        if data.ndim == 1:
            # 1次元配列はそのまま返す
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
            else:
                # デフォルトはclose
                return data[:, 3]
        else:
            raise ValueError("配列は1次元またはOHLC形式の2次元である必要があります")
    else:
        raise ValueError("サポートされていないデータ型です")