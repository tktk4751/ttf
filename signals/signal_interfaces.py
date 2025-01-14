#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Union, Dict, Any
import numpy as np
import pandas as pd

class ISignal(Protocol):
    """シグナル生成の基本インターフェース"""
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """シグナルを生成する"""
        ...

class IDirectionSignal(ISignal, Protocol):
    """方向性シグナルのインターフェース"""
    pass

class IEntrySignal(ISignal, Protocol):
    """エントリーシグナルのインターフェース"""
    pass

class IExitSignal(ISignal, Protocol):
    """エグジットシグナルのインターフェース"""
    pass

class IFilterSignal(ISignal, Protocol):
    """フィルターシグナルのインターフェース"""
    pass

class ISignalParameters(Protocol):
    """シグナルパラメータのインターフェース"""
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        ... 