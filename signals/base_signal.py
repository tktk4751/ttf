#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from .signal_interfaces import ISignal, ISignalParameters

class BaseSignal(ISignal, ISignalParameters):
    """シグナルの基底クラス"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            name: シグナルの名前
            params: パラメータ辞書
        """
        self.name = name
        self._params = params or {}
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        return self._params.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        self._params.update(params)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """シグナルを生成する"""
        raise NotImplementedError 