#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import logging

class BaseSignalGenerator:
    """シグナル生成器の基底クラス"""
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: シグナル生成器の名前
        """
        self.name = name
        self._signals_cache: Dict[str, np.ndarray] = {}
        # ロガーの設定
        self.logger = logging.getLogger(f"signal_generator.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """全てのシグナルを計算してキャッシュする"""
        raise NotImplementedError
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        raise NotImplementedError
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        raise NotImplementedError
    
    def clear_cache(self) -> None:
        """シグナルのキャッシュをクリアする"""
        self._signals_cache.clear()
    
    def _get_cached_signal(self, key: str) -> Union[np.ndarray, None]:
        """キャッシュからシグナルを取得する"""
        return self._signals_cache.get(key)
    
    def _set_cached_signal(self, key: str, signal: np.ndarray) -> None:
        """シグナルをキャッシュに保存する"""
        self._signals_cache[key] = signal 