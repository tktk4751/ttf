#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Union
import numpy as np
import pandas as pd

class IExitSignal(Protocol):
    """エグジットシグナルのインターフェース"""
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """シグナルを生成する"""
        ... 