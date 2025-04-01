#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_roc import ZROC
from .hidden_divergence_signal import HiddenDivergenceSignal


class ZROCHiddenDivergenceSignal(BaseSignal, IEntrySignal):
    """
    ZROC隠れダイバージェンスシグナル
    
    価格とZROCの間の隠れダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気の隠れダイバージェンス（ロングエントリー）：
      価格が高値を切り上げているのに対し、ZROCが高値を切り下げている状態で、
      上昇トレンドの継続を示唆。
    
    - 弱気の隠れダイバージェンス（ショートエントリー）：
      価格が安値を切り下げているのに対し、ZROCが安値を切り上げている状態で、
      下降トレンドの継続を示唆。
    
    特徴:
    - サイクルベースの動的な期間設定
    - 効率比（ER）に基づいて適応的に調整される移動平均
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        ma_type: str = "zma",
        er_period: int = 10,
        max_dc_period: int = 55,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            ma_type: 移動平均のタイプ ('zma', 'sma', 'ema', 'wma', 'kama')
            er_period: 効率比の計算期間
            max_dc_period: 最大ドミナントサイクル期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("ZROCHiddenDivergence")
        
        # パラメータの設定
        self.ma_type = ma_type
        self.er_period = er_period
        self.max_dc_period = max_dc_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.z_roc = ZROC(
            ma_type=ma_type,
            er_period=er_period,
            max_dc_period=max_dc_period
        )
        self.hidden_divergence = HiddenDivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ZROC隠れダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        try:
            # データフレームの作成
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # ZROCの計算
            self.z_roc.calculate(data)
            
            # ZROCのデータを取得
            roc_values = self.z_roc.get_values()
            
            # 隠れダイバージェンスの検出
            signals = self.hidden_divergence.generate(df, roc_values)
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZROC隠れダイバージェンスシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はゼロシグナルを返す
            return np.zeros(len(data))
    
    def get_z_roc_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ZROCの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: ZROCの値
        """
        try:
            # ZROCの計算
            self.z_roc.calculate(data)
            
            # ZROCの値を取得
            return self.z_roc.get_values()
        except Exception as e:
            print(f"ZROC値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            # ZROCの計算
            self.z_roc.calculate(data)
            
            # 効率比の取得
            return self.z_roc.get_efficiency_ratio()
        except Exception as e:
            print(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_period(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的な期間の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的な期間の値
        """
        try:
            # ZROCの計算
            self.z_roc.calculate(data)
            
            # 動的な期間の取得
            return self.z_roc.get_period()
        except Exception as e:
            print(f"動的期間取得中にエラー: {str(e)}")
            return np.array([]) 