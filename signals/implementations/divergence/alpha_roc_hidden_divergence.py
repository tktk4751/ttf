#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_roc import AlphaROC
from indicators.hidden_divergence import HiddenDivergence


class AlphaROCHiddenDivergenceSignal(BaseSignal, IEntrySignal):
    """
    アルファROCヒドゥンダイバージェンスシグナル
    
    価格とアルファROCの間のヒドゥンダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ヒドゥンダイバージェンス（ロングエントリー）：
      価格が安値を切り上げているのに対し、アルファROCが安値を切り下げている状態。
      上昇トレンド継続の可能性を示唆。
    
    - 弱気ヒドゥンダイバージェンス（ショートエントリー）：
      価格が高値を切り下げているのに対し、アルファROCが高値を切り上げている状態。
      下降トレンド継続の可能性を示唆。
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるROC期間
    - トレンドが強い時は短い期間で素早く反応
    - レンジ相場時は長い期間でノイズを除去
    - トレンド継続の確認に有効
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_roc_period: int = 50,
        min_roc_period: int = 5,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_roc_period: ROC期間の最大値（デフォルト: 50）
            min_roc_period: ROC期間の最小値（デフォルト: 5）
            lookback: ヒドゥンダイバージェンス検出のルックバック期間（デフォルト: 30）
            params: その他のパラメータ（オプション）
        """
        super().__init__("AlphaROCHiddenDivergence")
        
        # パラメータの設定
        self.er_period = er_period
        self.max_roc_period = max_roc_period
        self.min_roc_period = min_roc_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.alpha_roc = AlphaROC(
            er_period=er_period,
            max_roc_period=max_roc_period,
            min_roc_period=min_roc_period
        )
        self.hidden_divergence = HiddenDivergence(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファROCヒドゥンダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # アルファROCの計算
            alpha_roc_result = self.alpha_roc.calculate(data)
            
            # ヒドゥンダイバージェンスの検出
            signals = self.hidden_divergence.calculate(close, alpha_roc_result.roc)
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"AlphaROCヒドゥンダイバージェンスシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はゼロシグナルを返す
            return np.zeros(len(data))
    
    def get_alpha_roc_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファROCの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: アルファROCの値
        """
        try:
            # アルファROCの計算
            result = self.alpha_roc.calculate(data)
            
            return result.roc
        except Exception as e:
            print(f"アルファROC値取得中にエラー: {str(e)}")
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
            # アルファROCの計算
            self.alpha_roc.calculate(data)
            
            # 効率比の取得
            return self.alpha_roc.get_efficiency_ratio()
        except Exception as e:
            print(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_period(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的なROC期間の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的なROC期間の値
        """
        try:
            # アルファROCの計算
            self.alpha_roc.calculate(data)
            
            # 動的な期間の取得
            return self.alpha_roc.get_dynamic_period()
        except Exception as e:
            print(f"動的期間取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_divergence_states(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        ヒドゥンダイバージェンスの状態を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, np.ndarray]: ヒドゥンダイバージェンスの状態
                {'bullish': 強気ヒドゥンダイバージェンス, 'bearish': 弱気ヒドゥンダイバージェンス}
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # アルファROCの計算
            alpha_roc_result = self.alpha_roc.calculate(data)
            
            # ヒドゥンダイバージェンスの検出
            self.hidden_divergence.calculate(close, alpha_roc_result.roc)
            
            # 状態の取得
            bullish, bearish = self.hidden_divergence.get_divergence_states()
            
            return {
                'bullish': bullish,
                'bearish': bearish
            }
        except Exception as e:
            print(f"ヒドゥンダイバージェンス状態取得中にエラー: {str(e)}")
            return {'bullish': np.array([]), 'bearish': np.array([])} 