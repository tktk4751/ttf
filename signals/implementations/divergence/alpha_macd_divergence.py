#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_macd import AlphaMACD
from .divergence_signal import DivergenceSignal


class AlphaMACDDivergenceSignal(BaseSignal, IEntrySignal):
    """
    アルファMACDダイバージェンスシグナル
    
    価格とアルファMACDの間のダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、アルファMACDが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、アルファMACDが高値を切り下げている状態。
      下落転換の可能性を示唆。
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用したMACD
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        fast_max_kama_period: int = 89,
        fast_min_kama_period: int = 8,
        slow_max_kama_period: int = 144,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 55,
        signal_min_kama_period: int = 5,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間
            fast_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            fast_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            slow_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            slow_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            signal_max_kama_period: シグナルAlphaMAのKAMAピリオドの最大値
            signal_min_kama_period: シグナルAlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("AlphaMACDDivergence")
        
        # パラメータの設定
        self.er_period = er_period
        self.fast_max_kama_period = fast_max_kama_period
        self.fast_min_kama_period = fast_min_kama_period
        self.slow_max_kama_period = slow_max_kama_period
        self.slow_min_kama_period = slow_min_kama_period
        self.signal_max_kama_period = signal_max_kama_period
        self.signal_min_kama_period = signal_min_kama_period
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.alpha_macd = AlphaMACD(
            er_period=er_period,
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        self.divergence = DivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファMACDダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        try:
            # データフレームの作成
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # アルファMACDの計算
            alpha_macd_result = self.alpha_macd.calculate(data)
            
            # ダイバージェンスの検出（MACDラインを使用）
            signals = self.divergence.generate(df, alpha_macd_result.macd)
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"AlphaMACDダイバージェンスシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はゼロシグナルを返す
            return np.zeros(len(data))
    
    def get_alpha_macd_values(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        アルファMACDの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, np.ndarray]: アルファMACDの値（macd, signal, histogram）
        """
        try:
            # アルファMACDの計算
            result = self.alpha_macd.calculate(data)
            
            return {
                'macd': result.macd,
                'signal': result.signal,
                'histogram': result.histogram
            }
        except Exception as e:
            print(f"アルファMACD値取得中にエラー: {str(e)}")
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            # アルファMACDの計算
            self.alpha_macd.calculate(data)
            
            # 効率比の取得
            return self.alpha_macd.get_efficiency_ratio()
        except Exception as e:
            print(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        動的な期間の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, np.ndarray]: 動的な期間の値
        """
        try:
            # アルファMACDの計算
            self.alpha_macd.calculate(data)
            
            # 動的な期間の取得
            kama_period, fast_period, slow_period = self.alpha_macd.get_dynamic_periods()
            
            return {
                'kama_period': kama_period,
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        except Exception as e:
            print(f"動的期間取得中にエラー: {str(e)}")
            return {'kama_period': np.array([]), 'fast_period': np.array([]), 'slow_period': np.array([])} 