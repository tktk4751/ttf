#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_macd import ZMACD
from .hidden_divergence_signal import HiddenDivergenceSignal


class ZMACDHiddenDivergenceSignal(BaseSignal, IEntrySignal):
    """
    ZMACD隠れダイバージェンスシグナル
    
    価格とZMACDの間の隠れダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気の隠れダイバージェンス（ロングエントリー）：
      価格が高値を切り上げているのに対し、ZMACDが高値を切り下げている状態で、
      上昇トレンドの継続を示唆。
    
    - 弱気の隠れダイバージェンス（ショートエントリー）：
      価格が安値を切り下げているのに対し、ZMACDが安値を切り上げている状態で、
      下降トレンドの継続を示唆。
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるZMAを使用したMACD
    - ドミナントサイクルを用いた動的な期間計算
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        # 短期線用パラメータ
        fast_max_dc_max_output: int = 21,
        fast_max_dc_min_output: int = 5,
        
        # 長期線用パラメータ
        slow_max_dc_max_output: int = 55,
        slow_max_dc_min_output: int = 13,
        
        # シグナル線用パラメータ
        signal_max_dc_max_output: int = 21,
        signal_max_dc_min_output: int = 5,
        
        max_slow_period: int = 34,
        min_slow_period: int = 13,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間
            fast_max_dc_max_output: 短期線用最大期間出力値
            fast_max_dc_min_output: 短期線用最大期間出力の最小値
            slow_max_dc_max_output: 長期線用最大期間出力値
            slow_max_dc_min_output: 長期線用最大期間出力の最小値
            signal_max_dc_max_output: シグナル線用最大期間出力値
            signal_max_dc_min_output: シグナル線用最大期間出力の最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("ZMACDHiddenDivergence")
        
        # パラメータの設定
        self.er_period = er_period
        self.fast_max_dc_max_output = fast_max_dc_max_output
        self.fast_max_dc_min_output = fast_max_dc_min_output
        self.slow_max_dc_max_output = slow_max_dc_max_output
        self.slow_max_dc_min_output = slow_max_dc_min_output
        self.signal_max_dc_max_output = signal_max_dc_max_output
        self.signal_max_dc_min_output = signal_max_dc_min_output
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.z_macd = ZMACD(
            er_period=er_period,
            # 短期線用パラメータ
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            # 長期線用パラメータ
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            # シグナル線用パラメータ
            signal_max_dc_max_output=signal_max_dc_max_output,
            signal_max_dc_min_output=signal_max_dc_min_output,
            # その他のパラメータ
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        self.hidden_divergence = HiddenDivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ZMACD隠れダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        try:
            # データフレームの作成
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # ZMACDの計算
            self.z_macd.calculate(data)
            
            # MACD, シグナル, ヒストグラムを取得
            macd_line, _, _ = self.z_macd.get_lines()
            
            # 隠れダイバージェンスの検出（MACDラインを使用）
            signals = self.hidden_divergence.generate(df, macd_line)
            
            return signals
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"ZMACD隠れダイバージェンスシグナル生成中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はゼロシグナルを返す
            return np.zeros(len(data))
    
    def get_z_macd_values(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        ZMACDの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Dict[str, np.ndarray]: ZMACDの値（macd, signal, histogram）
        """
        try:
            # ZMACDの計算
            self.z_macd.calculate(data)
            
            # MACD, シグナル, ヒストグラムを取得
            macd, signal, histogram = self.z_macd.get_lines()
            
            return {
                'macd': macd,
                'signal': signal,
                'histogram': histogram
            }
        except Exception as e:
            print(f"ZMACD値取得中にエラー: {str(e)}")
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
            # ZMACDの計算
            self.z_macd.calculate(data)
            
            # 効率比の取得
            return self.z_macd.get_efficiency_ratio()
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
            # ZMACDの計算
            self.z_macd.calculate(data)
            
            # 動的な期間の取得
            fast_period, slow_period, signal_period = self.z_macd.get_dynamic_periods()
            
            return {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
        except Exception as e:
            print(f"動的期間取得中にエラー: {str(e)}")
            return {'fast_period': np.array([]), 'slow_period': np.array([]), 'signal_period': np.array([])} 