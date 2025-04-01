#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.alpha_trend import AlphaTrend


@njit(cache=True)
def calculate_direction_from_trend(trend: np.ndarray) -> np.ndarray:
    """トレンド配列から方向シグナルを計算する（1=上昇、-1=下降）

    Parameters
    ----------
    trend : np.ndarray
        トレンド配列（1=上昇、-1=下降）

    Returns
    -------
    np.ndarray
        方向シグナル配列
    """
    return trend


class AlphaTrendDirectionSignal(BaseSignal, IDirectionSignal):
    """
    AlphaTrendに基づく方向性シグナル

    AlphaTrendインジケーターを使用して、トレンド方向シグナルを生成します。
    効率比（ER）に基づいた動的パラメータ調整と内部キャッシュによる高速化を実装しています。
    シグナルは1（上昇トレンド/ロング）または-1（下降トレンド/ショート）を返します。
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_percentile_length: int = 55,
        min_percentile_length: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        alma_offset: float = 0.85,
        alma_sigma: int = 6
    ):
        """コンストラクタ

        Parameters
        ----------
        er_period : int, default=21
            効率比の期間
        max_percentile_length : int, default=55
            パーセンタイル計算の最大期間
        min_percentile_length : int, default=13
            パーセンタイル計算の最小期間
        max_atr_period : int, default=89
            Alpha ATR期間の最大値
        min_atr_period : int, default=13
            Alpha ATR期間の最小値
        max_multiplier : float, default=3.0
            ATR乗数の最大値
        min_multiplier : float, default=1.0
            ATR乗数の最小値
        alma_offset : float, default=0.85
            ALMAオフセット
        alma_sigma : int, default=6
            ALMAシグマ
        """
        params = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma
        }
        super().__init__(
            f"AlphaTrendDirection({er_period}, {max_percentile_length}, {min_percentile_length}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier}, "
            f"{alma_offset}, {alma_sigma})",
            params
        )
        
        # AlphaTrendインジケーターのインスタンス化
        self.alpha_trend = AlphaTrend(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        
        # 内部キャッシュの初期化（最適化用）
        self._cache = {}
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendに基づく方向性シグナルを生成する

        上昇トレンドでは1（ロング）、下降トレンドでは-1（ショート）を返します。
        内部キャッシュを使用して、同一データに対する計算を高速化します。

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ。DataFrameの場合は'high', 'low', 'close'カラムを使用。
            ndarrayの場合は[high, low, close]の3次元配列が必要。

        Returns
        -------
        np.ndarray
            方向性シグナルの配列。1=ロング、-1=ショート。
        """
        # キャッシュキーの生成
        cache_key = self._get_data_hash(data)
        
        # キャッシュされた結果がある場合はそれを返す
        if cache_key in self._cache:
            return self._cache[cache_key]['direction']
            
        try:
            # データの変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                # DataFrameでない場合は3次元配列と仮定
                if data.ndim != 2 or data.shape[1] < 3:
                    raise ValueError("ndarrayは[high, low, close]を含む2次元配列である必要があります")
                high = data[:, 0]
                low = data[:, 1]
                close = data[:, 2]
                
            # データフレームを作成してAlphaTrendの計算に渡す
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame({
                    'high': high,
                    'low': low,
                    'close': close
                })
                self.alpha_trend.calculate(df)
            else:
                self.alpha_trend.calculate(data)
            
            # トレンド方向を取得
            trend = self.alpha_trend.get_trend()
            
            # 方向シグナルの計算
            direction = calculate_direction_from_trend(trend)
            
            # 結果をキャッシュ
            self._cache[cache_key] = {
                'direction': direction,
                'upper_band': self.alpha_trend.get_bands()[0],
                'lower_band': self.alpha_trend.get_bands()[1],
                'er': self.alpha_trend.get_efficiency_ratio(),
                'dynamic_multiplier': self.alpha_trend.get_dynamic_parameters()[0],
                'dynamic_percentile_length': self.alpha_trend.get_dynamic_parameters()[1]
            }
            
            return direction
            
        except Exception as e:
            # エラーが発生した場合はゼロ配列を返す
            print(f"AlphaTrendDirectionSignal生成エラー: {str(e)}")
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(len(data)) if data.ndim == 2 else np.zeros(1)
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendの上バンド値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            上バンド値の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'upper_band' in self._cache[cache_key]:
            return self._cache[cache_key]['upper_band']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['upper_band'] if cache_key in self._cache else np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendの下バンド値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            下バンド値の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'lower_band' in self._cache[cache_key]:
            return self._cache[cache_key]['lower_band']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['lower_band'] if cache_key in self._cache else np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendの効率比(ER)値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            効率比の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'er' in self._cache[cache_key]:
            return self._cache[cache_key]['er']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['er'] if cache_key in self._cache else np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendの動的乗数を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            動的乗数の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'dynamic_multiplier' in self._cache[cache_key]:
            return self._cache[cache_key]['dynamic_multiplier']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['dynamic_multiplier'] if cache_key in self._cache else np.array([])
    
    def get_dynamic_percentile_length(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """AlphaTrendの動的パーセンタイル期間を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            動的パーセンタイル期間の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'dynamic_percentile_length' in self._cache[cache_key]:
            return self._cache[cache_key]['dynamic_percentile_length']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['dynamic_percentile_length'] if cache_key in self._cache else np.array([])
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データからハッシュキーを生成する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            ハッシュを計算するデータ

        Returns
        -------
        str
            データに対するハッシュキー
        """
        if isinstance(data, pd.DataFrame):
            # DataFrameのハッシュはメモリアドレスとデータ長の組み合わせ
            return f"{id(data)}_{len(data)}"
        else:
            # ndarrayのハッシュはメモリアドレスとデータ形状の組み合わせ
            return f"{id(data)}_{data.shape}" 