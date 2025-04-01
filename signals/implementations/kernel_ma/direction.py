#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.kernel_ma import KernelMA


@njit(cache=True)
def calculate_direction_from_slope(slope: np.ndarray, threshold: float = 0.0001) -> np.ndarray:
    """傾きから方向シグナルを計算する（1=上昇、-1=下降、0=横ばい）

    Parameters
    ----------
    slope : np.ndarray
        傾き配列
    threshold : float, default=0.0001
        傾きの閾値

    Returns
    -------
    np.ndarray
        方向シグナル配列
    """
    direction = np.zeros_like(slope)
    direction[slope > threshold] = 1     # 上昇トレンド
    direction[slope < -threshold] = -1   # 下降トレンド
    return direction


class KernelMADirectionSignal(BaseSignal, IDirectionSignal):
    """
    カーネル回帰移動平均線に基づく方向性シグナル

    カーネル回帰法を用いた適応型移動平均線インジケーターを使用して、
    トレンド方向シグナルを生成します。効率比（ER）に基づいた動的バンド幅調整と
    内部キャッシュによる高速化を実装しています。
    
    シグナルは1（上昇トレンド/ロング）、-1（下降トレンド/ショート）、
    または0（横ばい/ニュートラル）を返します。
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_bandwidth: float = 10.0,
        min_bandwidth: float = 2.0,
        kernel_type: str = 'gaussian',
        confidence_level: float = 0.95,
        slope_period: int = 5,
        slope_threshold: float = 0.0001
    ):
        """コンストラクタ

        Parameters
        ----------
        er_period : int, default=21
            効率比の計算期間
        max_bandwidth : float, default=10.0
            バンド幅の最大値
        min_bandwidth : float, default=2.0
            バンド幅の最小値
        kernel_type : str, default='gaussian'
            カーネルの種類（'gaussian'または'epanechnikov'）
        confidence_level : float, default=0.95
            信頼区間のレベル
        slope_period : int, default=5
            傾きを計算する期間
        slope_threshold : float, default=0.0001
            トレンド判定の傾き閾値
        """
        params = {
            'er_period': er_period,
            'max_bandwidth': max_bandwidth,
            'min_bandwidth': min_bandwidth,
            'kernel_type': kernel_type,
            'confidence_level': confidence_level,
            'slope_period': slope_period,
            'slope_threshold': slope_threshold
        }
        super().__init__(
            f"KernelMADirection({er_period}, {max_bandwidth}, {min_bandwidth}, "
            f"{kernel_type}, {slope_period}, {slope_threshold})",
            params
        )
        
        # KernelMAインジケーターのインスタンス化
        self.kernel_ma = KernelMA(
            er_period=er_period,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            kernel_type=kernel_type,
            confidence_level=confidence_level,
            slope_period=slope_period
        )
        
        self.slope_threshold = slope_threshold
        
        # 内部キャッシュの初期化（最適化用）
        self._cache = {}
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """カーネル回帰移動平均線に基づく方向性シグナルを生成する

        上昇トレンドでは1（ロング）、下降トレンドでは-1（ショート）、
        横ばいでは0（ニュートラル）を返します。
        内部キャッシュを使用して、同一データに対する計算を高速化します。

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ。DataFrameの場合は'close'カラムを使用。
            ndarrayの場合は[close]の1次元配列または[open, high, low, close]の2次元配列が必要。

        Returns
        -------
        np.ndarray
            方向性シグナルの配列。1=ロング、-1=ショート、0=ニュートラル。
        """
        # キャッシュキーの生成
        cache_key = self._get_data_hash(data)
        
        # キャッシュされた結果がある場合はそれを返す
        if cache_key in self._cache:
            return self._cache[cache_key]['direction']
            
        try:
            # データの変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                # DataFrameでない場合は配列と仮定
                if data.ndim == 1:
                    close = data  # 1次元配列として扱う
                elif data.ndim == 2 and data.shape[1] >= 4:
                    close = data[:, 3]  # close
                else:
                    raise ValueError("ndarrayは1次元配列または[open, high, low, close]を含む2次元配列である必要があります")
                
            # データフレームを作成してKernelMAの計算に渡す
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame({'close': close})
                self.kernel_ma.calculate(df)
            else:
                self.kernel_ma.calculate(data)
            
            # 傾きを取得
            slope = self.kernel_ma.get_slope()
            
            # 方向シグナルの計算
            direction = calculate_direction_from_slope(slope, self.slope_threshold)
            
            # 結果をキャッシュ
            self._cache[cache_key] = {
                'direction': direction,
                'values': self.kernel_ma._values,
                'upper_band': self.kernel_ma.get_bands()[0],
                'lower_band': self.kernel_ma.get_bands()[1],
                'er': self.kernel_ma.get_efficiency_ratio(),
                'bandwidth': self.kernel_ma.get_bandwidth(),
                'slope': slope
            }
            
            return direction
            
        except Exception as e:
            # エラーが発生した場合はゼロ配列を返す
            print(f"KernelMADirectionSignal生成エラー: {str(e)}")
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(len(data)) if data.ndim > 0 else np.zeros(1)
    
    def get_ma_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """KernelMAの値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            KernelMAの値の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'values' in self._cache[cache_key]:
            return self._cache[cache_key]['values']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['values'] if cache_key in self._cache else np.array([])
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """KernelMAの上バンド値を取得する

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
        """KernelMAの下バンド値を取得する

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
        """KernelMAの効率比(ER)値を取得する

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
    
    def get_bandwidth(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """KernelMAの動的バンド幅を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            動的バンド幅の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'bandwidth' in self._cache[cache_key]:
            return self._cache[cache_key]['bandwidth']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['bandwidth'] if cache_key in self._cache else np.array([])
    
    def get_slope(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """KernelMAの傾きを取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            傾きの配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'slope' in self._cache[cache_key]:
            return self._cache[cache_key]['slope']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['slope'] if cache_key in self._cache else np.array([])
    
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