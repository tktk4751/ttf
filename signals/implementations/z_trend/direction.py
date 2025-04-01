#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.z_trend import ZTrend


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


class ZTrendDirectionSignal(BaseSignal, IDirectionSignal):
    """
    ZTrendに基づく方向性シグナル

    ZTrendインジケーターを使用して、トレンド方向シグナルを生成します。
    サイクル効率比（CER）に基づいた動的パラメータ調整と内部キャッシュによる高速化を実装しています。
    シグナルは1（上昇トレンド/ロング）または-1（下降トレンド/ショート）を返します。
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 233,
        cer_min_cycle: int = 13,
        cer_max_output: int = 144,
        cer_min_output: int = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        # ATR乗数
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """コンストラクタ

        Parameters
        ----------
        cycle_detector_type : str, default='hody_dc'
            サイクル検出器のタイプ
        lp_period : int, default=13
            ローパスフィルター期間
        hp_period : int, default=144
            ハイパスフィルター期間
        cycle_part : float, default=0.5
            サイクル部分の倍率
        cer_max_cycle : int, default=233
            CER用ドミナントサイクル検出器の最大サイクル
        cer_min_cycle : int, default=13
            CER用ドミナントサイクル検出器の最小サイクル
        cer_max_output : int, default=144
            CER用ドミナントサイクル検出器の最大出力
        cer_min_output : int, default=21
            CER用ドミナントサイクル検出器の最小出力
        max_percentile_dc_cycle_part : float, default=0.5
            最大パーセンタイル期間用DCのサイクル部分
        max_percentile_dc_max_cycle : int, default=233
            最大パーセンタイル期間用DCの最大サイクル
        max_percentile_dc_min_cycle : int, default=13
            最大パーセンタイル期間用DCの最小サイクル
        max_percentile_dc_max_output : int, default=144
            最大パーセンタイル期間用DCの最大出力
        max_percentile_dc_min_output : int, default=21
            最大パーセンタイル期間用DCの最小出力
        min_percentile_dc_cycle_part : float, default=0.5
            最小パーセンタイル期間用DCのサイクル部分
        min_percentile_dc_max_cycle : int, default=55
            最小パーセンタイル期間用DCの最大サイクル
        min_percentile_dc_min_cycle : int, default=5
            最小パーセンタイル期間用DCの最小サイクル
        min_percentile_dc_max_output : int, default=34
            最小パーセンタイル期間用DCの最大出力
        min_percentile_dc_min_output : int, default=8
            最小パーセンタイル期間用DCの最小出力
        zatr_max_dc_cycle_part : float, default=0.5
            ZATR最大DCのサイクル部分
        zatr_max_dc_max_cycle : int, default=55
            ZATR最大DCの最大サイクル
        zatr_max_dc_min_cycle : int, default=5
            ZATR最大DCの最小サイクル
        zatr_max_dc_max_output : int, default=55
            ZATR最大DCの最大出力
        zatr_max_dc_min_output : int, default=5
            ZATR最大DCの最小出力
        zatr_min_dc_cycle_part : float, default=0.25
            ZATR最小DCのサイクル部分
        zatr_min_dc_max_cycle : int, default=34
            ZATR最小DCの最大サイクル
        zatr_min_dc_min_cycle : int, default=3
            ZATR最小DCの最小サイクル
        zatr_min_dc_max_output : int, default=13
            ZATR最小DCの最大出力
        zatr_min_dc_min_output : int, default=3
            ZATR最小DCの最小出力
        max_percentile_cycle_mult : float, default=0.5
            最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult : float, default=0.25
            最小パーセンタイル期間のサイクル乗数
        max_multiplier : float, default=3.0
            ATR乗数の最大値
        min_multiplier : float, default=1.0
            ATR乗数の最小値
        smoother_type : str, default='alma'
            平滑化アルゴリズム（'alma'または'hyper'）
        src_type : str, default='hlc3'
            ソースタイプ
        """
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'max_percentile_dc_cycle_part': max_percentile_dc_cycle_part,
            'max_percentile_dc_max_cycle': max_percentile_dc_max_cycle,
            'max_percentile_dc_min_cycle': max_percentile_dc_min_cycle,
            'max_percentile_dc_max_output': max_percentile_dc_max_output,
            'max_percentile_dc_min_output': max_percentile_dc_min_output,
            'min_percentile_dc_cycle_part': min_percentile_dc_cycle_part,
            'min_percentile_dc_max_cycle': min_percentile_dc_max_cycle,
            'min_percentile_dc_min_cycle': min_percentile_dc_min_cycle,
            'min_percentile_dc_max_output': min_percentile_dc_max_output,
            'min_percentile_dc_min_output': min_percentile_dc_min_output,
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            'max_percentile_cycle_mult': max_percentile_cycle_mult,
            'min_percentile_cycle_mult': min_percentile_cycle_mult,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        super().__init__(
            f"ZTrendDirection({cycle_detector_type}, {lp_period}, {hp_period}, {cycle_part}, "
            f"{max_multiplier}, {min_multiplier}, {smoother_type}, {src_type})",
            params
        )
        
        # ZTrendインジケーターのインスタンス化
        self.z_trend = ZTrend(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type
        )
        
        # 内部キャッシュの初期化（最適化用）
        self._cache = {}
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ZTrendに基づく方向性シグナルを生成する

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
                
            # データフレームを作成してZTrendの計算に渡す
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame({
                    'high': high,
                    'low': low,
                    'close': close
                })
                z_trend_result = self.z_trend.calculate(df)
            else:
                z_trend_result = self.z_trend.calculate(data)
            
            # トレンド方向を取得
            trend = self.z_trend.get_trend()
            
            # 方向シグナルの計算
            direction = calculate_direction_from_trend(trend)
            
            # 結果をキャッシュ
            self._cache[cache_key] = {
                'direction': direction,
                'upper_band': self.z_trend.get_bands()[0],
                'lower_band': self.z_trend.get_bands()[1],
                'cer': self.z_trend.get_cycle_er(),
                'dynamic_multiplier': self.z_trend.get_dynamic_parameters()[0],
                'dynamic_percentile_length': self.z_trend.get_dynamic_parameters()[1],
                'percentiles': self.z_trend.get_percentiles(),
                'z_atr': self.z_trend.get_z_atr(),
                'dominant_cycles': self.z_trend.get_dominant_cycles()
            }
            
            return direction
            
        except Exception as e:
            # エラーが発生した場合はゼロ配列を返す
            print(f"ZTrendDirectionSignal生成エラー: {str(e)}")
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(len(data)) if data.ndim == 2 else np.zeros(1)
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ZTrendの上バンド値を取得する

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
        """ZTrendの下バンド値を取得する

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
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ZTrendのサイクル効率比(CER)値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            サイクル効率比の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'cer' in self._cache[cache_key]:
            return self._cache[cache_key]['cer']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['cer'] if cache_key in self._cache else np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ZTrendの動的乗数を取得する

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
        """ZTrendの動的パーセンタイル期間を取得する

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
    
    def get_percentiles(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """ZTrendのパーセンタイル値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (下限パーセンタイル, 上限パーセンタイル)の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'percentiles' in self._cache[cache_key]:
            return self._cache[cache_key]['percentiles']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['percentiles'] if cache_key in self._cache else (np.array([]), np.array([]))
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ZTrendのZATR値を取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        np.ndarray
            ZATR値の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'z_atr' in self._cache[cache_key]:
            return self._cache[cache_key]['z_atr']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['z_atr'] if cache_key in self._cache else np.array([])
    
    def get_dominant_cycles(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """ZTrendのドミナントサイクルを取得する

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (最大DCサイクル, 最小DCサイクル)の配列
        """
        cache_key = self._get_data_hash(data)
        
        if cache_key in self._cache and 'dominant_cycles' in self._cache[cache_key]:
            return self._cache[cache_key]['dominant_cycles']
            
        # キャッシュにない場合はシグナルを生成して結果を取得
        self.generate(data)
        return self._cache[cache_key]['dominant_cycles'] if cache_key in self._cache else (np.array([]), np.array([]))
    
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