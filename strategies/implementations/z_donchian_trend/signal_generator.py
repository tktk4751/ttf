#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_donchian.entry import ZDonchianBreakoutEntrySignal
from signals.implementations.z_trend_filter.filter import ZTrendFilterSignal


@njit(fastmath=True, parallel=True)
def calculate_entry_signals(z_donchian: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """
    エントリーシグナルを一度に計算（高速化版）
    
    Args:
        z_donchian: Zドンチャンチャネルのシグナル配列 (1:ロング, -1:ショート, 0:ニュートラル)
        filter_signal: Zトレンドフィルターのシグナル配列 (1:トレンド相場, -1:レンジ相場)
    
    Returns:
        np.ndarray: 組み合わせたエントリーシグナル (1:ロング, -1:ショート, 0:ニュートラル)
    """
    signals = np.zeros_like(z_donchian, dtype=np.int8)
    
    # ロングエントリー: Zドンチャンの買いシグナル(1) + Zトレンドフィルターがトレンド相場(1)
    long_condition = (z_donchian == 1)
    
    # ショートエントリー: Zドンチャンの売りシグナル(-1) + Zトレンドフィルターがトレンド相場(1)
    short_condition = (z_donchian == -1) 
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class ZDonchianTrendSignalGenerator(BaseSignalGenerator):
    """
    Zドンチャン+Zトレンドフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: Zドンチャンのブレイクアウトで買いシグナル + Zトレンドフィルターがトレンド相場
    - ショート: Zドンチャンのブレイクアウトで売りシグナル + Zトレンドフィルターがトレンド相場
    
    エグジット条件:
    - ロング: Zドンチャンの売りシグナル
    - ショート: Zドンチャンの買いシグナル
    """
    
    def __init__(
        self,
        # 共通パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        src_type: str = 'hlc3',
        
        # Zドンチャンチャネル用パラメータ
        # 最大期間用パラメータ
        max_dc_cycle_part: float = 0.5,
        max_dc_max_cycle: int = 144,
        max_dc_min_cycle: int = 5,
        max_dc_max_output: int = 89,
        max_dc_min_output: int = 21,
        
        # 最小期間用パラメータ
        min_dc_cycle_part: float = 0.25,
        min_dc_max_cycle: int = 55,
        min_dc_min_cycle: int = 5,
        min_dc_max_output: int = 21,
        min_dc_min_output: int = 8,
        
        # ブレイクアウトパラメータ
        lookback: int = 1,
        
        # Zトレンドフィルター用パラメータ
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        combination_weight: float = 0.6,
        zadx_weight: float = 0.4,
        combination_method: str = "sigmoid",
        
        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5
    ):
        """初期化"""
        super().__init__("ZDonchianTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'src_type': src_type,
            'max_dc_cycle_part': max_dc_cycle_part,
            'max_dc_max_cycle': max_dc_max_cycle,
            'max_dc_min_cycle': max_dc_min_cycle,
            'max_dc_max_output': max_dc_max_output,
            'max_dc_min_output': max_dc_min_output,
            'min_dc_cycle_part': min_dc_cycle_part,
            'min_dc_max_cycle': min_dc_max_cycle,
            'min_dc_min_cycle': min_dc_min_cycle,
            'min_dc_max_output': min_dc_max_output,
            'min_dc_min_output': min_dc_min_output,
            'lookback': lookback,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_weight': combination_weight,
            'zadx_weight': zadx_weight,
            'combination_method': combination_method,
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output
        }
        
        # Zドンチャンブレイクアウトシグナルの初期化
        self.z_donchian_signal = ZDonchianBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=max_dc_max_output,
            max_dc_min_output=max_dc_min_output,
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=min_dc_max_output,
            min_dc_min_output=min_dc_min_output,
            lookback=lookback,
            src_type=src_type
        )
        
        # Zトレンドフィルターシグナルの初期化
        self.z_trend_filter_signal = ZTrendFilterSignal(
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            combination_weight=combination_weight,
            zadx_weight=zadx_weight,
            combination_method=combination_method,
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output
        )
        
        # シグナルキャッシュのキー
        self._entry_key = "entry_signals"
        self._z_donchian_key = "z_donchian_signals"
        self._filter_key = "filter_signals"
        
        # データハッシュの保存用
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        import hashlib
        
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{hash(frozenset(self._params.items()))}"
        
        return f"{data_hash}_{param_str}"
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        全てのシグナルを計算してキャッシュする
        
        Args:
            data: 価格データ
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._get_cached_signal(self._entry_key) is not None:
                return
                
            self._data_hash = data_hash
            
            # Zドンチャンシグナルの生成
            z_donchian_signals = self.z_donchian_signal.generate(data)
            self._set_cached_signal(self._z_donchian_key, z_donchian_signals)
            
            # Zトレンドフィルターシグナルの生成
            filter_signals = self.z_trend_filter_signal.generate(data)
            self._set_cached_signal(self._filter_key, filter_signals)
            
            # エントリーシグナルの生成（高速化版）
            entry_signals = calculate_entry_signals(z_donchian_signals, filter_signals)
            self._set_cached_signal(self._entry_key, entry_signals)
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"シグナル計算中にエラー: {error_msg}\n{stack_trace}")
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル (1:ロング, -1:ショート, 0:ニュートラル)
        """
        # キャッシュにないか古い場合は再計算
        self.calculate_signals(data)
        
        # キャッシュから取得
        entry_signals = self._get_cached_signal(self._entry_key)
        if entry_signals is None:
            self.logger.warning("エントリーシグナルがキャッシュにありません。計算に失敗した可能性があります。")
            return np.zeros(len(data), dtype=np.int8)
            
        return entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1:ロング, -1:ショート)
            index: チェックするインデックス（デフォルト: -1=最新）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if position == 0:
            return False
            
        # キャッシュにないか古い場合は再計算
        self.calculate_signals(data)
        
        # Zドンチャンシグナルを取得
        z_donchian_signals = self._get_cached_signal(self._z_donchian_key)
        if z_donchian_signals is None:
            self.logger.warning("Zドンチャンシグナルがキャッシュにありません。計算に失敗した可能性があります。")
            return False
        
        # インデックスの調整
        if index < 0:
            index = len(z_donchian_signals) + index
        
        if index < 0 or index >= len(z_donchian_signals):
            return False
        
        # ロングポジションのエグジット: Zドンチャンの売りシグナル(-1)
        if position == 1 and z_donchian_signals[index] == -1:
            return True
        
        # ショートポジションのエグジット: Zドンチャンの買いシグナル(1)
        if position == -1 and z_donchian_signals[index] == 1:
            return True
        
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zドンチャンチャネルのバンド値を取得
        
        Args:
            data: 価格データ（Noneの場合はキャッシュから取得を試みる）
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (上限バンド, 下限バンド, 中央線)
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.z_donchian_signal.get_band_values()
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）を取得
        
        Args:
            data: 価格データ（Noneの場合はキャッシュから取得を試みる）
            
        Returns:
            np.ndarray: サイクル効率比の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.z_donchian_signal.get_efficiency_ratio()
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Zトレンドフィルター値を取得
        
        Args:
            data: 価格データ（Noneの場合はキャッシュから取得を試みる）
            
        Returns:
            np.ndarray: フィルター値の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.z_trend_filter_signal.get_filter_values()
    
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的しきい値を取得
        
        Args:
            data: 価格データ（Noneの場合はキャッシュから取得を試みる）
            
        Returns:
            np.ndarray: 動的しきい値の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.z_trend_filter_signal.get_threshold_values()
    
    def get_dynamic_period(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的なドンチャン期間を取得
        
        Args:
            data: 価格データ（Noneの場合はキャッシュから取得を試みる）
            
        Returns:
            np.ndarray: 動的な期間の値
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.z_donchian_signal.get_dynamic_period() 