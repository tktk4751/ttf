#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_trend.direction import ZTrendDirectionSignal


@njit(fastmath=True, parallel=True)
def calculate_entry_signals(direction_signal: np.ndarray) -> np.ndarray:
    """
    エントリーシグナルを一度に計算（高速化版）
    
    Args:
        direction_signal: ZTrendDirectionSignalの配列（1=上昇トレンド、-1=下降トレンド）
    
    Returns:
        np.ndarray: エントリーシグナルの配列 (1=ロング、-1=ショート、0=ニュートラル)
    """
    # ZTrendFilterを使用せず、ZTrendDirectionSignalをそのままエントリーシグナルとして使用
    return np.copy(direction_signal)


class SimpleZTrendSignalGenerator(BaseSignalGenerator):
    """
    シンプルなZトレンドシグナル生成クラス（ZTrendFilterなし）
    
    エントリー条件:
    - ロング: ZTrendDirectionSignalが上昇トレンド(1)
    - ショート: ZTrendDirectionSignalが下降トレンド(-1)
    
    エグジット条件:
    - ロング: ZTrendDirectionSignalが下降トレンド(-1)
    - ショート: ZTrendDirectionSignalが上昇トレンド(1)
    """
    
    def __init__(
        self,
        # ZTrendDirectionSignalのパラメータ
        cycle_detector_type: str = 'dudi_dc',
        lp_period: int = 5,
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
        
        # 動的乗数の範囲
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.3,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            cer_max_cycle: CER用最大サイクル
            cer_min_cycle: CER用最小サイクル
            cer_max_output: CER用最大出力
            cer_min_output: CER用最小出力
            max_percentile_dc_cycle_part: 最大パーセンタイル期間用DCの周期部分
            max_percentile_dc_max_cycle: 最大パーセンタイル期間用DCの最大周期
            max_percentile_dc_min_cycle: 最大パーセンタイル期間用DCの最小周期
            max_percentile_dc_max_output: 最大パーセンタイル期間用DCの最大出力
            max_percentile_dc_min_output: 最大パーセンタイル期間用DCの最小出力
            min_percentile_dc_cycle_part: 最小パーセンタイル期間用DCの周期部分
            min_percentile_dc_max_cycle: 最小パーセンタイル期間用DCの最大周期
            min_percentile_dc_min_cycle: 最小パーセンタイル期間用DCの最小周期
            min_percentile_dc_max_output: 最小パーセンタイル期間用DCの最大出力
            min_percentile_dc_min_output: 最小パーセンタイル期間用DCの最小出力
            zatr_max_dc_cycle_part: ZATR最大DCの周期部分
            zatr_max_dc_max_cycle: ZATR最大DCの最大周期
            zatr_max_dc_min_cycle: ZATR最大DCの最小周期
            zatr_max_dc_max_output: ZATR最大DCの最大出力
            zatr_max_dc_min_output: ZATR最大DCの最小出力
            zatr_min_dc_cycle_part: ZATR最小DCの周期部分
            zatr_min_dc_max_cycle: ZATR最小DCの最大周期
            zatr_min_dc_min_cycle: ZATR最小DCの最小周期
            zatr_min_dc_max_output: ZATR最小DCの最大出力
            zatr_min_dc_min_output: ZATR最小DCの最小出力
            max_percentile_cycle_mult: 最大パーセンタイル期間の周期乗数
            min_percentile_cycle_mult: 最小パーセンタイル期間の周期乗数
            max_multiplier: ATR乗数の最大値（レガシーパラメータ）
            min_multiplier: ATR乗数の最小値（レガシーパラメータ）
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            smoother_type: スムーザーの種類
            src_type: ソースタイプ
        """
        super().__init__("SimpleZTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
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
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        
        # ZTrendDirectionSignalの初期化（ZTrendのパラメータをそのまま渡す）
        self.direction_signal = ZTrendDirectionSignal(
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
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type
        )
        
        # シグナルキャッシュ
        self._signals = None
        self._data_len = 0
    
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
        シグナルを計算する
        
        Args:
            data: 価格データ
        """
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # ZTrendDirectionSignalを生成
                direction_values = self.direction_signal.generate(data)
                
                # エントリーシグナルを計算（ZTrendDirectionSignalと同じ）
                self._signals = calculate_entry_signals(direction_values)
                self._data_len = current_len
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"シグナル計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はゼロシグナルを設定
            self._signals = np.zeros(len(data), dtype=np.int8)
            self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル (1=ロング、-1=ショート、0=ニュートラル)
        """
        self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1=ロング、-1=ショート)
            index: チェックするインデックス（デフォルト: -1=最新値）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if position == 0:
            return False
            
        self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        if index < 0 or index >= len(self._signals):
            return False
        
        # ロングポジションは下降トレンド(-1)でエグジット
        if position == 1 and self._signals[index] == -1:
            return True
            
        # ショートポジションは上昇トレンド(1)でエグジット
        if position == -1 and self._signals[index] == 1:
            return True
            
        return False
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZTrendの上バンド値を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: 上バンド値の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_upper_band()
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZTrendの下バンド値を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: 下バンド値の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_lower_band()
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: サイクル効率比の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_cycle_er()
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: 動的乗数の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_dynamic_multiplier()
    
    def get_dynamic_percentile_length(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的パーセンタイル期間を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: 動的パーセンタイル期間の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_dynamic_percentile_length()
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATR値を取得
        
        Args:
            data: 価格データ（Noneの場合は既存データを使用）
            
        Returns:
            np.ndarray: ZATR値の配列
        """
        if data is not None:
            self.calculate_signals(data)
            
        return self.direction_signal.get_z_atr() 