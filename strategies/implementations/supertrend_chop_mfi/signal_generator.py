from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from functools import lru_cache

from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.mfi.exit import MFIExitSignal


class SupertrendChopMfiSignalGenerator:
    """スーパートレンド、チョピネス、MFIのシグナル生成器"""
    
    def __init__(
        self,
        supertrend_params: Dict[str, Any],
        chop_params: Dict[str, Any],
        mfi_params: Dict[str, Any]
    ):
        """
        コンストラクタ
        
        Args:
            supertrend_params: スーパートレンドのパラメータ
            chop_params: チョピネスインデックスのパラメータ
            mfi_params: MFIのパラメータ
        """
        # シグナルクラスの初期化
        self.supertrend = SupertrendDirectionSignal(**supertrend_params)
        self.chop = ChopFilterSignal(**chop_params)
        self.mfi_exit = MFIExitSignal(**mfi_params)
        
        # パラメータの保存
        self._supertrend_params = supertrend_params
        self._chop_params = chop_params
        self._mfi_params = mfi_params
        
        # シグナルのキャッシュ
        self._cached_data_len = None
        self._cached_supertrend_signals = None
        self._cached_chop_signals = None
        self._cached_mfi_signals = None
    
    def _calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナルを計算してキャッシュに保存"""
        data_len = len(data)
        
        # データが変更されている場合のみ再計算
        if self._cached_data_len != data_len:
            self._cached_supertrend_signals = self.supertrend.generate(data)
            self._cached_chop_signals = self.chop.generate(data)
            self._cached_mfi_signals = self.mfi_exit.generate(data)
            self._cached_data_len = data_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成
        
        スーパートレンドディレクションシグナルが買いの状態の時に、
        チョピネスインデックスが50以下になったら買い。
        スーパートレンドディレクションシグナルが売りの状態のときに、
        チョピネスインデックスが50以下になったら売り。
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1: ロング、-1: ショート、0: シグナルなし）
        """
        # シグナルの計算（キャッシュを利用）
        self._calculate_signals(data)
        
        # シグナルの生成
        signals = np.zeros(len(data))
        
        # チョピネスが閾値以上の場合のみスーパートレンドのシグナルを採用
        signals = np.where(self._cached_chop_signals == 1, self._cached_supertrend_signals, signals)
        
        return signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成
        
        1. スーパートレンドディレクションシグナルが買いから売りに切り替わったら買いポジションを決済。
        2. スーパートレンドディレクションシグナルが売りから買いに切り替わったら売りポジションを決済。
        3. MFIシグナルでもエグジット。
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックする時点のインデックス
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if index < 0:
            index = len(data) + index
        
        if index <= 0:
            return False
        
        # シグナルの計算（キャッシュを利用）
        self._calculate_signals(data)
        
        # 現在のシグナル値を取得
        current_supertrend = self._cached_supertrend_signals[index]
        current_mfi = self._cached_mfi_signals[index]
        
        # スーパートレンドの方向転換によるエグジット
        supertrend_exit = False
        if index > 0:
            prev_supertrend = self._cached_supertrend_signals[index-1]
            supertrend_exit = (
                (position == 1 and current_supertrend == -1 and prev_supertrend != -1) or  # ロングポジションで売りシグナルに転換
                (position == -1 and current_supertrend == 1 and prev_supertrend != 1)      # ショートポジションで買いシグナルに転換
            )
        
        # MFIによるエグジット
        mfi_exit = False
        if index > 0:
            prev_mfi = self._cached_mfi_signals[index-1]
            mfi_exit = (
                (position == 1 and current_mfi == 1 and prev_mfi != 1) or    # ロングポジションでMFIエグジット
                (position == -1 and current_mfi == -1 and prev_mfi != -1)    # ショートポジションでMFIエグジット
            )
        
        return supertrend_exit or mfi_exit 