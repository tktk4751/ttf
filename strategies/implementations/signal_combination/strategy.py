from typing import List

import numpy as np

from signals.base_signal import BaseSignal
from strategies.base import BaseStrategy


class SignalCombinationStrategy(BaseStrategy):
    """複数のシグナルを組み合わせた戦略クラス"""
    
    def __init__(self, signals: List[BaseSignal], name: str = "SignalCombination"):
        """
        Args:
            signals (List[BaseSignal]): 組み合わせるシグナルのリスト
            name (str, optional): 戦略名
        """
        super().__init__(name)
        self.signals = signals
    
    def generate_entry(self, data: np.ndarray) -> np.ndarray:
        """エントリーシグナルを生成
        
        全てのシグナルが同じ方向を示している場合にのみエントリーシグナルを生成します。
        
        Args:
            data (np.ndarray): 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1: ロング、-1: ショート、0: シグナルなし）
        """
        if not self.signals:
            return np.zeros(len(data))
        
        # 各シグナルの結果を取得
        signal_results = []
        for signal in self.signals:
            signal_results.append(signal.generate(data))
        
        # シグナルの合意を取る
        combined_signal = np.zeros(len(data))
        for i in range(len(data)):
            current_signals = [result[i] for result in signal_results]
            
            # 全てのシグナルがロングを示している場合
            if all(s == 1 for s in current_signals):
                combined_signal[i] = 1
            # 全てのシグナルがショートを示している場合
            elif all(s == -1 for s in current_signals):
                combined_signal[i] = -1
        
        return combined_signal
    
    def generate_exit(self, data: np.ndarray, position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成
        
        いずれかのシグナルが現在のポジションと反対の方向を示した場合にエグジットします。
        
        Args:
            data (np.ndarray): 価格データ
            position (int): 現在のポジション（1: ロング、-1: ショート）
            index (int, optional): チェックする時点のインデックス
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if not self.signals:
            return False
        
        # 各シグナルの現在の値を確認
        for signal in self.signals:
            current_signal = signal.generate(data)[index]
            
            # ロングポジションで、ショートシグナルが出た場合
            if position == 1 and current_signal == -1:
                return True
            # ショートポジションで、ロングシグナルが出た場合
            elif position == -1 and current_signal == 1:
                return True
        
        return False
    
    @staticmethod
    def convert_params_to_strategy_format(params: dict) -> dict:
        """パラメータを戦略フォーマットに変換
        
        Args:
            params (dict): 最適化パラメータ
            
        Returns:
            dict: 戦略フォーマットのパラメータ
        """
        return params  # このクラスでは特別な変換は不要 