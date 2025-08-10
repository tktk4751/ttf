from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numba import njit

from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX
from signals.implementations.donchian_frama.entry import DonchianFRAMACrossoverEntrySignal


class FilterType(Enum):
    """ドンチャンFRAMAストラテジー用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit
def combine_signals_numba(
    entry_signals: np.ndarray,
    filter_signals: np.ndarray,
    use_filter: bool
) -> np.ndarray:
    """エントリーシグナルとフィルターシグナルを結合"""
    n = len(entry_signals)
    result = np.zeros(n)
    
    for i in range(n):
        if use_filter:
            # フィルターが1（トレンド）の時のみエントリーシグナルを有効化
            if filter_signals[i] == 1.0:
                result[i] = entry_signals[i]
        else:
            # フィルター無しの場合はエントリーシグナルをそのまま使用
            result[i] = entry_signals[i]
    
    return result


@njit
def consensus_filter_numba(
    hyper_er_signals: np.ndarray,
    trend_index_signals: np.ndarray,
    hyper_adx_signals: np.ndarray
) -> np.ndarray:
    """3つの指標のうち2つ以上が1の場合に1を出力"""
    n = len(hyper_er_signals)
    result = np.zeros(n)
    
    for i in range(n):
        count = 0
        if hyper_er_signals[i] == 1.0:
            count += 1
        if trend_index_signals[i] == 1.0:
            count += 1
        if hyper_adx_signals[i] == 1.0:
            count += 1
        
        # 2つ以上が1の場合に1を出力、それ以外は-1
        if count >= 2:
            result[i] = 1.0
        else:
            result[i] = -1.0
    
    return result


class DonchianFRAMASignalGenerator:
    """ドンチャンFRAMAシグナルジェネレーター"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
        # フィルタータイプ
        filter_type_str = params.get('filter_type', 'none')
        self.filter_type = FilterType(filter_type_str)
        
        # エントリーシグナル
        entry_params = params.get('entry', {})
        self.entry_signal = DonchianFRAMACrossoverEntrySignal(
            donchian_period=entry_params.get('donchian_period', 89),
            frama_period=entry_params.get('frama_period', 16),
            frama_fc=entry_params.get('frama_fc', 2),
            frama_sc=entry_params.get('frama_sc', 198),
            position_mode=entry_params.get('signal_mode', 'position') == 'position'
        )
        
        # フィルターインジケーター（必要に応じて初期化）
        self.hyper_er = None
        self.hyper_trend_index = None
        self.hyper_adx = None
        
        if self.filter_type != FilterType.NONE:
            # フィルター用インジケーターを初期化
            hyper_er_params = params.get('hyper_er', {})
            self.hyper_er = HyperER(
                period=hyper_er_params.get('period', 14),
                midline_period=hyper_er_params.get('midline_period', 100)
            )
            
            trend_index_params = params.get('hyper_trend_index', {})
            self.hyper_trend_index = HyperTrendIndex(
                period=trend_index_params.get('period', 14),
                midline_period=trend_index_params.get('midline_period', 100)
            )
            
            hyper_adx_params = params.get('hyper_adx', {})
            self.hyper_adx = HyperADX(
                period=hyper_adx_params.get('period', 14),
                midline_period=hyper_adx_params.get('midline_period', 100)
            )
    
    def generate_entry_signals(self, data: pd.DataFrame) -> np.ndarray:
        """エントリーシグナルを生成"""
        # 基本エントリーシグナル
        entry_signals = self.entry_signal.generate(data)
        
        # フィルター適用
        if self.filter_type == FilterType.NONE:
            return entry_signals
        
        # フィルターシグナル取得
        filter_signals = self._get_filter_signals(data)
        
        # シグナル結合
        return combine_signals_numba(entry_signals, filter_signals, True)
    
    def _get_filter_signals(self, data: pd.DataFrame) -> np.ndarray:
        """フィルターシグナルを取得"""
        if self.filter_type == FilterType.HYPER_ER:
            hyper_er_result = self.hyper_er.calculate(data)
            return self.hyper_er.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_TREND_INDEX:
            trend_index_result = self.hyper_trend_index.calculate(data)
            return self.hyper_trend_index.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_ADX:
            hyper_adx_result = self.hyper_adx.calculate(data)
            return self.hyper_adx.get_trend_signal()
            
        elif self.filter_type == FilterType.CONSENSUS:
            # 統合フィルター（3つのうち2つが1なら1）
            hyper_er_result = self.hyper_er.calculate(data)
            hyper_er_signals = self.hyper_er.get_trend_signal()
            
            trend_index_result = self.hyper_trend_index.calculate(data)
            trend_index_signals = self.hyper_trend_index.get_trend_signal()
            
            hyper_adx_result = self.hyper_adx.calculate(data)
            hyper_adx_signals = self.hyper_adx.get_trend_signal()
            
            return consensus_filter_numba(
                hyper_er_signals,
                trend_index_signals, 
                hyper_adx_signals
            )
        
        # デフォルト（フィルターなし）
        return np.ones(len(data))
    
    def generate_exit_signals(self, data: pd.DataFrame, position: int = None, index: int = -1) -> Union[np.ndarray, bool]:
        """エグジットシグナルを生成"""
        if position is not None:
            # 特定のポジションとインデックスでのエグジット判定
            entry_signals = self.generate_entry_signals(data)
            if index == -1:
                index = len(entry_signals) - 1
            if index < 0 or index >= len(entry_signals):
                return False
            
            current_signal = entry_signals[index]
            # ロングポジション: 現在のシグナルが-1（ショート）ならエグジット
            if position == 1 and current_signal == -1:
                return True
            # ショートポジション: 現在のシグナルが1（ロング）ならエグジット
            elif position == -1 and current_signal == 1:
                return True
            return False
        else:
            # 全データのエグジットシグナル（基本的にはエントリーの逆）
            entry_signals = self.generate_entry_signals(data)
            return -entry_signals
    
    def get_signal_info(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """シグナル詳細情報を取得（デバッグ用）"""
        entry_base = self.entry_signal.calculate(data)
        entry_filtered = self.generate_entry_signals(data)
        
        info = {
            'entry_base': entry_base,
            'entry_filtered': entry_filtered,
            'filter_type': str(self.filter_type.value)
        }
        
        # フィルターシグナル詳細
        if self.filter_type != FilterType.NONE:
            filter_signals = self._get_filter_signals(data)
            info['filter_signals'] = filter_signals
            
            if self.filter_type == FilterType.CONSENSUS:
                # 各インジケーターの個別シグナルも追加
                hyper_er_result = self.hyper_er.calculate(data)
                info['hyper_er_signals'] = self.hyper_er.get_trend_signal()
                
                trend_index_result = self.hyper_trend_index.calculate(data)
                info['trend_index_signals'] = self.hyper_trend_index.get_trend_signal()
                
                hyper_adx_result = self.hyper_adx.calculate(data)
                info['hyper_adx_signals'] = self.hyper_adx.get_trend_signal()
        
        return info