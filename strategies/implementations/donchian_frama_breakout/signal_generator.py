#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numba import njit

from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX
from signals.implementations.donchian_frama_breakout.entry import DonchianFRAMABreakoutEntrySignal


class FilterType(Enum):
    """ドンチャンFRAMAブレイクアウトストラテジー用のフィルタータイプ"""
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


class DonchianFRAMABreakoutSignalGenerator:
    """ドンチャンFRAMAブレイクアウトシグナルジェネレーター"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
        # フィルタータイプ
        filter_type_str = params.get('filter_type', 'none')
        self.filter_type = FilterType(filter_type_str)
        
        # エントリーシグナル
        entry_params = params.get('entry', {})
        
        # HyperER動的適応パラメータの取得
        enable_hyper_er_adaptation = entry_params.get('enable_hyper_er_adaptation', False)
        hyper_er_period = entry_params.get('hyper_er_period', 14)
        hyper_er_midline_period = entry_params.get('hyper_er_midline_period', 100)
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min = entry_params.get('frama_fc_min', 1.0)
        frama_fc_max = entry_params.get('frama_fc_max', 13.0)
        frama_sc_min = entry_params.get('frama_sc_min', 60.0)
        frama_sc_max = entry_params.get('frama_sc_max', 250.0)
        
        # ドンチャンミッドライン HyperER動的適応パラメータ
        donchian_midline_period_min = entry_params.get('donchian_midline_period_min', 55.0)
        donchian_midline_period_max = entry_params.get('donchian_midline_period_max', 250.0)
        
        # トリガー用ドンチャン HyperER動的適応パラメータ
        trigger_donchian_period_min = entry_params.get('trigger_donchian_period_min', 20.0)
        trigger_donchian_period_max = entry_params.get('trigger_donchian_period_max', 100.0)
        
        self.entry_signal = DonchianFRAMABreakoutEntrySignal(
            # ドンチャンミッドラインパラメータ（フィルター用）
            donchian_midline_period=entry_params.get('donchian_midline_period', 200),
            donchian_midline_src_type=entry_params.get('donchian_midline_src_type', 'hlc3'),
            
            # FRAMAパラメータ（トレンド方向判定用）
            frama_period=entry_params.get('frama_period', 16),
            frama_src_type=entry_params.get('frama_src_type', 'hlc3'),
            frama_fc=entry_params.get('frama_fc', 2),
            frama_sc=entry_params.get('frama_sc', 198),
            frama_period_mode=entry_params.get('frama_period_mode', 'fixed'),
            
            # トリガー用ドンチャンチャネルパラメータ
            trigger_donchian_period=entry_params.get('trigger_donchian_period', 60),
            trigger_donchian_src_type=entry_params.get('trigger_donchian_src_type', 'hlc3'),
            
            # HyperER動的適応パラメータ
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            
            # FRAMA HyperER動的適応パラメータ
            frama_fc_min=frama_fc_min,
            frama_fc_max=frama_fc_max,
            frama_sc_min=frama_sc_min,
            frama_sc_max=frama_sc_max,
            
            # ドンチャンミッドライン HyperER動的適応パラメータ
            donchian_midline_period_min=donchian_midline_period_min,
            donchian_midline_period_max=donchian_midline_period_max,
            
            # トリガー用ドンチャン HyperER動的適応パラメータ
            trigger_donchian_period_min=trigger_donchian_period_min,
            trigger_donchian_period_max=trigger_donchian_period_max,
            
            signal_mode=entry_params.get('signal_mode', 'position')
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
                midline_period=hyper_er_params.get('midline_period', 100),
                er_src_type=hyper_er_params.get('src_type', 'close'),
                use_roofing_filter=hyper_er_params.get('use_roofing_filter', False),
                roofing_hp_cutoff=hyper_er_params.get('roofing_hp_cutoff', 48.0),
                roofing_ss_band_edge=hyper_er_params.get('roofing_ss_band_edge', 10.0),
                use_laguerre_filter=hyper_er_params.get('use_laguerre_filter', False),
                laguerre_gamma=hyper_er_params.get('laguerre_gamma', 0.5),
                smoother_type=hyper_er_params.get('smoother_type', 'super_smoother'),
                smoother_period=hyper_er_params.get('smoother_period', 10),
                detector_type=hyper_er_params.get('detector_type', 'hody'),
                lp_period=hyper_er_params.get('lp_period', 10),
                hp_period=hyper_er_params.get('hp_period', 60),
                cycle_part=hyper_er_params.get('cycle_part', 0.5),
                max_cycle=hyper_er_params.get('max_cycle', 60),
                min_cycle=hyper_er_params.get('min_cycle', 10),
                max_output=hyper_er_params.get('max_output', 50),
                min_output=hyper_er_params.get('min_output', 5)
            )
            
            trend_index_params = params.get('hyper_trend_index', {})
            self.hyper_trend_index = HyperTrendIndex(
                period=trend_index_params.get('period', 14),
                midline_period=trend_index_params.get('midline_period', 100),
                src_type=trend_index_params.get('src_type', 'close'),
                use_kalman_filter=trend_index_params.get('use_kalman_filter', False),
                kalman_filter_type=trend_index_params.get('kalman_filter_type', 'simple'),
                use_dynamic_period=trend_index_params.get('use_dynamic_period', False),
                detector_type=trend_index_params.get('detector_type', 'dft_dominant'),
                use_roofing_filter=trend_index_params.get('use_roofing_filter', False),
                roofing_hp_cutoff=trend_index_params.get('roofing_hp_cutoff', 48.0),
                roofing_ss_band_edge=trend_index_params.get('roofing_ss_band_edge', 10.0)
            )
            
            hyper_adx_params = params.get('hyper_adx', {})
            self.hyper_adx = HyperADX(
                period=hyper_adx_params.get('period', 14),
                midline_period=hyper_adx_params.get('midline_period', 100),
                use_kalman_filter=hyper_adx_params.get('use_kalman_filter', False),
                kalman_filter_type=hyper_adx_params.get('kalman_filter_type', 'simple'),
                use_dynamic_period=hyper_adx_params.get('use_dynamic_period', False),
                detector_type=hyper_adx_params.get('detector_type', 'dft_dominant'),
                use_roofing_filter=hyper_adx_params.get('use_roofing_filter', False),
                roofing_hp_cutoff=hyper_adx_params.get('roofing_hp_cutoff', 48.0),
                roofing_ss_band_edge=hyper_adx_params.get('roofing_ss_band_edge', 10.0)
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
            # 2つ目のドンチャンチャネルを使用したエグジット
            exit_signals = self.entry_signal.generate_exit_signals(data, position)
            if index == -1:
                index = len(exit_signals) - 1
            if index < 0 or index >= len(exit_signals):
                return False
            
            return bool(exit_signals[index] == 1)
        else:
            # 全データのエグジットシグナル（ロングとショート両方）
            exit_signals_long = self.entry_signal.generate_exit_signals(data, position=1)
            exit_signals_short = self.entry_signal.generate_exit_signals(data, position=-1)
            
            # 統合エグジットシグナル（どちらかのポジションのエグジットがあればエグジット）
            combined_exits = np.logical_or(exit_signals_long, exit_signals_short).astype(np.int8)
            
            return combined_exits
    
    def get_signal_info(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """シグナル詳細情報を取得（デバッグ用）"""
        entry_base = self.entry_signal.generate(data)
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
        
        # インジケーター値も追加
        info['donchian_midline_values'] = self.entry_signal.get_donchian_midline_values(data)
        info['frama_values'] = self.entry_signal.get_frama_values(data)
        
        trigger_bands = self.entry_signal.get_trigger_donchian_bands(data)
        if trigger_bands:
            info['trigger_donchian_upper'] = trigger_bands[0]
            info['trigger_donchian_lower'] = trigger_bands[1]
        
        return info
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        if hasattr(self.entry_signal, 'reset'):
            self.entry_signal.reset()
        
        if self.hyper_er and hasattr(self.hyper_er, 'reset'):
            self.hyper_er.reset()
        if self.hyper_trend_index and hasattr(self.hyper_trend_index, 'reset'):
            self.hyper_trend_index.reset()
        if self.hyper_adx and hasattr(self.hyper_adx, 'reset'):
            self.hyper_adx.reset()


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ドンチャンFRAMAブレイクアウトシグナルジェネレーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 300
    base_price = 100.0
    
    prices = [base_price]
    for i in range(1, length):
        if i < 100:
            change = 0.002 + np.random.normal(0, 0.008)
        elif i < 200:
            change = np.random.normal(0, 0.010)
        else:
            change = -0.002 + np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    # シグナルジェネレーターのテスト
    signal_params = {
        'filter_type': 'none',
        'entry': {
            'donchian_midline_period': 200,
            'frama_period': 16,
            'frama_fc': 2,
            'frama_sc': 198,
            'trigger_donchian_period': 60,
            'signal_mode': 'position',
            'enable_hyper_er_adaptation': False
        }
    }
    
    signal_generator = DonchianFRAMABreakoutSignalGenerator(signal_params)
    
    # エントリーシグナルテスト
    entry_signals = signal_generator.generate_entry_signals(df)
    long_entries = np.sum(entry_signals == 1)
    short_entries = np.sum(entry_signals == -1)
    neutral = np.sum(entry_signals == 0)
    
    print(f"エントリーシグナル:")
    print(f"  ロング: {long_entries} ({long_entries/len(df)*100:.1f}%)")
    print(f"  ショート: {short_entries} ({short_entries/len(df)*100:.1f}%)")
    print(f"  中立: {neutral} ({neutral/len(df)*100:.1f}%)")
    
    # エグジットシグナルテスト
    exit_signals = signal_generator.generate_exit_signals(df)
    total_exits = np.sum(exit_signals == 1)
    
    print(f"\nエグジットシグナル:")
    print(f"  エグジット: {total_exits} ({total_exits/len(df)*100:.1f}%)")
    
    print("\n=== テスト完了 ===")