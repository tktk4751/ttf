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
from signals.implementations.hyper_donchian.entry import HyperDonchianBreakoutEntrySignal


class FilterType(Enum):
    """Hyperドンチャンブレイクアウトストラテジー用のフィルタータイプ"""
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


class HyperDonchianSignalGenerator:
    """Hyperドンチャンブレイクアウトシグナルジェネレーター"""
    
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
        
        # Hyperドンチャン HyperER動的適応パラメータ
        hyper_donchian_period_min = entry_params.get('hyper_donchian_period_min', 15.0)
        hyper_donchian_period_max = entry_params.get('hyper_donchian_period_max', 60.0)
        
        self.entry_signal = HyperDonchianBreakoutEntrySignal(
            period=entry_params.get('hyper_donchian_period', 20),
            src_type=entry_params.get('hyper_donchian_src_type', 'close'),
            # HyperER動的適応パラメータを追加
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            period_min=hyper_donchian_period_min,
            period_max=hyper_donchian_period_max
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
        info['hyper_donchian_values'] = self.entry_signal.get_hyper_donchian_values(data)
        
        hyper_donchian_bands = self.entry_signal.get_hyper_donchian_bands(data)
        if hyper_donchian_bands:
            info['hyper_donchian_upper'] = hyper_donchian_bands[0]
            info['hyper_donchian_lower'] = hyper_donchian_bands[1]
        
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
    
    print("=== Hyperドンチャンブレイクアウトシグナルジェネレーターのテスト ===")
    
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
            'hyper_donchian_period': 20,
            'hyper_donchian_src_type': 'close',
            'enable_hyper_er_adaptation': False
        }
    }
    
    signal_generator = HyperDonchianSignalGenerator(signal_params)
    
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
    total_exits = np.sum(exit_signals == 1) + np.sum(exit_signals == -1)
    
    print(f"\\nエグジットシグナル:")
    print(f"  エグジット: {total_exits} ({total_exits/len(df)*100:.1f}%)")
    
    # コンセンサスフィルターのテスト
    print("\\nコンセンサスフィルターのテスト...")
    consensus_params = signal_params.copy()
    consensus_params['filter_type'] = 'consensus'
    consensus_params['hyper_er'] = {'period': 14, 'midline_period': 100}
    consensus_params['hyper_trend_index'] = {'period': 14, 'midline_period': 100}
    consensus_params['hyper_adx'] = {'period': 14, 'midline_period': 100}
    
    consensus_generator = HyperDonchianSignalGenerator(consensus_params)
    consensus_signals = consensus_generator.generate_entry_signals(df)
    
    consensus_long = np.sum(consensus_signals == 1)
    consensus_short = np.sum(consensus_signals == -1)
    consensus_neutral = np.sum(consensus_signals == 0)
    
    print(f"  コンセンサスロング: {consensus_long} ({consensus_long/len(df)*100:.1f}%)")
    print(f"  コンセンサスショート: {consensus_short} ({consensus_short/len(df)*100:.1f}%)")
    print(f"  コンセンサス中立: {consensus_neutral} ({consensus_neutral/len(df)*100:.1f}%)")
    
    # 従来版との比較
    print("\\n従来のドンチャンブレイクアウトとの比較...")
    try:
        from ..donchian.signal_generator import DonchianSignalGenerator
        
        traditional_params = {
            'filter_type': 'none',
            'entry': {
                'donchian_period': 20,
                'enable_hyper_er_adaptation': False
            }
        }
        
        traditional_generator = DonchianSignalGenerator(traditional_params)
        traditional_signals = traditional_generator.generate_entry_signals(df)
        
        trad_long = np.sum(traditional_signals == 1)
        trad_short = np.sum(traditional_signals == -1)
        trad_neutral = np.sum(traditional_signals == 0)
        
        print(f"  従来版ロング: {trad_long} ({trad_long/len(df)*100:.1f}%)")
        print(f"  従来版ショート: {trad_short} ({trad_short/len(df)*100:.1f}%)")
        print(f"  従来版中立: {trad_neutral} ({trad_neutral/len(df)*100:.1f}%)")
        
        # シグナル安定性比較
        hyper_changes = np.sum(np.diff(entry_signals) != 0)
        trad_changes = np.sum(np.diff(traditional_signals) != 0)
        
        print(f"\\nシグナル安定性比較:")
        print(f"  Hyperドンチャン版変化回数: {hyper_changes}")
        print(f"  従来版変化回数: {trad_changes}")
        if trad_changes > 0:
            print(f"  安定性改善: {((trad_changes - hyper_changes) / trad_changes * 100):.1f}%")
        
    except ImportError:
        print("  従来のドンチャンブレイクアウトシグナルジェネレーターが見つかりませんでした")
    
    print("\\n=== テスト完了 ===")