#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_frama.entry import HyperFRAMAPositionEntrySignal, HyperFRAMACrossoverEntrySignal
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class FilterType(Enum):
    """HyperFRAMAストラテジー用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True, parallel=False)
def combine_signals_numba(hyper_frama_signals: np.ndarray, filter_signals: np.ndarray, filter_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperFRAMAシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        hyper_frama_signals: HyperFRAMAシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=HyperER, 2=HyperTrendIndex, 3=HyperADX, 4=Consensus）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ロングシグナル, ショートシグナル)
    """
    length = len(hyper_frama_signals)
    long_signals = np.zeros(length, dtype=np.int8)
    short_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        hyper_frama_signal = hyper_frama_signals[i]
        
        # フィルターなしの場合
        if filter_type == 0:
            if hyper_frama_signal == 1:
                long_signals[i] = 1
            elif hyper_frama_signal == -1:
                short_signals[i] = 1
        else:
            # フィルターありの場合
            if i < len(filter_signals):
                filter_signal = filter_signals[i]
                
                # ロングエントリー条件: HyperFRAMAシグナル=1 かつ フィルターシグナル=1
                if hyper_frama_signal == 1 and filter_signal == 1:
                    long_signals[i] = 1
                
                # ショートエントリー条件: HyperFRAMAシグナル=-1 かつ フィルターシグナル=1 (フィルターが許可)
                elif hyper_frama_signal == -1 and filter_signal == 1:
                    short_signals[i] = 1
    
    return long_signals, short_signals


@njit(fastmath=True)
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


@njit(fastmath=True, parallel=False)
def check_exit_conditions_numba(hyper_frama_signals: np.ndarray, position: int, index: int) -> bool:
    """
    エグジット条件をチェックする（Numba最適化版）
    
    Args:
        hyper_frama_signals: HyperFRAMAシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(hyper_frama_signals):
        return False
    
    hyper_frama_signal = hyper_frama_signals[index]
    
    # ロングポジション: HyperFRAMAシグナル=-1でエグジット
    if position == 1 and hyper_frama_signal == -1:
        return True
    
    # ショートポジション: HyperFRAMAシグナル=1でエグジット
    if position == -1 and hyper_frama_signal == 1:
        return True
    
    return False


class HyperFRAMAEnhancedSignalGenerator(BaseSignalGenerator):
    """
    HyperFRAMA Enhanced シグナル生成クラス
    
    特徴:
    - HyperFRAMAシグナルをベースとした高度なエントリー・エグジット制御
    - フラクタル次元に基づく適応型移動平均線
    - アルファ調整係数による柔軟な設定
    - 4つのフィルターから選択可能（HyperER, HyperTrendIndex, HyperADX, Consensus）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: HyperFRAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: HyperFRAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0の場合はスルー
    
    エグジット条件:
    - ロング: HyperFRAMAシグナル=-1
    - ショート: HyperFRAMAシグナル=1
    """
    
    def __init__(
        self,
        # HyperFRAMAパラメータ
        period: int = 16,
        src_type: str = 'hl2',
        fc: int = 1,
        sc: int = 198,
        alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'hody_e',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 89,
        min_cycle: int = 8,
        max_output: int = 124,
        min_output: int = 8,
        # シグナル設定
        position_mode: bool = False,             # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ（2つ目）
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier: アルファ調整係数（デフォルト: 0.5）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            ... その他の動的期間パラメータ
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        
        super().__init__(f"HyperFRAMA_Enhanced_{signal_type}_{filter_name}")
        
        # パラメータの設定
        self._params = {
            # HyperFRAMAパラメータ
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'alpha_multiplier': alpha_multiplier,
            # 動的期間パラメータ
            'period_mode': period_mode,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            # シグナル設定
            'position_mode': position_mode,
            # フィルター設定
            'filter_type': filter_type,
            # HyperER フィルターパラメータ
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        self.filter_type = filter_type if isinstance(filter_type, FilterType) else FilterType(filter_type)
        self.position_mode = position_mode
        
        # HyperFRAMAエントリーシグナルの初期化
        if position_mode:
            self.hyper_frama_entry_signal = HyperFRAMAPositionEntrySignal(
                period=period,
                src_type=src_type,
                fc=fc,
                sc=sc,
                alpha_multiplier=alpha_multiplier,
                period_mode=period_mode,
                cycle_detector_type=cycle_detector_type,
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output
            )
        else:
            self.hyper_frama_entry_signal = HyperFRAMACrossoverEntrySignal(
                period=period,
                src_type=src_type,
                fc=fc,
                sc=sc,
                alpha_multiplier=alpha_multiplier,
                period_mode=period_mode,
                cycle_detector_type=cycle_detector_type,
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output
            )
        
        # フィルターインジケーター（必要に応じて初期化）
        self.hyper_er_filter = None
        self.hyper_trend_index_filter = None
        self.hyper_adx_filter = None
        
        if self.filter_type != FilterType.NONE:
            # フィルター用インジケーターを初期化
            self.hyper_er_filter = HyperER(
                period=filter_hyper_er_period,
                midline_period=filter_hyper_er_midline_period
            )
            
            self.hyper_trend_index_filter = HyperTrendIndex(
                period=filter_hyper_trend_index_period,
                midline_period=filter_hyper_trend_index_midline_period
            )
            
            self.hyper_adx_filter = HyperADX(
                period=filter_hyper_adx_period,
                midline_period=filter_hyper_adx_midline_period
            )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._hyper_frama_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._long_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                    if 'volume' in data.columns:
                        df = data[['open', 'high', 'low', 'close', 'volume']]
                else:
                    if data.shape[1] >= 5:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                    else:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # HyperFRAMAシグナルの計算
                    hyper_frama_signals = self.hyper_frama_entry_signal.generate(df)
                    self._hyper_frama_signals = hyper_frama_signals
                    
                    # フィルターシグナルの計算
                    self._filter_signals = self._get_filter_signals(df)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    self._long_signals, self._short_signals = combine_signals_numba(
                        hyper_frama_signals, self._filter_signals, filter_type_int
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._long_signals = np.zeros(current_len, dtype=np.int8)
                    self._short_signals = np.zeros(current_len, dtype=np.int8)
                    self._hyper_frama_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._long_signals = np.zeros(len(data), dtype=np.int8)
                self._short_signals = np.zeros(len(data), dtype=np.int8)
                self._hyper_frama_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def _get_filter_signals(self, data: pd.DataFrame) -> np.ndarray:
        """フィルターシグナルを取得"""
        if self.filter_type == FilterType.HYPER_ER:
            hyper_er_result = self.hyper_er_filter.calculate(data)
            return self.hyper_er_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_TREND_INDEX:
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            return self.hyper_trend_index_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_ADX:
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            return self.hyper_adx_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.CONSENSUS:
            # 統合フィルター（3つのうち2つが1なら1）
            hyper_er_result = self.hyper_er_filter.calculate(data)
            hyper_er_signals = self.hyper_er_filter.get_trend_signal()
            
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            trend_index_signals = self.hyper_trend_index_filter.get_trend_signal()
            
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            hyper_adx_signals = self.hyper_adx_filter.get_trend_signal()
            
            return consensus_filter_numba(
                hyper_er_signals,
                trend_index_signals, 
                hyper_adx_signals
            )
        
        # デフォルト（フィルターなし）
        return np.ones(len(data))

    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得
        
        Returns:
            統合されたエントリーシグナル（ロング=1、ショート=-1、なし=0）
        """
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        # ロング・ショートシグナルを統合
        combined_signals = self._long_signals - self._short_signals
        return combined_signals
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._long_signals
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        if self._short_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._short_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._hyper_frama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_exit_conditions_numba(self._hyper_frama_signals, position, index)
    
    def get_hyper_frama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperFRAMAシグナル取得"""
        if self._hyper_frama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._hyper_frama_signals
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        if self._filter_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._filter_signals
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """FRAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_frama_entry_signal.get_frama_values()
        except Exception as e:
            self.logger.error(f"FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_adjusted_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Adjusted FRAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_frama_entry_signal.get_adjusted_frama_values()
        except Exception as e:
            self.logger.error(f"Adjusted FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """フラクタル次元を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_frama_entry_signal.get_fractal_dimension()
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Alpha値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_frama_entry_signal.get_alpha_values()
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        フィルター詳細情報を取得
        
        Returns:
            フィルタータイプに応じた詳細データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self.filter_type == FilterType.NONE:
                return {}
            
            details = {}
            
            if self.filter_type == FilterType.HYPER_ER:
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_trend_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_TREND_INDEX:
                details.update({
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_ADX:
                details.update({
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.CONSENSUS:
                # 各インジケーターの個別シグナルも追加
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([]),
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([]),
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            
            return details
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            HyperFRAMAとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            metrics = {
                # HyperFRAMAメトリクス
                'frama_values': self.get_frama_values(),
                'adjusted_frama_values': self.get_adjusted_frama_values(),
                'fractal_dimension': self.get_fractal_dimension(),
                'alpha_values': self.get_alpha_values(),
                'hyper_frama_signals': self.get_hyper_frama_signals(data),
                # シグナルメトリクス
                'long_signals': self.get_long_signals(data),
                'short_signals': self.get_short_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # フィルター詳細
                'filter_type': self.filter_type.value
            }
            
            # フィルター固有のメトリクスを追加
            filter_details = self.get_filter_details()
            metrics.update(filter_details)
            
            return metrics
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._hyper_frama_signals = None
        self._filter_signals = None
        
        if hasattr(self.hyper_frama_entry_signal, 'reset'):
            self.hyper_frama_entry_signal.reset()
        if self.hyper_er_filter is not None and hasattr(self.hyper_er_filter, 'reset'):
            self.hyper_er_filter.reset()
        if self.hyper_trend_index_filter is not None and hasattr(self.hyper_trend_index_filter, 'reset'):
            self.hyper_trend_index_filter.reset()
        if self.hyper_adx_filter is not None and hasattr(self.hyper_adx_filter, 'reset'):
            self.hyper_adx_filter.reset()