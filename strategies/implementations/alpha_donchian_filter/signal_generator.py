#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_donchian.entry import AlphaDonchianBreakoutEntrySignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal


@jit(nopython=True)
def calculate_entry_signals(donchian_signals: np.ndarray, filter_signals: np.ndarray) -> np.ndarray:
    """
    エントリーシグナルを一度に計算（高速化版）
    
    Args:
        donchian_signals: アルファドンチャンのブレイクアウトシグナル配列
        filter_signals: アルファフィルターシグナル配列
    
    Returns:
        エントリーシグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    signals = np.zeros_like(donchian_signals, dtype=np.int8)
    
    # ロングエントリー: アルファドンチャンの買いシグナル + アルファフィルターがトレンド相場
    long_condition = (donchian_signals == 1) & (filter_signals == 1)
    
    # ショートエントリー: アルファドンチャンの売りシグナル + アルファフィルターがトレンド相場
    short_condition = (donchian_signals == -1) & (filter_signals == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaDonchianFilterSignalGenerator(BaseSignalGenerator):
    """
    アルファドンチャン+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファドンチャンのブレイクアウトで買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファドンチャンのブレイクアウトで売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファドンチャンの売りシグナル
    - ショート: アルファドンチャンの買いシグナル
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ調整で市場環境に適応
    - アルファフィルターによる市場状態フィルタリングでトレンド相場に集中
    - 高性能なNumba JITコンパイルによる計算の高速化
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # アルファドンチャン用パラメータ
        max_donchian_period: int = 55,
        min_donchian_period: int = 13,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_donchian_period: ドンチャン期間の最大値（デフォルト: 55）
            min_donchian_period: ドンチャン期間の最小値（デフォルト: 13）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaDonchianFilterSignalGenerator")
        
        # パラメータの保存
        self.params = {
            'er_period': er_period,
            'max_donchian_period': max_donchian_period,
            'min_donchian_period': min_donchian_period,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold
        }
        
        # アルファドンチャンブレイクアウトシグナルの初期化
        self.donchian_signal = AlphaDonchianBreakoutEntrySignal(
            er_period=er_period,
            max_donchian_period=max_donchian_period,
            min_donchian_period=min_donchian_period
        )
        
        # アルファフィルターシグナルの初期化
        self.filter_signal = AlphaFilterSignal(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            solid={
                'threshold': filter_threshold
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._last_close_value = None
        self._signals = None
        self._donchian_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        シグナル計算（高速化版）
        
        Args:
            data: 価格データ
        """
        try:
            # 最適化されたキャッシュ機構 - データ長とデータの最終値で判断
            current_len = len(data)
            
            # 終値の取得（キャッシュチェック用）
            if isinstance(data, pd.DataFrame):
                close_value = data['close'].iloc[-1] if not data.empty else None
            else:
                close_value = data[-1, 3] if len(data) > 0 else None
            
            # キャッシュチェック - 同じデータなら再計算しない
            if (self._signals is not None and current_len == self._data_len and 
                close_value == self._last_close_value):
                return
                
            # データフレームの作成（必要な列のみ）
            if isinstance(data, pd.DataFrame):
                df = data[['open', 'high', 'low', 'close']]
            else:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            
            # 各シグナルの計算
            donchian_signals = self.donchian_signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(donchian_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._donchian_signals = donchian_signals
            
            # キャッシュ更新
            self._data_len = current_len
            self._last_close_value = close_value
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"シグナル計算中にエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._donchian_signals = np.zeros(len(data), dtype=np.int8)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する（BaseSignalの要件を満たすため）
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        return self.get_entry_signals(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得（高速化版）
        
        Args:
            data: 価格データ
        
        Returns:
            np.ndarray: エントリーシグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナル生成（高速化版）
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート, 0: なし)
            index: 評価するインデックス（デフォルト: -1＝最新）
        
        Returns:
            bool: エグジットすべきかどうか
        """
        self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        if position == 1:  # ロングポジション
            # アルファドンチャンの売りシグナルでロングをエグジット
            return bool(self._donchian_signals[index] == -1)
        elif position == -1:  # ショートポジション
            # アルファドンチャンの買いシグナルでショートをエグジット
            return bool(self._donchian_signals[index] == 1)
        return False
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルファドンチャンのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 上部バンド、下部バンド、中央バンド
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.donchian_signal.get_alpha_donchian().get_bands()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"バンド取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.donchian_signal.get_efficiency_ratio()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファフィルターの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.filter_signal.get_filter_values()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
        
    def get_filter_components(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        アルファフィルターのコンポーネント値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: コンポーネント値のディクショナリ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.filter_signal.get_component_values()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"フィルターコンポーネント取得中にエラー: {str(e)}")
            # エラー時は空のディクショナリを返す
            return {'er': np.array([]), 'alpha_chop': np.array([]), 'alpha_adx': np.array([]), 'dynamic_period': np.array([])}