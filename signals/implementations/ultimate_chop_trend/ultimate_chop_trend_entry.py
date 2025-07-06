#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.ultimate_chop_trend_v3 import UltimateChopTrendV3


@njit(fastmath=True, parallel=True)
def calculate_ultimate_chop_trend_signals(trend_direction: np.ndarray, 
                                        confidence_threshold: float = 0.3) -> np.ndarray:
    """
    Ultimate Chop Trend V3のシグナルを計算する（高速化版）
    
    Args:
        trend_direction: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        confidence_threshold: 信頼度しきい値
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(trend_direction)
    signals = np.zeros(length, dtype=np.int8)
    
    # トレンド方向変化検出（並列処理化）
    for i in prange(1, length):
        # NaN値チェック
        if np.isnan(trend_direction[i]) or np.isnan(trend_direction[i-1]):
            signals[i] = 0
            continue
        
        prev_trend = trend_direction[i-1]
        curr_trend = trend_direction[i]
        
        # トレンド方向変化によるシグナル生成
        # -1から1に切り替わったらロングエントリー
        if prev_trend == -1 and curr_trend == 1:
            signals[i] = 1
        # 1から-1に切り替わったらショートエントリー
        elif prev_trend == 1 and curr_trend == -1:
            signals[i] = -1
        else:
            signals[i] = 0
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_ultimate_chop_trend_exit_signals(trend_direction: np.ndarray, 
                                             current_position: np.ndarray = None,
                                             confidence_threshold: float = 0.3) -> np.ndarray:
    """
    Ultimate Chop Trend V3の決済シグナルを計算する（高速化版）
    
    Args:
        trend_direction: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        current_position: 現在のポジション（1=ロング、-1=ショート、0=ポジションなし）
        confidence_threshold: 信頼度しきい値
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(trend_direction)
    exit_signals = np.zeros(length, dtype=np.int8)
    
    if current_position is None:
        return exit_signals
    
    # 決済シグナル判定（並列処理化）
    for i in prange(1, length):
        # NaN値チェック
        if np.isnan(trend_direction[i]) or np.isnan(trend_direction[i-1]):
            continue
        
        prev_trend = trend_direction[i-1]
        curr_trend = trend_direction[i]
        
        # ロングポジション決済条件
        if current_position[i-1] == 1:
            # 1から-1または0に変化したらロング決済
            if prev_trend == 1 and (curr_trend == -1 or curr_trend == 0):
                exit_signals[i] = 1  # ロング決済
        
        # ショートポジション決済条件
        elif current_position[i-1] == -1:
            # -1から1または0に変化したらショート決済
            if prev_trend == -1 and (curr_trend == 1 or curr_trend == 0):
                exit_signals[i] = -1  # ショート決済
    
    return exit_signals


@njit(fastmath=True)
def detect_trend_direction_changes(trend_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    トレンド方向の変化を検出する（高速化版）
    
    Args:
        trend_direction: トレンド方向の配列
    
    Returns:
        (long_entry_points, short_entry_points): エントリーポイントの配列
    """
    length = len(trend_direction)
    long_entry = np.zeros(length, dtype=np.bool_)
    short_entry = np.zeros(length, dtype=np.bool_)
    
    for i in range(1, length):
        if not (np.isnan(trend_direction[i]) or np.isnan(trend_direction[i-1])):
            prev_trend = trend_direction[i-1]
            curr_trend = trend_direction[i]
            
            # -1から1への変化（ロングエントリー）
            if prev_trend == -1 and curr_trend == 1:
                long_entry[i] = True
            
            # 1から-1への変化（ショートエントリー）
            elif prev_trend == 1 and curr_trend == -1:
                short_entry[i] = True
    
    return long_entry, short_entry


class UltimateChopTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    Ultimate Chop Trend V3によるエントリーシグナル
    
    特徴:
    - Ultimate Chop Trend V3のtrend_direction変化を検出
    - トレンド方向の転換点でエントリー/決済シグナルを生成
    - Numbaによる高速化処理
    - 信頼度による品質フィルタリング
    
    エントリー条件:
    - ロング: trend_direction が -1 から 1 に変化
    - ショート: trend_direction が 1 から -1 に変化
    
    決済条件:
    - ロング決済: trend_direction が 1 から -1 または 0 に変化
    - ショート決済: trend_direction が -1 から 1 または 0 に変化
    """
    
    def __init__(
        self,
        # Ultimate Chop Trend V3パラメータ
        analysis_period: int = 14,
        fast_period: int = 7,
        trend_threshold: float = 0.58,
        confidence_threshold: float = 0.3,
        
        # アルゴリズム有効化（全て軽量で効果的）
        enable_hilbert: bool = True,
        enable_regression: bool = True,
        enable_consensus: bool = True,
        enable_volatility: bool = True,
        enable_zerollag: bool = True,
        
        # シグナル生成パラメータ
        enable_exit_signals: bool = True,
        min_confidence_for_entry: float = 0.3,
        require_strong_confidence: bool = False
    ):
        """
        コンストラクタ
        
        Args:
            # Ultimate Chop Trend V3パラメータ
            analysis_period: 分析期間（デフォルト: 14）
            fast_period: 高速期間（デフォルト: 7）
            trend_threshold: トレンド判定しきい値（デフォルト: 0.58）
            confidence_threshold: 信頼度しきい値（デフォルト: 0.3）
            
            # アルゴリズム有効化
            enable_hilbert: ヒルベルト変換を有効にするか
            enable_regression: 増分回帰を有効にするか
            enable_consensus: コンセンサスを有効にするか
            enable_volatility: ボラティリティ分析を有効にするか
            enable_zerollag: ゼロラグEMAを有効にするか
            
            # シグナル生成パラメータ
            enable_exit_signals: 決済シグナルを有効にするか
            min_confidence_for_entry: エントリーに必要な最小信頼度
            require_strong_confidence: 強い信頼度を要求するか
        """
        params = {
            'analysis_period': analysis_period,
            'fast_period': fast_period,
            'trend_threshold': trend_threshold,
            'confidence_threshold': confidence_threshold,
            'enable_hilbert': enable_hilbert,
            'enable_regression': enable_regression,
            'enable_consensus': enable_consensus,
            'enable_volatility': enable_volatility,
            'enable_zerollag': enable_zerollag,
            'enable_exit_signals': enable_exit_signals,
            'min_confidence_for_entry': min_confidence_for_entry,
            'require_strong_confidence': require_strong_confidence
        }
        
        super().__init__(
            f"UltimateChopTrendEntry(P={analysis_period}, thresh={trend_threshold}, conf={confidence_threshold})",
            params
        )
        
        # Ultimate Chop Trend V3のインスタンス化
        self._ultimate_chop_trend = UltimateChopTrendV3(
            analysis_period=analysis_period,
            fast_period=fast_period,
            trend_threshold=trend_threshold,
            confidence_threshold=confidence_threshold,
            enable_hilbert=enable_hilbert,
            enable_regression=enable_regression,
            enable_consensus=enable_consensus,
            enable_volatility=enable_volatility,
            enable_zerollag=enable_zerollag
        )
        
        # 結果キャッシュ
        self._entry_signals = None
        self._exit_signals = None
        self._data_hash = None
        self._current_position = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラム（OHLC）のハッシュ
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in data.columns]
            if available_cols:
                data_hash = hash(tuple(map(tuple, data[available_cols].values)))
            else:
                # フォールバック
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合
                data_hash = hash(tuple(map(tuple, data)))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ（OHLC必須）
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._entry_signals is not None:
                return self._entry_signals
                
            self._data_hash = data_hash
            
            # Ultimate Chop Trend V3の計算
            result = self._ultimate_chop_trend.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # トレンド方向と信頼度の取得
            trend_direction = result.trend_direction.astype(np.int8)
            confidence_score = result.confidence_score
            
            # 信頼度フィルタリング（オプション）
            if self._params['require_strong_confidence']:
                # 強い信頼度を要求する場合
                low_confidence_mask = confidence_score < self._params['min_confidence_for_entry']
                trend_direction = trend_direction.copy()
                trend_direction[low_confidence_mask] = 0  # レンジに設定
            
            # エントリーシグナルの計算（高速化版）
            entry_signals = calculate_ultimate_chop_trend_signals(
                trend_direction,
                self._params['confidence_threshold']
            )
            
            # 信頼度による最終フィルタリング
            for i in range(len(entry_signals)):
                if entry_signals[i] != 0:  # シグナルがある場合
                    if confidence_score[i] < self._params['min_confidence_for_entry']:
                        entry_signals[i] = 0  # 信頼度が低い場合はシグナル無効
            
            # 結果をキャッシュ
            self._entry_signals = entry_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                # 簡易的なポジション追跡
                current_position = self._track_position(entry_signals)
                self._exit_signals = calculate_ultimate_chop_trend_exit_signals(
                    trend_direction,
                    current_position,
                    self._params['confidence_threshold']
                )
                self._current_position = current_position
            
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltimateChopTrendEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._entry_signals = np.zeros(len(data), dtype=np.int8)
            return self._entry_signals
    
    def _track_position(self, signals: np.ndarray) -> np.ndarray:
        """
        シグナルからポジションを追跡する
        
        Args:
            signals: エントリーシグナル（1=ロング、-1=ショート、0=なし）
        
        Returns:
            ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        length = len(signals)
        position = np.zeros(length, dtype=np.int8)
        current_pos = 0
        
        for i in range(length):
            if signals[i] == 1:  # ロングエントリー
                current_pos = 1
            elif signals[i] == -1:  # ショートエントリー
                current_pos = -1
            # signals[i] == 0の場合は現在のポジションを維持
            
            position[i] = current_pos
        
        return position
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        決済シグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
        """
        if data is not None:
            self.generate(data)
        
        if self._exit_signals is not None:
            return self._exit_signals.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        if data is not None:
            self.generate(data)
        
        if self._current_position is not None:
            return self._current_position.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_ultimate_chop_trend_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Ultimate Chop Trend V3の計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UltimateChopTrendV3Result: Ultimate Chop Trend V3の計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._ultimate_chop_trend.calculate(data)
    
    def get_trend_direction(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        return result.trend_direction if result is not None else np.array([])
    
    def get_trend_index(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        統合トレンド指数を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 統合トレンド指数の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        return result.trend_index if result is not None else np.array([])
    
    def get_trend_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド強度を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド強度の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        return result.trend_strength if result is not None else np.array([])
    
    def get_confidence_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        予測信頼度を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 予測信頼度の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        return result.confidence_score if result is not None else np.array([])
    
    def get_trend_change_points(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        トレンド変化点を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ロングエントリーポイント, ショートエントリーポイント)
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        if result is not None:
            return detect_trend_direction_changes(result.trend_direction)
        else:
            return np.array([]), np.array([])
    
    def get_current_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        現在の状態情報を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, Any]: 現在の状態情報
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_chop_trend.get_result()
        if result is not None:
            return {
                'current_trend': result.current_trend,
                'current_strength': result.current_strength,
                'current_confidence': result.current_confidence,
                'latest_direction': result.trend_direction[-1] if len(result.trend_direction) > 0 else 0,
                'latest_index': result.trend_index[-1] if len(result.trend_index) > 0 else 0.5,
                'component_values': {
                    'hilbert': result.hilbert_component[-1] if len(result.hilbert_component) > 0 else 0.5,
                    'regression': result.regression_component[-1] if len(result.regression_component) > 0 else 0.5,
                    'consensus': result.consensus_component[-1] if len(result.consensus_component) > 0 else 0.5,
                    'volatility': result.volatility_component[-1] if len(result.volatility_component) > 0 else 0.5,
                    'zerollag': result.zerollag_component[-1] if len(result.zerollag_component) > 0 else 0.5
                }
            }
        else:
            return {
                'current_trend': 'range',
                'current_strength': 0.0,
                'current_confidence': 0.0,
                'latest_direction': 0,
                'latest_index': 0.5,
                'component_values': {}
            }
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._ultimate_chop_trend.reset() if hasattr(self._ultimate_chop_trend, 'reset') else None
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None