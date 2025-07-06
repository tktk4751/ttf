#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_chop_trend.ultimate_chop_trend_entry import UltimateChopTrendEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_exit_signals_numba(trend_direction: np.ndarray, entry_signals: np.ndarray,
                                current_position: int, current_index: int, 
                                use_strong_exit: bool) -> bool:
    """
    決済シグナル計算（Numba高速化版）
    
    Args:
        trend_direction: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        entry_signals: エントリーシグナル配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
        use_strong_exit: 強い決済条件を使用するか
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(trend_direction) or current_index < 1:
        return False
    
    # NaN値チェック
    if np.isnan(trend_direction[current_index]) or np.isnan(trend_direction[current_index-1]):
        return False
    
    prev_trend = trend_direction[current_index-1]
    curr_trend = trend_direction[current_index]
    
    if use_strong_exit:
        # 強い決済条件: トレンド方向が完全に反転または中立になった場合
        if current_position == 1:  # ロングポジション
            # 上昇トレンドから下降トレンドまたはレンジに変化
            return prev_trend == 1 and (curr_trend == -1 or curr_trend == 0)
        elif current_position == -1:  # ショートポジション
            # 下降トレンドから上昇トレンドまたはレンジに変化
            return prev_trend == -1 and (curr_trend == 1 or curr_trend == 0)
    else:
        # 通常の決済条件: トレンド方向が反転した場合のみ
        if current_position == 1:  # ロングポジション
            # 上昇トレンドから下降トレンドに変化
            return prev_trend == 1 and curr_trend == -1
        elif current_position == -1:  # ショートポジション
            # 下降トレンドから上昇トレンドに変化
            return prev_trend == -1 and curr_trend == 1
    
    return False


@njit(fastmath=True, parallel=True)
def filter_signals_by_confidence(signals: np.ndarray, confidence: np.ndarray, 
                                min_confidence: float) -> np.ndarray:
    """
    信頼度によるシグナルフィルタリング（Numba高速化版）
    
    Args:
        signals: 元のシグナル配列
        confidence: 信頼度配列
        min_confidence: 最小信頼度
    
    Returns:
        フィルタリングされたシグナル配列
    """
    length = len(signals)
    filtered_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        if not np.isnan(confidence[i]) and confidence[i] >= min_confidence:
            filtered_signals[i] = signals[i]
        else:
            filtered_signals[i] = 0
    
    return filtered_signals


class UltimateChopTrendSignalGenerator(BaseSignalGenerator):
    """
    Ultimate Chop Trend V3のシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: trend_direction が -1 から 1 に変化 + 信頼度条件
    - ショート: trend_direction が 1 から -1 に変化 + 信頼度条件
    
    エグジット条件:
    - 通常: トレンド方向が反転した時
    - 強制: トレンド方向が反転またはレンジになった時
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
        min_confidence_for_entry: float = 0.4,
        require_strong_confidence: bool = True,
        use_strong_exit: bool = False,
        confidence_filter_strength: float = 0.3
    ):
        """初期化"""
        super().__init__("UltimateChopTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
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
            'require_strong_confidence': require_strong_confidence,
            'use_strong_exit': use_strong_exit,
            'confidence_filter_strength': confidence_filter_strength
        }
        
        # Ultimate Chop Trend V3エントリーシグナルの初期化
        self.ultimate_chop_trend_signal = UltimateChopTrendEntrySignal(
            analysis_period=analysis_period,
            fast_period=fast_period,
            trend_threshold=trend_threshold,
            confidence_threshold=confidence_threshold,
            enable_hilbert=enable_hilbert,
            enable_regression=enable_regression,
            enable_consensus=enable_consensus,
            enable_volatility=enable_volatility,
            enable_zerollag=enable_zerollag,
            enable_exit_signals=enable_exit_signals,
            min_confidence_for_entry=min_confidence_for_entry,
            require_strong_confidence=require_strong_confidence
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._trend_direction = None
        self._confidence_score = None
        self._ultimate_chop_trend_result = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # Ultimate Chop Trend V3シグナルの計算
                try:
                    # エントリーシグナルの計算
                    ultimate_chop_trend_signals = self.ultimate_chop_trend_signal.generate(df)
                    
                    # Ultimate Chop Trend V3の結果を取得
                    self._ultimate_chop_trend_result = self.ultimate_chop_trend_signal.get_ultimate_chop_trend_result(df)
                    
                    # トレンド方向と信頼度スコアを取得
                    self._trend_direction = self.ultimate_chop_trend_signal.get_trend_direction(df)
                    self._confidence_score = self.ultimate_chop_trend_signal.get_confidence_score(df)
                    
                    # 信頼度によるフィルタリング（オプション）
                    if self._params['confidence_filter_strength'] > 0:
                        filtered_signals = filter_signals_by_confidence(
                            ultimate_chop_trend_signals,
                            self._confidence_score,
                            self._params['confidence_filter_strength']
                        )
                        self._signals = filtered_signals
                    else:
                        self._signals = ultimate_chop_trend_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate Chop Trend V3シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._trend_direction = np.zeros(current_len, dtype=np.int8)
                    self._confidence_score = np.zeros(current_len, dtype=np.float32)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._trend_direction = np.zeros(len(data), dtype=np.int8)
                self._confidence_score = np.zeros(len(data), dtype=np.float32)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba高速化された決済シグナル計算
        if self._trend_direction is not None:
            return calculate_exit_signals_numba(
                self._trend_direction,
                self._signals,
                position,
                index,
                self._params['use_strong_exit']
            )
        
        return False
    
    def get_trend_direction(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=レンジ）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._trend_direction is not None:
                return self._trend_direction.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"トレンド方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        統合トレンド指数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 統合トレンド指数の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_chop_trend_signal.get_trend_index(data)
        except Exception as e:
            self.logger.error(f"統合トレンド指数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド強度を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド強度の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_chop_trend_signal.get_trend_strength(data)
        except Exception as e:
            self.logger.error(f"トレンド強度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_confidence_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        予測信頼度を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 予測信頼度の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._confidence_score is not None:
                return self._confidence_score.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"予測信頼度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_chop_trend_signal.get_current_position(data)
        except Exception as e:
            self.logger.error(f"現在のポジション状態取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_change_points(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        トレンド変化点を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ロングエントリーポイント, ショートエントリーポイント)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_chop_trend_signal.get_trend_change_points(data)
        except Exception as e:
            self.logger.error(f"トレンド変化点取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_current_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        現在の状態情報を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, Any]: 現在の状態情報
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_chop_trend_signal.get_current_state(data)
        except Exception as e:
            self.logger.error(f"現在の状態情報取得中にエラー: {str(e)}")
            return {}
    
    def get_ultimate_chop_trend_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Ultimate Chop Trend V3の完全な結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Ultimate Chop Trend V3の計算結果オブジェクト
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_chop_trend_result is not None:
                return self._ultimate_chop_trend_result
            else:
                return self.ultimate_chop_trend_signal.get_ultimate_chop_trend_result(data)
        except Exception as e:
            self.logger.error(f"Ultimate Chop Trend V3結果取得中にエラー: {str(e)}")
            return None
    
    def get_all_components(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Ultimate Chop Trend V3の全コンポーネントを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: 全コンポーネントの辞書
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            result = self.get_ultimate_chop_trend_result(data)
            if result is not None:
                return {
                    'trend_direction': result.trend_direction.copy(),
                    'trend_index': result.trend_index.copy(),
                    'trend_strength': result.trend_strength.copy(),
                    'confidence_score': result.confidence_score.copy(),
                    'hilbert_component': result.hilbert_component.copy() if hasattr(result, 'hilbert_component') else np.array([]),
                    'regression_component': result.regression_component.copy() if hasattr(result, 'regression_component') else np.array([]),
                    'consensus_component': result.consensus_component.copy() if hasattr(result, 'consensus_component') else np.array([]),
                    'volatility_component': result.volatility_component.copy() if hasattr(result, 'volatility_component') else np.array([]),
                    'zerollag_component': result.zerollag_component.copy() if hasattr(result, 'zerollag_component') else np.array([])
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"全コンポーネント取得中にエラー: {str(e)}")
            return {}
    
    def get_filter_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        フィルタリング統計を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, Any]: フィルタリング統計
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._signals is not None and self._confidence_score is not None:
                total_signals = np.sum(np.abs(self._signals))
                high_conf_signals = np.sum(np.abs(self._signals[self._confidence_score >= self._params['min_confidence_for_entry']]))
                avg_confidence = np.mean(self._confidence_score[self._confidence_score > 0])
                
                return {
                    'total_signals': int(total_signals),
                    'high_confidence_signals': int(high_conf_signals),
                    'signal_quality_ratio': high_conf_signals / max(total_signals, 1),
                    'average_confidence': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                    'confidence_threshold': self._params['min_confidence_for_entry'],
                    'filter_strength': self._params['confidence_filter_strength']
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"フィルタリング統計取得中にエラー: {str(e)}")
            return {} 