#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_filter.filter import AlphaFilterSignal
from signals.implementations.alpha_keltner.breakout_entry import AlphaKeltnerBreakoutEntrySignal
from signals.implementations.alpha_ma.direction import AlphaMACirculationSignal
from signals.implementations.alpha_trend.direction import AlphaTrendDirectionSignal
from signals.implementations.alpha_squeeze.entry import AlphaSqueezeEntrySignal
from signals.implementations.divergence.alpha_macd_divergence import AlphaMACDDivergenceSignal
from signals.implementations.divergence.alpha_macd_hidden_divergence import AlphaMACDHiddenDivergenceSignal
from signals.implementations.divergence.alpha_roc_divergence import AlphaROCDivergenceSignal
from signals.implementations.divergence.alpha_roc_hidden_divergence import AlphaROCHiddenDivergenceSignal

# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@jit(nopython=True)
def normalize_score(score: np.ndarray, target_max: float = 20.0) -> np.ndarray:
    """
    スコアを正規化する（最大値が20、最小値が-20になるように調整）
    
    Args:
        score: 元のスコア配列
        target_max: 目標とする最大値
        
    Returns:
        正規化されたスコア配列
    """
    # 0でない値の絶対値の最大値を計算
    non_zero = np.abs(score) > 0
    if np.sum(non_zero) == 0:
        return np.zeros_like(score)
    
    abs_max = np.max(np.abs(score[non_zero]))
    if abs_max == 0:
        return np.zeros_like(score)
    
    # 最大値がtarget_maxになるように調整
    normalized = score * (target_max / abs_max)
    
    # 最大値がtarget_maxを超えないようにクリップ
    return np.clip(normalized, -target_max, target_max)


class AlphaTrendPredictorGenerator(BaseSignalGenerator):
    """
    Alphaシグナルを組み合わせてスコアリングするトレンドプレディクター
    
    特徴:
    - 複数のAlphaシグナルを重み付けして組み合わせる
    - スコアを-20から20の範囲に正規化
    - スコアが10以上でロングエントリー、-10以下でショートエントリー
    
    使用するシグナル:
    - AlphaFilterSignal: フィルターシグナル
    - AlphaKeltnerBreakoutEntrySignal: エントリーシグナル
    - AlphaMACirculationSignal: ディレクションシグナル
    - AlphaTrendDirectionSignal: ディレクションシグナル
    - AlphaSqueezeEntrySignal: エントリーシグナル
    - AlphaMACDDivergenceSignal: エントリーシグナル
    - AlphaMACDHiddenDivergenceSignal: エントリーシグナル
    - AlphaROCDivergenceSignal: エントリーシグナル
    - AlphaROCHiddenDivergenceSignal: エントリーシグナル
    """
    
    def __init__(
        self,
        # 各シグナルの重み
        alpha_filter_weight: float = 2.0,
        alpha_keltner_weight: float = 2.5,
        alpha_ma_circulation_weight: float = 3.0,
        alpha_trend_weight: float = 3.0,
        alpha_squeeze_weight: float = 2.0,
        alpha_macd_divergence_weight: float = 2.5,
        alpha_macd_hidden_divergence_weight: float = 2.5,
        alpha_roc_divergence_weight: float = 2.5,
        alpha_roc_hidden_divergence_weight: float = 2.5,
        
        # エントリーしきい値
        entry_threshold: float = 10.0,
        
        # 各シグナルのパラメータ
        alpha_filter_period: int = 14,
        alpha_keltner_period: int = 20,
        alpha_keltner_atr_period: int = 10,
        alpha_keltner_atr_multiplier: float = 2.0,
        alpha_ma_circulation_er_period: int = 21,
        alpha_trend_period: int = 14,
        alpha_squeeze_bb_period: int = 20,
        alpha_squeeze_bb_std_dev: float = 2.0,
        alpha_squeeze_kc_period: int = 20,
        alpha_squeeze_kc_atr_period: int = 10,
        alpha_squeeze_kc_atr_multiplier: float = 1.5,
        alpha_macd_fast_period: int = 12,
        alpha_macd_slow_period: int = 26,
        alpha_macd_signal_period: int = 9,
        alpha_roc_period: int = 14,
        alpha_roc_smooth_period: int = 3,
        divergence_lookback: int = 10
    ):
        """初期化"""
        super().__init__("AlphaTrendPredictorGenerator")
        
        # エントリーしきい値
        self.entry_threshold = entry_threshold
        
        # 各シグナルの重み
        self.weights = {
            'alpha_filter': alpha_filter_weight,
            'alpha_keltner': alpha_keltner_weight,
            'alpha_ma_circulation': alpha_ma_circulation_weight,
            'alpha_trend': alpha_trend_weight,
            'alpha_squeeze': alpha_squeeze_weight,
            'alpha_macd_divergence': alpha_macd_divergence_weight,
            'alpha_macd_hidden_divergence': alpha_macd_hidden_divergence_weight,
            'alpha_roc_divergence': alpha_roc_divergence_weight,
            'alpha_roc_hidden_divergence': alpha_roc_hidden_divergence_weight
        }
        
        # シグナルの初期化
        self.alpha_filter = AlphaFilterSignal(
            period=alpha_filter_period
        )
        
        self.alpha_keltner = AlphaKeltnerBreakoutEntrySignal(
            period=alpha_keltner_period,
            atr_period=alpha_keltner_atr_period,
            atr_multiplier=alpha_keltner_atr_multiplier
        )
        
        self.alpha_ma_circulation = AlphaMACirculationSignal(
            er_period=alpha_ma_circulation_er_period
        )
        
        self.alpha_trend = AlphaTrendDirectionSignal(
            period=alpha_trend_period
        )
        
        self.alpha_squeeze = AlphaSqueezeEntrySignal(
            bb_period=alpha_squeeze_bb_period,
            bb_std_dev=alpha_squeeze_bb_std_dev,
            kc_period=alpha_squeeze_kc_period,
            kc_atr_period=alpha_squeeze_kc_atr_period,
            kc_atr_multiplier=alpha_squeeze_kc_atr_multiplier
        )
        
        self.alpha_macd_divergence = AlphaMACDDivergenceSignal(
            fast_period=alpha_macd_fast_period,
            slow_period=alpha_macd_slow_period,
            signal_period=alpha_macd_signal_period,
            lookback=divergence_lookback
        )
        
        self.alpha_macd_hidden_divergence = AlphaMACDHiddenDivergenceSignal(
            fast_period=alpha_macd_fast_period,
            slow_period=alpha_macd_slow_period,
            signal_period=alpha_macd_signal_period,
            lookback=divergence_lookback
        )
        
        self.alpha_roc_divergence = AlphaROCDivergenceSignal(
            period=alpha_roc_period,
            smooth_period=alpha_roc_smooth_period,
            lookback=divergence_lookback
        )
        
        self.alpha_roc_hidden_divergence = AlphaROCHiddenDivergenceSignal(
            period=alpha_roc_period,
            smooth_period=alpha_roc_smooth_period,
            lookback=divergence_lookback
        )
        
        # 計算結果のキャッシュ
        self._signals_cache = {}
        self._score_cache = None
        
        # パラメータの保存
        self._params = {
            'alpha_filter_weight': alpha_filter_weight,
            'alpha_keltner_weight': alpha_keltner_weight,
            'alpha_ma_circulation_weight': alpha_ma_circulation_weight,
            'alpha_trend_weight': alpha_trend_weight,
            'alpha_squeeze_weight': alpha_squeeze_weight,
            'alpha_macd_divergence_weight': alpha_macd_divergence_weight,
            'alpha_macd_hidden_divergence_weight': alpha_macd_hidden_divergence_weight,
            'alpha_roc_divergence_weight': alpha_roc_divergence_weight,
            'alpha_roc_hidden_divergence_weight': alpha_roc_hidden_divergence_weight,
            'entry_threshold': entry_threshold,
            'alpha_filter_period': alpha_filter_period,
            'alpha_keltner_period': alpha_keltner_period,
            'alpha_keltner_atr_period': alpha_keltner_atr_period,
            'alpha_keltner_atr_multiplier': alpha_keltner_atr_multiplier,
            'alpha_ma_circulation_er_period': alpha_ma_circulation_er_period,
            'alpha_trend_period': alpha_trend_period,
            'alpha_squeeze_bb_period': alpha_squeeze_bb_period,
            'alpha_squeeze_bb_std_dev': alpha_squeeze_bb_std_dev,
            'alpha_squeeze_kc_period': alpha_squeeze_kc_period,
            'alpha_squeeze_kc_atr_period': alpha_squeeze_kc_atr_period,
            'alpha_squeeze_kc_atr_multiplier': alpha_squeeze_kc_atr_multiplier,
            'alpha_macd_fast_period': alpha_macd_fast_period,
            'alpha_macd_slow_period': alpha_macd_slow_period,
            'alpha_macd_signal_period': alpha_macd_signal_period,
            'alpha_roc_period': alpha_roc_period,
            'alpha_roc_smooth_period': alpha_roc_smooth_period,
            'divergence_lookback': divergence_lookback
        }
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        全てのシグナルを計算する
        
        Args:
            data: 価格データ
        """
        # 各シグナルの計算
        self._signals_cache = {
            'alpha_filter': self.alpha_filter.generate(data),
            'alpha_keltner': self.alpha_keltner.generate(data),
            'alpha_ma_circulation': self.alpha_ma_circulation.generate(data),
            'alpha_trend': self.alpha_trend.generate(data),
            'alpha_squeeze': self.alpha_squeeze.generate(data),
            'alpha_macd_divergence': self.alpha_macd_divergence.generate(data),
            'alpha_macd_hidden_divergence': self.alpha_macd_hidden_divergence.generate(data),
            'alpha_roc_divergence': self.alpha_roc_divergence.generate(data),
            'alpha_roc_hidden_divergence': self.alpha_roc_hidden_divergence.generate(data)
        }
        
        # スコアの計算
        self._calculate_score()
    
    def _calculate_score(self) -> None:
        """
        各シグナルを組み合わせてスコアを計算する
        """
        if not self._signals_cache:
            logger.warning("シグナルが計算されていません。calculate_signals()を先に呼び出してください。")
            return
        
        # 各シグナルに重みを掛けて合計
        weighted_sum = np.zeros_like(self._signals_cache['alpha_trend'], dtype=float)
        
        for signal_name, signal_values in self._signals_cache.items():
            weight = self.weights.get(signal_name, 1.0)
            
            # AlphaFilterSignalは反転させる（フィルターがかかっている時は負のスコア）
            if signal_name == 'alpha_filter':
                # 1（フィルターがかかっている）を-1に、0（フィルターなし）を1に変換
                adjusted_signal = 1.0 - 2.0 * signal_values.astype(float)
            else:
                adjusted_signal = signal_values.astype(float)
            
            weighted_sum += weight * adjusted_signal
        
        # スコアを正規化（-20から20の範囲に）
        self._score_cache = normalize_score(weighted_sum)
    
    def get_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        スコアを取得する
        
        Args:
            data: 価格データ（Noneの場合はキャッシュを使用）
            
        Returns:
            スコア配列
        """
        if data is not None or self._score_cache is None:
            self.calculate_signals(data)
        
        return self._score_cache
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得する
        
        Args:
            data: 価格データ
            
        Returns:
            エントリーシグナル配列 (1: ロング、-1: ショート、0: エントリーなし)
        """
        # スコアの計算
        score = self.get_score(data)
        
        # エントリーシグナルの生成
        entry_signals = np.zeros_like(score, dtype=np.int8)
        
        # スコアがしきい値以上ならロング
        entry_signals[score >= self.entry_threshold] = 1
        
        # スコアがしきい値以下ならショート
        entry_signals[score <= -self.entry_threshold] = -1
        
        return entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルトは最新）
            
        Returns:
            エグジットすべきかどうか
        """
        # スコアの計算
        score = self.get_score(data)
        
        # インデックスの調整
        if index < 0:
            index = len(score) + index
        
        # 現在のスコア
        current_score = score[index]
        
        # ロングポジションの場合、スコアが0未満になったらエグジット
        if position == 1:
            return current_score < 0
        
        # ショートポジションの場合、スコアが0より大きくなったらエグジット
        elif position == -1:
            return current_score > 0
        
        # ポジションがない場合
        return False
    
    def get_signal_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全てのシグナル値を取得する
        
        Args:
            data: 価格データ（Noneの場合はキャッシュを使用）
            
        Returns:
            各シグナルの値を含む辞書
        """
        if data is not None or not self._signals_cache:
            self.calculate_signals(data)
        
        return self._signals_cache.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        パラメータを設定する
        
        Args:
            params: パラメータ辞書
        """
        # パラメータの更新
        self._params.update(params)
        
        # 重みの更新
        for key, value in params.items():
            if key.endswith('_weight') and key in self.weights:
                signal_name = key.replace('_weight', '')
                self.weights[signal_name] = value
        
        # エントリーしきい値の更新
        if 'entry_threshold' in params:
            self.entry_threshold = params['entry_threshold']
        
        # キャッシュのクリア
        self._signals_cache = {}
        self._score_cache = None 