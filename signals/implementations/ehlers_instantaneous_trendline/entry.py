#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.ehlers_instantaneous_trendline import EhlersInstantaneousTrendline


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    itrend_values: np.ndarray, 
    trigger_values: np.ndarray
) -> np.ndarray:
    """
    ITrendとTriggerの位置関係シグナルを計算する（高速化版）
    
    Args:
        itrend_values: ITrend値の配列
        trigger_values: Trigger値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(itrend_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # ITrend値とTrigger値が有効かチェック
        if np.isnan(itrend_values[i]) or np.isnan(trigger_values[i]):
            signals[i] = 0
            continue
            
        # Trigger > ITrend: ロングシグナル（Bullish）
        if trigger_values[i] > itrend_values[i]:
            signals[i] = 1
        # Trigger < ITrend: ショートシグナル（Bearish）
        elif trigger_values[i] < itrend_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=False)
def calculate_crossover_signals(
    itrend_values: np.ndarray, 
    trigger_values: np.ndarray
) -> np.ndarray:
    """
    ITrendとTriggerのクロスオーバーシグナルを計算する（改良版）
    
    クロス検出アルゴリズム:
    1. Trigger（短期）とITrend（長期）の位置関係を示す配列を作成
    2. 前日のPositionと比較するために、Positionを1つずらした配列を作成
    3. Positionが-1から1に変わった点をゴールデンクロス（ロングシグナル）とする
    4. Positionが1から-1に変わった点をデッドクロス（ショートシグナル）とする
    
    Args:
        itrend_values: ITrend値の配列（長期）
        trigger_values: Trigger値の配列（短期）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(itrend_values)
    signals = np.zeros(length, dtype=np.int8)
    
    if length < 2:
        return signals
    
    # 1. 短期（Trigger）と長期（ITrend）の位置関係を示す配列を作成
    # （1: Trigger > ITrend, -1: Trigger <= ITrend, 0: 無効）
    position = np.zeros(length, dtype=np.int8)
    for i in range(length):
        if np.isnan(itrend_values[i]) or np.isnan(trigger_values[i]):
            position[i] = 0  # 無効データ
        elif trigger_values[i] > itrend_values[i]:
            position[i] = 1   # Trigger > ITrend
        else:
            position[i] = -1  # Trigger <= ITrend
    
    # 2. 前日のPositionと比較してクロスを検出
    for i in range(1, length):
        # 現在と前回のPositionが両方とも有効な場合のみクロス判定
        if position[i] != 0 and position[i-1] != 0:
            # 3. Positionが-1から1に変わった点をゴールデンクロス（ロングシグナル）
            if position[i-1] == -1 and position[i] == 1:
                signals[i] = 1
            # 4. Positionが1から-1に変わった点をデッドクロス（ショートシグナル）
            elif position[i-1] == 1 and position[i] == -1:
                signals[i] = -1
    
    return signals


class EhlersInstantaneousTrendlinePositionEntrySignal(BaseSignal, IEntrySignal):
    """
    Ehlers Instantaneous Trendline位置関係によるエントリーシグナル
    
    特徴:
    - ITrend（瞬時トレンドライン）とTrigger線の位置関係でシグナル生成
    - HyperERによる動的アルファ適応対応
    - カルマン統合フィルター + アルティメットスムーサーによる平滑化対応
    - プライスソース対応
    
    シグナル条件:
    - Trigger > ITrend: ロングシグナル (1) - Bullish
    - Trigger < ITrend: ショートシグナル (-1) - Bearish
    - Trigger = ITrend: シグナルなし (0) - Neutral
    """
    
    def __init__(
        self,
        # Ehlers Instantaneous Trendlineパラメータ
        alpha: float = 0.07,
        src_type: str = 'hl2',
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        alpha_min: float = 0.04,
        alpha_max: float = 0.15,
        # 平滑化モード設定
        smoothing_mode: str = 'none',
        # 統合カルマンフィルターパラメータ
        kalman_filter_type: str = 'simple',
        kalman_process_noise: float = 1e-5,
        kalman_min_observation_noise: float = 1e-6,
        kalman_adaptation_window: int = 5,
        # Ultimate Smootherパラメータ
        ultimate_smoother_period: int = 10
    ):
        """
        初期化
        
        Args:
            alpha: アルファ値（0.01-1.0の範囲、デフォルト: 0.07）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか（デフォルト: True）
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン期間（デフォルト: 100）
            alpha_min: アルファ最小値（HyperER低い時）（デフォルト: 0.04）
            alpha_max: アルファ最大値（HyperER高い時）（デフォルト: 0.15）
            smoothing_mode: 平滑化モード（デフォルト: 'none'） - 'none', 'kalman', 'ultimate', 'kalman_ultimate'
            kalman_filter_type: カルマンフィルタータイプ（'simple', 'unscented', 'unscented_v2', 'adaptive', 'multivariate', 'quantum_adaptive'）（デフォルト: 'simple'）
            kalman_process_noise: カルマンフィルター プロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: カルマンフィルター 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: カルマンフィルター 適応ウィンドウ（デフォルト: 5）
            ultimate_smoother_period: Ultimate Smoother 期間（デフォルト: 10）
        """
        # 動的適応・平滑化文字列の作成
        feature_str = ""
        if enable_hyper_er_adaptation:
            feature_str += f"_hyper_er({hyper_er_period},{hyper_er_midline_period})"
        if smoothing_mode != 'none':
            if smoothing_mode == 'kalman_ultimate':
                feature_str += f"_smooth(kalman+ultimate)"
            else:
                feature_str += f"_smooth({smoothing_mode})"
        
        super().__init__(
            f"EhlersInstantaneousTrendlinePositionEntrySignal(alpha={alpha_min}-{alpha_max if enable_hyper_er_adaptation else alpha}, {src_type}{feature_str})"
        )
        
        # パラメータの保存
        self._params = {
            'alpha': alpha,
            'src_type': src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'smoothing_mode': smoothing_mode,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_min_observation_noise': kalman_min_observation_noise,
            'kalman_adaptation_window': kalman_adaptation_window,
            'ultimate_smoother_period': ultimate_smoother_period
        }
        
        # Ehlers Instantaneous Trendlineインジケーターの初期化
        self.ehlers_indicator = EhlersInstantaneousTrendline(
            alpha=alpha,
            src_type=src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            smoothing_mode=smoothing_mode,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_min_observation_noise=kalman_min_observation_noise,
            kalman_adaptation_window=kalman_adaptation_window,
            ultimate_smoother_period=ultimate_smoother_period
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # Ehlers Instantaneous Trendlineの計算
            ehlers_result = self.ehlers_indicator.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if ehlers_result is None or len(ehlers_result.itrend_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ITrend値とTrigger値の取得
            itrend_values = ehlers_result.itrend_values
            trigger_values = ehlers_result.trigger_values
            
            # 位置関係シグナルの計算（高速化版）
            signals = calculate_position_signals(
                itrend_values,
                trigger_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"EhlersInstantaneousTrendlinePositionEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_itrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ITrend値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ITrend値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_itrend_values()
    
    def get_trigger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Trigger値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Trigger値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_trigger_values()
    
    def get_signal_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        内部シグナル値（指標計算結果のシグナル）を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: シグナル値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_signal_values()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        使用されたアルファ値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルファ値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_alpha_values()
    
    def get_smoothed_prices(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        平滑化後の価格を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 平滑化後の価格
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_smoothed_prices()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.ehlers_indicator.reset() if hasattr(self.ehlers_indicator, 'reset') else None
        self._signals_cache = {}


class EhlersInstantaneousTrendlineCrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    Ehlers Instantaneous Trendlineクロスオーバーによるエントリーシグナル
    
    特徴:
    - ITrend（瞬時トレンドライン）とTrigger線のクロスオーバーでシグナル生成
    - HyperERによる動的アルファ適応対応
    - カルマン統合フィルター + アルティメットスムーサーによる平滑化対応
    - プライスソース対応
    
    シグナル条件:
    - ロング: 前回 Trigger <= ITrend かつ 現在 Trigger > ITrend (1)
    - ショート: 前回 Trigger >= ITrend かつ 現在 Trigger < ITrend (-1)
    - その他: シグナルなし (0)
    """
    
    def __init__(
        self,
        # Ehlers Instantaneous Trendlineパラメータ
        alpha: float = 0.07,
        src_type: str = 'hl2',
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        alpha_min: float = 0.04,
        alpha_max: float = 0.15,
        # 平滑化モード設定
        smoothing_mode: str = 'none',
        # 統合カルマンフィルターパラメータ
        kalman_filter_type: str = 'simple',
        kalman_process_noise: float = 1e-5,
        kalman_min_observation_noise: float = 1e-6,
        kalman_adaptation_window: int = 5,
        # Ultimate Smootherパラメータ
        ultimate_smoother_period: int = 10
    ):
        """
        初期化
        
        Args:
            alpha: アルファ値（0.01-1.0の範囲、デフォルト: 0.07）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか（デフォルト: True）
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン期間（デフォルト: 100）
            alpha_min: アルファ最小値（HyperER低い時）（デフォルト: 0.04）
            alpha_max: アルファ最大値（HyperER高い時）（デフォルト: 0.15）
            smoothing_mode: 平滑化モード（デフォルト: 'none'） - 'none', 'kalman', 'ultimate', 'kalman_ultimate'
            kalman_filter_type: カルマンフィルタータイプ（'simple', 'unscented', 'unscented_v2', 'adaptive', 'multivariate', 'quantum_adaptive'）（デフォルト: 'simple'）
            kalman_process_noise: カルマンフィルター プロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: カルマンフィルター 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: カルマンフィルター 適応ウィンドウ（デフォルト: 5）
            ultimate_smoother_period: Ultimate Smoother 期間（デフォルト: 10）
        """
        # 動的適応・平滑化文字列の作成
        feature_str = ""
        if enable_hyper_er_adaptation:
            feature_str += f"_hyper_er({hyper_er_period},{hyper_er_midline_period})"
        if smoothing_mode != 'none':
            if smoothing_mode == 'kalman_ultimate':
                feature_str += f"_smooth(kalman+ultimate)"
            else:
                feature_str += f"_smooth({smoothing_mode})"
        
        super().__init__(
            f"EhlersInstantaneousTrendlineCrossoverEntrySignal(alpha={alpha_min}-{alpha_max if enable_hyper_er_adaptation else alpha}, {src_type}{feature_str})"
        )
        
        # パラメータの保存
        self._params = {
            'alpha': alpha,
            'src_type': src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'smoothing_mode': smoothing_mode,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_min_observation_noise': kalman_min_observation_noise,
            'kalman_adaptation_window': kalman_adaptation_window,
            'ultimate_smoother_period': ultimate_smoother_period
        }
        
        # Ehlers Instantaneous Trendlineインジケーターの初期化
        self.ehlers_indicator = EhlersInstantaneousTrendline(
            alpha=alpha,
            src_type=src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            smoothing_mode=smoothing_mode,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_min_observation_noise=kalman_min_observation_noise,
            kalman_adaptation_window=kalman_adaptation_window,
            ultimate_smoother_period=ultimate_smoother_period
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # Ehlers Instantaneous Trendlineの計算
            ehlers_result = self.ehlers_indicator.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if ehlers_result is None or len(ehlers_result.itrend_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ITrend値とTrigger値の取得
            itrend_values = ehlers_result.itrend_values
            trigger_values = ehlers_result.trigger_values
            
            # クロスオーバーシグナルの計算（高速化版）
            signals = calculate_crossover_signals(
                itrend_values,
                trigger_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"EhlersInstantaneousTrendlineCrossoverEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_itrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ITrend値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ITrend値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_itrend_values()
    
    def get_trigger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Trigger値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Trigger値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_trigger_values()
    
    def get_signal_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        内部シグナル値（指標計算結果のシグナル）を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: シグナル値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_signal_values()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        使用されたアルファ値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルファ値
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_alpha_values()
    
    def get_smoothed_prices(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        平滑化後の価格を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 平滑化後の価格
        """
        if data is not None:
            self.generate(data)
            
        return self.ehlers_indicator.get_smoothed_prices()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.ehlers_indicator.reset() if hasattr(self.ehlers_indicator, 'reset') else None
        self._signals_cache = {}