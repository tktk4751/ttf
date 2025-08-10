#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_adaptive_supertrend.entry import HyperAdaptiveSupertrendEntrySignal
from signals.implementations.phasor_trend_filter.filter import PhasorTrendFilterSignal
from signals.implementations.correlation_cycle.filter import CorrelationCycleFilterSignal
from signals.implementations.correlation_trend.filter import CorrelationTrendFilterSignal
from signals.implementations.unified_trend_cycle.filter import UnifiedTrendCycleFilterSignal


class HyperAdaptiveSupertrendSignalGenerator(BaseSignalGenerator):
    """
    ハイパーアダプティブスーパートレンドシグナル生成クラス
    
    エントリー条件:
    - トレンド変化モード: トレンド転換によるシグナル
      * 下降→上昇トレンド転換: ロングシグナル
      * 上昇→下降トレンド転換: ショートシグナル
    - 位置関係モード: 価格とスーパートレンドラインの位置関係 + トレンド方向
      * 上昇トレンドかつ価格 > スーパートレンドライン: ロングシグナル
      * 下降トレンドかつ価格 < スーパートレンドライン: ショートシグナル
    
    エグジット条件:
    - ロング: 下降トレンドに転換
    - ショート: 上昇トレンドに転換
    """
    
    def __init__(
        self,
        # ハイパーアダプティブスーパートレンドのパラメータ
        atr_period: int = 14,                     # X_ATR期間
        multiplier: float = 3.0,                  # ATR乗数
        atr_method: str = 'str',                  # X_ATRの計算方法
        atr_smoother_type: str = 'sma',           # X_ATRのスムーサータイプ
        midline_smoother_type: str = 'frama',     # ミッドラインスムーサータイプ
        midline_period: int = 21,                 # ミッドライン期間
        src_type: str = 'hlc3',                   # ソースタイプ
        # カルマンフィルターパラメータ
        enable_kalman: bool = True,               # カルマンフィルター使用フラグ
        kalman_alpha: float = 1.0,                # UKFアルファパラメータ
        kalman_beta: float = 2.0,                 # UKFベータパラメータ
        kalman_kappa: float = 0.0,                # UKFカッパパラメータ
        kalman_process_noise: float = 0.01,       # UKFプロセスノイズ
        # 動的期間パラメータ
        use_dynamic_period: bool = True,          # 動的期間を使用するか
        cycle_part: float = 0.5,                  # サイクル部分の倍率
        detector_type: str = 'hody_e',            # 検出器タイプ
        max_cycle: int = 124,                     # 最大サイクル期間
        min_cycle: int = 13,                      # 最小サイクル期間
        max_output: int = 124,                    # 最大出力値
        min_output: int = 13,                     # 最小出力値
        lp_period: int = 13,                      # ローパスフィルター期間
        hp_period: int = 124,                     # ハイパスフィルター期間
        # シグナル設定
        trend_change_mode: bool = True,           # トレンド変化シグナル(True)または位置関係シグナル(False)
        # フィルターシグナル設定
        enable_filter_signals: bool = True,       # フィルターシグナルを有効にするか
        phasor_filter_period: int = 20,           # PhasorTrendFilterの期間
        phasor_filter_threshold: float = 6.0,     # PhasorTrendFilterの閾値
        correlation_cycle_period: int = 20,       # CorrelationCycleFilterの期間
        correlation_cycle_threshold: float = 9.0, # CorrelationCycleFilterの閾値
        correlation_trend_length: int = 20,       # CorrelationTrendFilterの長さ
        correlation_trend_threshold: float = 0.3, # CorrelationTrendFilterの閾値
        unified_trend_cycle_period: int = 20,     # UnifiedTrendCycleFilterの期間
        unified_trend_cycle_threshold: float = 0.5, # UnifiedTrendCycleFilterの閾値
        filter_consensus_mode: bool = True        # フィルター合意モード（True=全フィルター合意、False=多数決）
    ):
        """
        初期化
        
        Args:
            atr_period: X_ATR期間（デフォルト: 14）
            multiplier: ATR乗数（デフォルト: 3.0）
            atr_method: X_ATRの計算方法（'atr' または 'str'、デフォルト: 'str'）
            atr_smoother_type: X_ATRのスムーサータイプ（デフォルト: 'sma'）
            midline_smoother_type: ミッドラインスムーサータイプ（デフォルト: 'frama'）
            midline_period: ミッドライン期間（デフォルト: 21）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            enable_kalman: カルマンフィルター使用フラグ（デフォルト: True）
            kalman_alpha: UKFアルファパラメータ（デフォルト: 1.0）
            kalman_beta: UKFベータパラメータ（デフォルト: 2.0）
            kalman_kappa: UKFカッパパラメータ（デフォルト: 0.0）
            kalman_process_noise: UKFプロセスノイズ（デフォルト: 0.01）
            use_dynamic_period: 動的期間を使用するか（デフォルト: True）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            detector_type: 検出器タイプ（デフォルト: 'hody_e'）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 124）
            min_output: 最小出力値（デフォルト: 13）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            trend_change_mode: トレンド変化シグナル(True)または位置関係シグナル(False)
            enable_filter_signals: フィルターシグナルを有効にするか（デフォルト: True）
            phasor_filter_period: PhasorTrendFilterの期間（デフォルト: 20）
            phasor_filter_threshold: PhasorTrendFilterの閾値（デフォルト: 6.0）
            correlation_cycle_period: CorrelationCycleFilterの期間（デフォルト: 20）
            correlation_cycle_threshold: CorrelationCycleFilterの閾値（デフォルト: 9.0）
            correlation_trend_length: CorrelationTrendFilterの長さ（デフォルト: 20）
            correlation_trend_threshold: CorrelationTrendFilterの閾値（デフォルト: 0.3）
            unified_trend_cycle_period: UnifiedTrendCycleFilterの期間（デフォルト: 20）
            unified_trend_cycle_threshold: UnifiedTrendCycleFilterの閾値（デフォルト: 0.5）
            filter_consensus_mode: フィルター合意モード（True=全フィルター合意、False=多数決）
        """
        signal_type = "TrendChange" if trend_change_mode else "Position"
        kalman_str = f"_kalman({kalman_alpha},{kalman_beta},{kalman_kappa})" if enable_kalman else ""
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        filter_str = "_filters" if enable_filter_signals else ""
        
        super().__init__(
            f"HyperAdaptiveSupertrend{signal_type}SignalGenerator("
            f"atr={atr_period}×{multiplier}_{atr_method}_{atr_smoother_type}, "
            f"mid={midline_period}_{midline_smoother_type}, "
            f"{src_type}{kalman_str}{dynamic_str}{filter_str})"
        )
        
        # パラメータの設定
        self._params = {
            'atr_period': atr_period,
            'multiplier': multiplier,
            'atr_method': atr_method,
            'atr_smoother_type': atr_smoother_type,
            'midline_smoother_type': midline_smoother_type,
            'midline_period': midline_period,
            'src_type': src_type,
            'enable_kalman': enable_kalman,
            'kalman_alpha': kalman_alpha,
            'kalman_beta': kalman_beta,
            'kalman_kappa': kalman_kappa,
            'kalman_process_noise': kalman_process_noise,
            'use_dynamic_period': use_dynamic_period,
            'cycle_part': cycle_part,
            'detector_type': detector_type,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'trend_change_mode': trend_change_mode,
            'enable_filter_signals': enable_filter_signals,
            'phasor_filter_period': phasor_filter_period,
            'phasor_filter_threshold': phasor_filter_threshold,
            'correlation_cycle_period': correlation_cycle_period,
            'correlation_cycle_threshold': correlation_cycle_threshold,
            'correlation_trend_length': correlation_trend_length,
            'correlation_trend_threshold': correlation_trend_threshold,
            'unified_trend_cycle_period': unified_trend_cycle_period,
            'unified_trend_cycle_threshold': unified_trend_cycle_threshold,
            'filter_consensus_mode': filter_consensus_mode
        }
        
        self.trend_change_mode = trend_change_mode
        self.enable_filter_signals = enable_filter_signals
        self.filter_consensus_mode = filter_consensus_mode
        
        # ハイパーアダプティブスーパートレンドエントリーシグナルの初期化
        self.hyper_supertrend_entry_signal = HyperAdaptiveSupertrendEntrySignal(
            atr_period=atr_period,
            multiplier=multiplier,
            atr_method=atr_method,
            atr_smoother_type=atr_smoother_type,
            midline_smoother_type=midline_smoother_type,
            midline_period=midline_period,
            src_type=src_type,
            enable_kalman=enable_kalman,
            kalman_alpha=kalman_alpha,
            kalman_beta=kalman_beta,
            kalman_kappa=kalman_kappa,
            kalman_process_noise=kalman_process_noise,
            use_dynamic_period=use_dynamic_period,
            cycle_part=cycle_part,
            detector_type=detector_type,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period,
            trend_change_mode=trend_change_mode
        )
        
        # フィルターシグナルの初期化（有効時のみ）
        self.phasor_filter = None
        self.correlation_cycle_filter = None
        self.correlation_trend_filter = None
        self.unified_trend_cycle_filter = None
        
        if self.enable_filter_signals:
            try:
                # PhasorTrendFilter
                self.phasor_filter = PhasorTrendFilterSignal(
                    period=phasor_filter_period,
                    trend_threshold=phasor_filter_threshold,
                    src_type=src_type
                )
                
                # CorrelationCycleFilter
                self.correlation_cycle_filter = CorrelationCycleFilterSignal(
                    period=correlation_cycle_period,
                    src_type=src_type,
                    trend_threshold=correlation_cycle_threshold
                )
                
                # CorrelationTrendFilter
                self.correlation_trend_filter = CorrelationTrendFilterSignal(
                    length=correlation_trend_length,
                    src_type=src_type,
                    trend_threshold=correlation_trend_threshold
                )
                
                # UnifiedTrendCycleFilter
                self.unified_trend_cycle_filter = UnifiedTrendCycleFilterSignal(
                    period=unified_trend_cycle_period,
                    trend_length=correlation_trend_length,
                    trend_threshold=unified_trend_cycle_threshold,
                    src_type=src_type
                )
                
                self.logger.info("フィルターシグナル初期化完了")
            except Exception as e:
                self.logger.error(f"フィルターシグナル初期化エラー: {e}")
                self.enable_filter_signals = False
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._trend_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # ハイパーアダプティブスーパートレンドシグナルの計算
                try:
                    supertrend_signals = self.hyper_supertrend_entry_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = supertrend_signals
                    
                    # エグジット用のトレンド情報を事前計算
                    trend_values = self.hyper_supertrend_entry_signal.get_trend_values(df)
                    self._trend_signals = trend_values
                    
                    # フィルターシグナルの計算（有効時のみ）
                    if self.enable_filter_signals:
                        self._filter_signals = self._calculate_filter_signals(df)
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._trend_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = None
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._trend_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = None
                self._data_len = len(data)
    
    def _calculate_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """フィルターシグナルを計算する"""
        filter_signals = {}
        
        try:
            if self.phasor_filter is not None:
                filter_signals['phasor'] = self.phasor_filter.generate(data)
            
            if self.correlation_cycle_filter is not None:
                filter_signals['correlation_cycle'] = self.correlation_cycle_filter.generate(data)
            
            if self.correlation_trend_filter is not None:
                filter_signals['correlation_trend'] = self.correlation_trend_filter.generate(data)
            
            if self.unified_trend_cycle_filter is not None:
                filter_signals['unified_trend_cycle'] = self.unified_trend_cycle_filter.generate(data)
                
        except Exception as e:
            self.logger.error(f"フィルターシグナル計算エラー: {e}")
            
        return filter_signals
    
    def _apply_filter_consensus(self, entry_signals: np.ndarray, filter_signals: Dict[str, np.ndarray]) -> np.ndarray:
        """フィルター合意を適用してエントリーシグナルを調整する"""
        if not filter_signals or len(filter_signals) == 0:
            return entry_signals
        
        filtered_signals = entry_signals.copy()
        data_length = len(entry_signals)
        
        try:
            # 各時点でのフィルター評価
            for i in range(data_length):
                if entry_signals[i] == 0:  # シグナルがない場合はそのまま
                    continue
                
                filter_votes = []
                
                # 各フィルターの投票を収集
                for filter_name, filter_signal in filter_signals.items():
                    if i < len(filter_signal) and not np.isnan(filter_signal[i]):
                        if entry_signals[i] > 0:  # ロングシグナル
                            # フィルターがトレンド（1）またはサイクル（0）の場合は許可
                            if filter_signal[i] >= 0:
                                filter_votes.append(1)
                            else:
                                filter_votes.append(0)
                        elif entry_signals[i] < 0:  # ショートシグナル
                            # フィルターが下降トレンド（-1）の場合は許可
                            if filter_signal[i] <= 0:
                                filter_votes.append(1)
                            else:
                                filter_votes.append(0)
                
                # 合意判定
                if len(filter_votes) > 0:
                    if self.filter_consensus_mode:
                        # 全フィルター合意モード：全てが賛成の場合のみシグナル維持
                        if sum(filter_votes) == len(filter_votes):
                            # シグナル維持
                            pass
                        else:
                            # シグナル無効化
                            filtered_signals[i] = 0
                    else:
                        # 多数決モード：過半数が賛成の場合シグナル維持
                        if sum(filter_votes) > len(filter_votes) / 2:
                            # シグナル維持
                            pass
                        else:
                            # シグナル無効化
                            filtered_signals[i] = 0
                            
        except Exception as e:
            self.logger.error(f"フィルター合意適用エラー: {e}")
            return entry_signals
        
        return filtered_signals
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        # フィルターが有効で計算済みの場合はフィルター適用
        if self.enable_filter_signals and self._filter_signals is not None:
            return self._apply_filter_consensus(self._signals, self._filter_signals)
        
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # インデックスの範囲チェック
        if index < 0 or index >= len(self._trend_signals):
            return False
        
        # キャッシュされたトレンド情報を使用してエグジット判定
        current_trend = self._trend_signals[index]
        
        if self.trend_change_mode:
            # トレンド変化モードでは、逆方向のトレンドが出たらエグジット
            if position == 1:  # ロングポジション
                return bool(current_trend == -1)  # 下降トレンドでエグジット
            elif position == -1:  # ショートポジション
                return bool(current_trend == 1)   # 上昇トレンドでエグジット
        else:
            # 位置関係モードでも同様にトレンド変化でエグジット
            if position == 1:  # ロングポジション
                return bool(current_trend == -1)  # 下降トレンドでエグジット
            elif position == -1:  # ショートポジション
                return bool(current_trend == 1)   # 上昇トレンドでエグジット
        
        return False
    
    def get_supertrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        スーパートレンドライン値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: スーパートレンドライン値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_supertrend_values()
        except Exception as e:
            self.logger.error(f"スーパートレンドライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        上側バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 上側バンド値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_upper_band()
        except Exception as e:
            self.logger.error(f"上側バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        下側バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 下側バンド値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_lower_band()
        except Exception as e:
            self.logger.error(f"下側バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向値（1=上昇、-1=下降）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_trend_values()
        except Exception as e:
            self.logger.error(f"トレンド方向値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_midline_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ミッドライン値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ミッドライン値（統合スムーサー結果）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_midline_values()
        except Exception as e:
            self.logger.error(f"ミッドライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_atr_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATR値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_ATR値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_supertrend_entry_signal.get_atr_values()
        except Exception as e:
            self.logger.error(f"X_ATR値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: 全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return {
                'supertrend_values': self.get_supertrend_values(),
                'upper_band': self.get_upper_band(),
                'lower_band': self.get_lower_band(),
                'trend_values': self.get_trend_values(),
                'midline_values': self.get_midline_values(),
                'atr_values': self.get_atr_values()
            }
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """
        シグナルジェネレーターの状態をリセット
        """
        super().reset()
        self._data_len = 0
        self._signals = None
        self._trend_signals = None
        self._filter_signals = None
        if hasattr(self.hyper_supertrend_entry_signal, 'reset'):
            self.hyper_supertrend_entry_signal.reset()
        
        # フィルターシグナルのリセット
        if self.enable_filter_signals:
            for filter_obj in [self.phasor_filter, self.correlation_cycle_filter, 
                              self.correlation_trend_filter, self.unified_trend_cycle_filter]:
                if filter_obj is not None and hasattr(filter_obj, 'reset'):
                    filter_obj.reset()
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        フィルターシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: フィルターシグナル辞書
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._filter_signals or {}
        except Exception as e:
            self.logger.error(f"フィルターシグナル取得中にエラー: {str(e)}")
            return {}
    
    def get_phasor_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        PhasorTrendFilterの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: PhasorTrendFilterの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._filter_signals and 'phasor' in self._filter_signals:
                return self._filter_signals['phasor']
            return np.array([])
        except Exception as e:
            self.logger.error(f"PhasorTrendFilter値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_correlation_cycle_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CorrelationCycleFilterの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: CorrelationCycleFilterの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._filter_signals and 'correlation_cycle' in self._filter_signals:
                return self._filter_signals['correlation_cycle']
            return np.array([])
        except Exception as e:
            self.logger.error(f"CorrelationCycleFilter値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_correlation_trend_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CorrelationTrendFilterの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: CorrelationTrendFilterの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._filter_signals and 'correlation_trend' in self._filter_signals:
                return self._filter_signals['correlation_trend']
            return np.array([])
        except Exception as e:
            self.logger.error(f"CorrelationTrendFilter値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_unified_trend_cycle_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UnifiedTrendCycleFilterの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: UnifiedTrendCycleFilterの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if self._filter_signals and 'unified_trend_cycle' in self._filter_signals:
                return self._filter_signals['unified_trend_cycle']
            return np.array([])
        except Exception as e:
            self.logger.error(f"UnifiedTrendCycleFilter値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_consensus_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フィルター合意強度を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 合意強度（0-1の範囲）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            if not self._filter_signals:
                return np.array([])
            
            # 各時点での合意度を計算
            data_length = len(next(iter(self._filter_signals.values())))
            consensus_strength = np.zeros(data_length)
            
            for i in range(data_length):
                valid_votes = 0
                positive_votes = 0
                
                for filter_signal in self._filter_signals.values():
                    if i < len(filter_signal) and not np.isnan(filter_signal[i]):
                        valid_votes += 1
                        if filter_signal[i] > 0:  # トレンド方向
                            positive_votes += 1
                
                if valid_votes > 0:
                    consensus_strength[i] = abs(2 * positive_votes / valid_votes - 1)  # 0-1に正規化
                    
            return consensus_strength
        except Exception as e:
            self.logger.error(f"フィルター合意強度取得中にエラー: {str(e)}")
            return np.array([])