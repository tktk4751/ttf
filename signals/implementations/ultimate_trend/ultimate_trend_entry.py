#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.ultimate_trend import UltimateTrend
from signals.implementations.uqatrd.filter import UQATRDFilterSignal


@njit(fastmath=True, parallel=True)
def calculate_ultimate_trend_signals(trend: np.ndarray, trend_signals: np.ndarray = None, 
                                   filter_signals: np.ndarray = None,
                                   use_filter: bool = False, use_uqatrd_filter: bool = False) -> np.ndarray:
    """
    Ultimate Trendのシグナルを計算する（高速化版）
    
    Args:
        trend: アルティメットトレンド方向の配列（1=上昇、-1=下降、0=なし）
        trend_signals: Ultimate MAのトレンドシグナル配列（1=上昇、-1=下降、0=range）
        filter_signals: UQATRDフィルターシグナル配列（1=トレンド相場、-1=レンジ相場）
        use_filter: フィルターオプション（True=trend_signalsでフィルタリング）
        use_uqatrd_filter: UQATRDフィルターオプション（True=UQATRDフィルターでフィルタリング）
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(trend)
    signals = np.zeros(length, dtype=np.int8)
    
    # エントリーシグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(trend[i]):
            signals[i] = 0
            continue
        
        # UQATRDフィルター条件チェック
        if use_uqatrd_filter and filter_signals is not None:
            if np.isnan(filter_signals[i]):
                signals[i] = 0
                continue
            # フィルターが1（トレンド相場）でない場合は見送り
            if filter_signals[i] != 1:
                signals[i] = 0
                continue
            
        if use_filter and trend_signals is not None:
            # フィルターオプション有効時
            if np.isnan(trend_signals[i]):
                signals[i] = 0
                continue
                
            # ロングエントリー: trend == 1 かつ trend_signals == 1
            if trend[i] == 1 and trend_signals[i] == 1:
                signals[i] = 1
            # ショートエントリー: trend == -1 かつ trend_signals == -1
            elif trend[i] == -1 and trend_signals[i] == -1:
                signals[i] = -1
            else:
                signals[i] = 0
        else:
            # ベースシグナル（フィルターなし）
            # ロングエントリー: trend == 1
            if trend[i] == 1:
                signals[i] = 1
            # ショートエントリー: trend == -1
            elif trend[i] == -1:
                signals[i] = -1
            else:
                signals[i] = 0
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_ultimate_trend_exit_signals(trend: np.ndarray, trend_signals: np.ndarray = None,
                                        filter_signals: np.ndarray = None,
                                        current_position: np.ndarray = None, 
                                        use_filter: bool = False, use_uqatrd_filter: bool = False) -> np.ndarray:
    """
    Ultimate Trendの決済シグナルを計算する（高速化版）
    
    Args:
        trend: アルティメットトレンド方向の配列（1=上昇、-1=下降、0=なし）
        trend_signals: Ultimate MAのトレンドシグナル配列（1=上昇、-1=下降、0=range）
        filter_signals: UQATRDフィルターシグナル配列（1=トレンド相場、-1=レンジ相場）
        current_position: 現在のポジション（1=ロング、-1=ショート、0=ポジションなし）
        use_filter: フィルターオプション（True=trend_signalsでフィルタリング）
        use_uqatrd_filter: UQATRDフィルターオプション（True=UQATRDフィルターでフィルタリング）
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(trend)
    exit_signals = np.zeros(length, dtype=np.int8)
    
    if current_position is None:
        return exit_signals
    
    # 決済シグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(trend[i]):
            continue
        
        # UQATRDフィルター条件チェック（決済時はフィルター条件を緩和）
        if use_uqatrd_filter and filter_signals is not None:
            if np.isnan(filter_signals[i]):
                continue
            # フィルターが-1（レンジ相場）の場合は決済を促進
            if filter_signals[i] == -1:
                # レンジ相場では積極的に決済
                if current_position[i] == 1:
                    exit_signals[i] = 1  # ロング決済
                elif current_position[i] == -1:
                    exit_signals[i] = -1  # ショート決済
                continue
            
        if use_filter and trend_signals is not None:
            # フィルターオプション有効時
            if np.isnan(trend_signals[i]):
                continue
                
            # ロングポジション決済: trend != 1 または trend_signals != 1
            if current_position[i] == 1:
                if trend[i] != 1 or trend_signals[i] != 1:
                    exit_signals[i] = 1  # ロング決済
            # ショートポジション決済: trend != -1 または trend_signals != -1
            elif current_position[i] == -1:
                if trend[i] != -1 or trend_signals[i] != -1:
                    exit_signals[i] = -1  # ショート決済
        else:
            # ベース決済（フィルターなし）
            # ロングポジション決済: trend != 1
            if current_position[i] == 1:
                if trend[i] != 1:
                    exit_signals[i] = 1  # ロング決済
            # ショートポジション決済: trend != -1
            elif current_position[i] == -1:
                if trend[i] != -1:
                    exit_signals[i] = -1  # ショート決済
    
    return exit_signals


class UltimateTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    Ultimate Trendによるエントリーシグナル
    
    特徴:
    - Ultimate MAフィルタリングシステムとスーパートレンドロジックの統合
    - アルティメットトレンドのトレンド方向によるシグナル生成
    - オプションでUltimate MAのトレンドシグナルによるフィルタリング
    - Numbaによる高速化処理
    - ATRベースの動的バンド調整
    
    エントリー条件:
    - ベース: trend == 1 でロング、trend == -1 でショート
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが同方向
    
    決済条件:
    - ベース: trendが反転または0になった時
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが反転
    """
    
    def __init__(
        self,
        # Ultimate Trendパラメータ
        length: int = 13,
        multiplier: float = 2.0,
        ultimate_smoother_period: int = 10,
        zero_lag_period: int = 21,
        filtering_mode: int = 1,
        # シグナル生成パラメータ
        use_filter: bool = False,
        enable_exit_signals: bool = True,
        # UQATRDフィルターパラメータ
        use_uqatrd_filter: bool = True,
        uqatrd_coherence_window: int = 21,
        uqatrd_entanglement_window: int = 34,
        uqatrd_efficiency_window: int = 21,
        uqatrd_uncertainty_window: int = 14,
        uqatrd_src_type: str = 'hlc3',
        uqatrd_adaptive_mode: bool = True,
        uqatrd_sensitivity: float = 1.0,
        uqatrd_str_period: float = 20.0,
        uqatrd_threshold_mode: str = 'dynamic',
        uqatrd_fixed_threshold: float = 0.5,
        uqatrd_min_data_points: int = 50,
        uqatrd_confidence_threshold: float = 0.7,
        # Ultimate MAの動的適応パラメータ
        zero_lag_period_mode: str = 'dynamic',
        # Ultimate MAのゼロラグ用サイクル検出器パラメータ
        zl_cycle_detector_type: str = 'absolute_ultimate',
        zl_cycle_detector_cycle_part: float = 1.0,
        zl_cycle_detector_max_cycle: int = 120,
        zl_cycle_detector_min_cycle: int = 5,
        zl_cycle_period_multiplier: float = 1.0,
        zl_cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        コンストラクタ
        
        Args:
            # Ultimate Trendパラメータ
            length: ATR計算期間（デフォルト: 13）
            multiplier: ATR乗数（デフォルト: 2.0）
            ultimate_smoother_period: スーパースムーザー期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            filtering_mode: フィルタリングモード（デフォルト: 1）
            
            # シグナル生成パラメータ
            use_filter: フィルターオプション（True=Ultimate MAのtrend_signalsでフィルタリング）
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            
            # UQATRDフィルターパラメータ
            use_uqatrd_filter: UQATRDフィルターオプション（True=UQATRDフィルターでフィルタリング）
            uqatrd_coherence_window: UQATRDフィルターのコヒーレンスウィンドウ
            uqatrd_entanglement_window: UQATRDフィルターのエンタングルメントウィンドウ
            uqatrd_efficiency_window: UQATRDフィルターのエフィシエンスウィンドウ
            uqatrd_uncertainty_window: UQATRDフィルターのアンカーンションウィンドウ
            uqatrd_src_type: UQATRDフィルターのソースタイプ
            uqatrd_adaptive_mode: UQATRDフィルターの適応モード
            uqatrd_sensitivity: UQATRDフィルターの感度
            uqatrd_str_period: UQATRDフィルターのストリング期間
            uqatrd_threshold_mode: UQATRDフィルターのしきい値モード
            uqatrd_fixed_threshold: UQATRDフィルターの固定しきい値
            uqatrd_min_data_points: UQATRDフィルターの最小データポイント
            uqatrd_confidence_threshold: UQATRDフィルターの信頼度しきい値
            
            # Ultimate MAの動的適応パラメータ
            zero_lag_period_mode: ゼロラグEMA期間モード（'dynamic' or 'fixed'）
            realtime_window_mode: リアルタイムウィンドウモード（'dynamic' or 'fixed'）
            
            # Ultimate MAのゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル部分
            zl_cycle_detector_max_cycle: ゼロラグ用最大サイクル
            zl_cycle_detector_min_cycle: ゼロラグ用最小サイクル
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間乗数
            zl_cycle_detector_period_range: ゼロラグ用period_rangeパラメータ
        """
        params = {
            'length': length,
            'multiplier': multiplier,
            'ultimate_smoother_period': ultimate_smoother_period,
            'zero_lag_period': zero_lag_period,
            'filtering_mode': filtering_mode,
            'use_filter': use_filter,
            'enable_exit_signals': enable_exit_signals,
            'use_uqatrd_filter': use_uqatrd_filter,
            'uqatrd_coherence_window': uqatrd_coherence_window,
            'uqatrd_entanglement_window': uqatrd_entanglement_window,
            'uqatrd_efficiency_window': uqatrd_efficiency_window,
            'uqatrd_uncertainty_window': uqatrd_uncertainty_window,
            'uqatrd_src_type': uqatrd_src_type,
            'uqatrd_adaptive_mode': uqatrd_adaptive_mode,
            'uqatrd_sensitivity': uqatrd_sensitivity,
            'uqatrd_str_period': uqatrd_str_period,
            'uqatrd_threshold_mode': uqatrd_threshold_mode,
            'uqatrd_fixed_threshold': uqatrd_fixed_threshold,
            'uqatrd_min_data_points': uqatrd_min_data_points,
            'uqatrd_confidence_threshold': uqatrd_confidence_threshold,
            'zero_lag_period_mode': zero_lag_period_mode,
            'zl_cycle_detector_type': zl_cycle_detector_type,
            'zl_cycle_detector_cycle_part': zl_cycle_detector_cycle_part,
            'zl_cycle_detector_max_cycle': zl_cycle_detector_max_cycle,
            'zl_cycle_detector_min_cycle': zl_cycle_detector_min_cycle,
            'zl_cycle_period_multiplier': zl_cycle_period_multiplier,
            'zl_cycle_detector_period_range': zl_cycle_detector_period_range
        }
        
        filter_desc = "Filtered" if use_filter else "Base"
        uqatrd_desc = "+UQATRD" if use_uqatrd_filter else ""
        super().__init__(
            f"UltimateTrendEntry(STR={length}, mult={multiplier}, mode={filtering_mode}, {filter_desc}{uqatrd_desc})",
            params
        )
        
        # Ultimate Trendのインスタンス化
        self._ultimate_trend = UltimateTrend(
            length=length,
            multiplier=multiplier,
            ultimate_smoother_period=ultimate_smoother_period,
            zero_lag_period=zero_lag_period,
            filtering_mode=filtering_mode,
            zero_lag_period_mode=zero_lag_period_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=zl_cycle_period_multiplier,
            zl_cycle_detector_period_range=zl_cycle_detector_period_range
        )
        
        # UQATRDフィルターの初期化（有効な場合のみ）
        self._uqatrd_filter = None
        if use_uqatrd_filter:
            self._uqatrd_filter = UQATRDFilterSignal(
                coherence_window=uqatrd_coherence_window,
                entanglement_window=uqatrd_entanglement_window,
                efficiency_window=uqatrd_efficiency_window,
                uncertainty_window=uqatrd_uncertainty_window,
                src_type=uqatrd_src_type,
                adaptive_mode=uqatrd_adaptive_mode,
                sensitivity=uqatrd_sensitivity,
                str_period=uqatrd_str_period,
                threshold_mode=uqatrd_threshold_mode,
                fixed_threshold=uqatrd_fixed_threshold,
                min_data_points=uqatrd_min_data_points,
                confidence_threshold=uqatrd_confidence_threshold
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
            
            # Ultimate Trendの計算
            result = self._ultimate_trend.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # トレンド方向とトレンドシグナルの取得
            trend = result.trend.astype(np.int8)
            trend_signals = result.trend_signals.astype(np.int8) if self._params['use_filter'] else None
            
            # UQATRDフィルターシグナルの取得（有効な場合）
            filter_signals = None
            if self._params['use_uqatrd_filter'] and self._uqatrd_filter is not None:
                filter_signals = self._uqatrd_filter.generate(data).astype(np.int8)
            
            # エントリーシグナルの計算（高速化版）
            entry_signals = calculate_ultimate_trend_signals(
                trend,
                trend_signals,
                filter_signals,
                self._params['use_filter'],
                self._params['use_uqatrd_filter']
            )
            
            # 結果をキャッシュ
            self._entry_signals = entry_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                # 簡易的なポジション追跡
                current_position = self._track_position(entry_signals)
                self._exit_signals = calculate_ultimate_trend_exit_signals(
                    trend,
                    trend_signals,
                    filter_signals,
                    current_position,
                    self._params['use_filter'],
                    self._params['use_uqatrd_filter']
                )
                self._current_position = current_position
            
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltimateTrendEntrySignal計算中にエラー: {str(e)}")
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
    
    def get_ultimate_trend_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Ultimate Trendの計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UltimateTrendResult: Ultimate Trendの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._ultimate_trend.calculate(data)
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルティメットトレンド方向を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=なし）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_trend.get_trend()
        return result if result is not None else np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate MAのトレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_trend.get_trend_signals()
        return result if result is not None else np.array([])
    
    def get_ultimate_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルティメットトレンドラインの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルティメットトレンドラインの値
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_trend.get_values()
        return result if result is not None else np.array([])
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        上側バンドを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 上側バンドの配列
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_trend.get_upper_band()
        return result if result is not None else np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        下側バンドを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 下側バンドの配列
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_trend.get_lower_band()
        return result if result is not None else np.array([])
    
    def get_filtering_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        フィルタリング統計を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: フィルタリング統計
        """
        if data is not None:
            self.generate(data)
        
        return self._ultimate_trend.get_filtering_stats()
    
    def get_uqatrd_filter_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRDフィルターシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRDフィルターシグナルの配列（1=トレンド相場、-1=レンジ相場）
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.generate(data)
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_uqatrd_trend_range_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRDトレンド/レンジ信号値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRDトレンド/レンジ信号値の配列
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.get_trend_range_values()
        else:
            return np.array([])
    
    def get_uqatrd_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRDしきい値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRDしきい値の配列
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.get_threshold_values()
        else:
            return np.array([])
    
    def get_uqatrd_signal_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRD信号強度を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRD信号強度の配列
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.get_signal_strength()
        else:
            return np.array([])
    
    def get_uqatrd_confidence_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        UQATRD信頼度スコアを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: UQATRD信頼度スコアの配列
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.get_confidence_score()
        else:
            return np.array([])
    
    def get_uqatrd_threshold_info(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        UQATRDしきい値の統計情報を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: UQATRDしきい値の統計情報
        """
        if data is not None:
            self.generate(data)
        
        if self._uqatrd_filter is not None:
            return self._uqatrd_filter.get_threshold_info()
        else:
            return {}
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._ultimate_trend.reset() if hasattr(self._ultimate_trend, 'reset') else None
        if self._uqatrd_filter is not None:
            self._uqatrd_filter.reset()
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None 