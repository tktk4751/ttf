#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.ultra_quantum_adaptive_trend_range_discriminator import UltraQuantumAdaptiveTrendRangeDiscriminator


@njit(fastmath=True, parallel=True)
def generate_signals_numba(
    trend_range_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        trend_range_values: UQATRD トレンド/レンジ信号値の配列 (0-1)
        threshold_values: しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(trend_range_values)
    signals = np.ones(length)  # デフォルトはトレンド相場 (1)
    
    for i in prange(length):
        if np.isnan(trend_range_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif trend_range_values[i] <= threshold_values[i]:
            signals[i] = -1  # レンジ相場
        else:
            signals[i] = 1   # トレンド相場
    
    return signals


class UQATRDFilterSignal(BaseSignal, IFilterSignal):
    """
    UQATRD (Ultra Quantum Adaptive Trend-Range Discriminator) を使用したフィルターシグナル
    
    特徴:
    - 量子アルゴリズムによる超高精度なトレンド/レンジ判別
    - 動的適応しきい値または固定しきい値を選択可能
    - 4つの核心量子アルゴリズムによる多次元解析
    - 複素数平面での高精度計算
    
    動作:
    - トレンド/レンジ信号がしきい値以上：トレンド相場 (1)
    - トレンド/レンジ信号がしきい値以下：レンジ相場 (-1)
    
    使用方法:
    - トレンド系/レンジ系ストラテジーの自動切り替え
    - 量子的アプローチによる市場状態の高精度識別
    - 動的適応による市場変動への自動調整
    """
    
    def __init__(
        self,
        # UQATRDの各量子アルゴリズムパラメータ
        coherence_window: int = 21,         # 量子コヒーレンス分析窓
        entanglement_window: int = 34,      # 量子エンタングルメント分析窓
        efficiency_window: int = 21,        # 量子効率スペクトラム分析窓
        uncertainty_window: int = 14,       # 量子不確定性分析窓
        
        # 一般パラメータ
        src_type: str = 'ukf_hlc3',         # 価格ソース
        adaptive_mode: bool = True,         # 適応モード
        sensitivity: float = 1.0,           # 感度調整
        
        # STRパラメータ
        str_period: float = 20.0,           # STR期間（ボラティリティ計算用）
        
        # しきい値パラメータ
        threshold_mode: str = 'dynamic',    # しきい値モード ('dynamic' または 'fixed')
        fixed_threshold: float = 0.5,      # 固定しきい値
        
        # 品質管理パラメータ
        min_data_points: int = 50,          # 最小データポイント数
        confidence_threshold: float = 0.7   # 信頼度閾値
    ):
        """
        コンストラクタ
        
        Args:
            coherence_window: 量子コヒーレンス分析ウィンドウ
            entanglement_window: 量子エンタングルメント分析ウィンドウ
            efficiency_window: 量子効率スペクトラム分析ウィンドウ
            uncertainty_window: 量子不確定性分析ウィンドウ
            src_type: 価格ソースタイプ
            adaptive_mode: 適応モード
            sensitivity: 感度調整倍率
            str_period: STR期間（ボラティリティ計算用）
            threshold_mode: しきい値モード
                'dynamic': 動的適応しきい値を使用
                'fixed': 固定しきい値を使用
            fixed_threshold: 固定しきい値（threshold_mode='fixed'時に使用）
            min_data_points: 最小データポイント数
            confidence_threshold: 信頼度閾値
        """
        # パラメータの設定
        params = {
            'coherence_window': coherence_window,
            'entanglement_window': entanglement_window,
            'efficiency_window': efficiency_window,
            'uncertainty_window': uncertainty_window,
            'src_type': src_type,
            'adaptive_mode': adaptive_mode,
            'sensitivity': sensitivity,
            'str_period': str_period,
            'threshold_mode': threshold_mode,
            'fixed_threshold': fixed_threshold,
            'min_data_points': min_data_points,
            'confidence_threshold': confidence_threshold
        }
        
        super().__init__(
            f"UQATRDFilter(C:{coherence_window},E:{entanglement_window},"
            f"Ef:{efficiency_window},U:{uncertainty_window},{threshold_mode})",
            params
        )
        
        # パラメータの保存
        self.threshold_mode = threshold_mode.lower()
        self.fixed_threshold = fixed_threshold
        
        # パラメータ検証
        if self.threshold_mode not in ['dynamic', 'fixed']:
            raise ValueError(f"無効なthreshold_mode: {threshold_mode}. 'dynamic' または 'fixed' を指定してください。")
        
        if not (0.0 <= self.fixed_threshold <= 1.0):
            raise ValueError(f"fixed_thresholdは0.0から1.0の間である必要があります: {fixed_threshold}")
        
        # UQATRDインジケーターの初期化
        self._uqatrd = UltraQuantumAdaptiveTrendRangeDiscriminator(
            coherence_window=coherence_window,
            entanglement_window=entanglement_window,
            efficiency_window=efficiency_window,
            uncertainty_window=uncertainty_window,
            src_type=src_type,
            adaptive_mode=adaptive_mode,
            sensitivity=sensitivity,
            str_period=str_period,
            min_data_points=min_data_points,
            confidence_threshold=confidence_threshold
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None
        self._uqatrd_result = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は必要なカラムのみハッシュする
                cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in cols if col in data.columns]
                if available_cols:
                    data_hash = hash(tuple(map(tuple, (data[col].values for col in available_cols))))
                else:
                    data_hash = hash(tuple(map(tuple, data.values)))
            else:
                # NumPy配列の場合は全体をハッシュする
                data_hash = hash(tuple(map(tuple, data)))
            
            # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
            param_str = f"{hash(frozenset(self._params.items()))}"
            
            return f"{data_hash}_{param_str}"
            
        except Exception:
            # フォールバック: オブジェクトIDを使用
            return f"{id(data)}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # データの検証
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    raise ValueError("入力データが空です")
                # 必要なカラムがあるかチェック（UQATRDが内部で適切に処理）
                required_cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in required_cols if col in data.columns]
                if not available_cols:
                    raise ValueError("DataFrameには少なくとも'open', 'high', 'low', 'close'の一つが必要です")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                if len(data) == 0:
                    raise ValueError("入力データが空です")
            
            # UQATRDの計算
            uqatrd_result = self._uqatrd.calculate(data)
            
            # 計算が失敗した場合はNaNシグナルを返す
            if uqatrd_result is None or len(uqatrd_result.trend_range_signal) == 0:
                self._signals = np.full(len(data), np.nan)
                return self._signals
            
            # 結果を保存
            self._uqatrd_result = uqatrd_result
            
            # トレンド/レンジ信号の取得
            trend_range_values = uqatrd_result.trend_range_signal
            
            # しきい値の決定
            if self.threshold_mode == 'dynamic':
                # 動的適応しきい値を使用
                threshold_values = uqatrd_result.adaptive_threshold
            else:
                # 固定しきい値を使用
                threshold_values = np.full(len(trend_range_values), self.fixed_threshold)
            
            # しきい値配列の長さを信号配列に合わせる
            if len(threshold_values) != len(trend_range_values):
                if len(threshold_values) > 0:
                    # しきい値配列を拡張または切り詰め
                    if len(threshold_values) < len(trend_range_values):
                        # 不足分を最後の値で埋める
                        last_threshold = threshold_values[-1]
                        threshold_values = np.concatenate([
                            threshold_values, 
                            np.full(len(trend_range_values) - len(threshold_values), last_threshold)
                        ])
                    else:
                        # 余分な部分を切り詰める
                        threshold_values = threshold_values[:len(trend_range_values)]
                else:
                    # しきい値配列が空の場合は固定値を使用
                    threshold_values = np.full(len(trend_range_values), self.fixed_threshold)
            
            # シグナルの生成（高速化版）
            signals = generate_signals_numba(trend_range_values, threshold_values)
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"UQATRDFilterSignal計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はNaNシグナルを返す
            data_length = len(data) if hasattr(data, '__len__') else 0
            self._signals = np.full(data_length, np.nan)
            return self._signals
    
    def get_trend_range_values(self) -> np.ndarray:
        """
        UQATRDトレンド/レンジ信号値を取得する
        
        Returns:
            トレンド/レンジ信号値の配列 (0-1の範囲)
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.trend_range_signal.copy()
        return np.array([])
    
    def get_threshold_values(self) -> np.ndarray:
        """
        使用されたしきい値を取得する
        
        Returns:
            しきい値の配列
        """
        if self._uqatrd_result is not None:
            if self.threshold_mode == 'dynamic':
                return self._uqatrd_result.adaptive_threshold.copy()
            else:
                return np.full(len(self._uqatrd_result.trend_range_signal), self.fixed_threshold)
        return np.array([])
    
    def get_signal_strength(self) -> np.ndarray:
        """
        信号強度を取得する
        
        Returns:
            信号強度の配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.signal_strength.copy()
        return np.array([])
    
    def get_confidence_score(self) -> np.ndarray:
        """
        信頼度スコアを取得する
        
        Returns:
            信頼度スコアの配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.confidence_score.copy()
        return np.array([])
    
    def get_quantum_coherence(self) -> np.ndarray:
        """
        量子コヒーレンス値を取得する
        
        Returns:
            量子コヒーレンス値の配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.quantum_coherence.copy()
        return np.array([])
    
    def get_trend_persistence(self) -> np.ndarray:
        """
        トレンド持続性を取得する
        
        Returns:
            トレンド持続性の配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.trend_persistence.copy()
        return np.array([])
    
    def get_efficiency_spectrum(self) -> np.ndarray:
        """
        効率スペクトラムを取得する
        
        Returns:
            効率スペクトラムの配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.efficiency_spectrum.copy()
        return np.array([])
    
    def get_uncertainty_range(self) -> np.ndarray:
        """
        不確定性レンジを取得する
        
        Returns:
            不確定性レンジの配列
        """
        if self._uqatrd_result is not None:
            return self._uqatrd_result.uncertainty_range.copy()
        return np.array([])
    
    def get_algorithm_breakdown(self) -> Optional[Dict[str, np.ndarray]]:
        """
        各量子アルゴリズムの詳細結果を取得する
        
        Returns:
            各アルゴリズムの結果を含む辞書
        """
        if self._uqatrd_result is not None:
            return {
                'quantum_coherence': self._uqatrd_result.quantum_coherence.copy(),
                'trend_persistence': self._uqatrd_result.trend_persistence.copy(),
                'efficiency_spectrum': self._uqatrd_result.efficiency_spectrum.copy(),
                'uncertainty_range': self._uqatrd_result.uncertainty_range.copy(),
                'signal_strength': self._uqatrd_result.signal_strength.copy(),
                'confidence_score': self._uqatrd_result.confidence_score.copy()
            }
        return None
    
    def get_threshold_info(self) -> Optional[Dict[str, Any]]:
        """
        しきい値の統計情報を取得する
        
        Returns:
            しきい値の統計情報を含む辞書
        """
        if self._uqatrd_result is not None:
            if self.threshold_mode == 'dynamic':
                threshold_values = self._uqatrd_result.adaptive_threshold
                return {
                    'threshold_mode': self.threshold_mode,
                    'mean_threshold': float(np.mean(threshold_values)),
                    'std_threshold': float(np.std(threshold_values)),
                    'min_threshold': float(np.min(threshold_values)),
                    'max_threshold': float(np.max(threshold_values)),
                    'median_threshold': float(np.median(threshold_values)),
                    'current_threshold': float(threshold_values[-1]) if len(threshold_values) > 0 else None
                }
            else:
                return {
                    'threshold_mode': self.threshold_mode,
                    'fixed_threshold': self.fixed_threshold,
                    'mean_threshold': self.fixed_threshold,
                    'std_threshold': 0.0,
                    'min_threshold': self.fixed_threshold,
                    'max_threshold': self.fixed_threshold,
                    'median_threshold': self.fixed_threshold,
                    'current_threshold': self.fixed_threshold
                }
        return None
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._uqatrd, 'reset'):
            self._uqatrd.reset()
        self._signals = None
        self._data_hash = None
        self._uqatrd_result = None 