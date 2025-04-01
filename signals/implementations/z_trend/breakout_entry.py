#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_trend import ZTrend


@njit(fastmath=True, parallel=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, trend: np.ndarray, lookback: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        trend: トレンド方向の配列（1=上昇トレンド、-1=下降トレンド）
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定（並列処理化）
    for i in prange(lookback, length):
        # 終値とバンドの値が有効かチェック
        if np.isnan(close[i]) or np.isnan(upper[i-lookback]) or np.isnan(lower[i-lookback]):
            signals[i] = 0
            continue
            
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-lookback] and trend[i] == 1:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback] and trend[i] == -1:
            signals[i] = -1
    
    return signals


class ZTrendBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    Zトレンドのブレイクアウトによるエントリーシグナル
    
    特徴:
    - サイクル効率比（CER）に基づく動的な適応性
    - 平滑化されたパーセンタイルバンド
    - トレンド方向による確認済みシグナル
    - ボラティリティに応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 上昇トレンド時に現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 下降トレンド時に現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'dudi_dc',
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.5,
        lookback: int = 1,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 233,
        cer_min_cycle: int = 13,
        cer_max_output: int = 144,
        cer_min_output: int = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        # 動的乗数の範囲
        max_max_multiplier: float = 5.0,    # 最大乗数の最大値
        min_max_multiplier: float = 2.5,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'dudi_dc'）
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 55）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            lookback: 過去のバンドを参照する期間（デフォルト: 1）
            
            cer_max_cycle: CER用の最大サイクル（デフォルト: 233）
            cer_min_cycle: CER用の最小サイクル（デフォルト: 13）
            cer_max_output: CER用の最大出力値（デフォルト: 144）
            cer_min_output: CER用の最小出力値（デフォルト: 21）
            
            max_percentile_dc_cycle_part: 最大パーセンタイル用サイクル部分（デフォルト: 0.5）
            max_percentile_dc_max_cycle: 最大パーセンタイル用最大サイクル（デフォルト: 233）
            max_percentile_dc_min_cycle: 最大パーセンタイル用最小サイクル（デフォルト: 13）
            max_percentile_dc_max_output: 最大パーセンタイル用最大出力値（デフォルト: 144）
            max_percentile_dc_min_output: 最大パーセンタイル用最小出力値（デフォルト: 21）
            
            min_percentile_dc_cycle_part: 最小パーセンタイル用サイクル部分（デフォルト: 0.5）
            min_percentile_dc_max_cycle: 最小パーセンタイル用最大サイクル（デフォルト: 55）
            min_percentile_dc_min_cycle: 最小パーセンタイル用最小サイクル（デフォルト: 5）
            min_percentile_dc_max_output: 最小パーセンタイル用最大出力値（デフォルト: 34）
            min_percentile_dc_min_output: 最小パーセンタイル用最小出力値（デフォルト: 8）
            
            zatr_max_dc_cycle_part: ZATR用最大サイクル部分（デフォルト: 0.5）
            zatr_max_dc_max_cycle: ZATR用最大サイクル（デフォルト: 55）
            zatr_max_dc_min_cycle: ZATR用最小サイクル（デフォルト: 5）
            zatr_max_dc_max_output: ZATR用最大出力値（デフォルト: 55）
            zatr_max_dc_min_output: ZATR用最小出力値（デフォルト: 5）
            zatr_min_dc_cycle_part: ZATR用最小サイクル部分（デフォルト: 0.25）
            zatr_min_dc_max_cycle: ZATR用最小サイクル（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR用最小サイクル（デフォルト: 3）
            zatr_min_dc_max_output: ZATR用最小出力値（デフォルト: 13）
            zatr_min_dc_min_output: ZATR用最小出力値（デフォルト: 3）
            
            max_percentile_cycle_mult: 最大パーセンタイル期間のサイクル乗数（デフォルト: 0.5）
            min_percentile_cycle_mult: 最小パーセンタイル期間のサイクル乗数（デフォルト: 0.25）
            
            max_max_multiplier: 最大乗数の最大値（デフォルト: 5.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 2.5）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
            src_type: ソースタイプ （デフォルト: 'hlc3'）
                'close' - 終値のみ使用
                'hlc3' - (高値+安値+終値)/3
                'hl2' - (高値+安値)/2
                'ohlc4' - (始値+高値+安値+終値)/4
        """
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'lookback': lookback,
            
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            
            'max_percentile_dc_cycle_part': max_percentile_dc_cycle_part,
            'max_percentile_dc_max_cycle': max_percentile_dc_max_cycle,
            'max_percentile_dc_min_cycle': max_percentile_dc_min_cycle,
            'max_percentile_dc_max_output': max_percentile_dc_max_output,
            'max_percentile_dc_min_output': max_percentile_dc_min_output,
            
            'min_percentile_dc_cycle_part': min_percentile_dc_cycle_part,
            'min_percentile_dc_max_cycle': min_percentile_dc_max_cycle,
            'min_percentile_dc_min_cycle': min_percentile_dc_min_cycle,
            'min_percentile_dc_max_output': min_percentile_dc_max_output,
            'min_percentile_dc_min_output': min_percentile_dc_min_output,
            
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            
            'max_percentile_cycle_mult': max_percentile_cycle_mult,
            'min_percentile_cycle_mult': min_percentile_cycle_mult,
            
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        super().__init__(
            f"ZTrendBreakout({cycle_detector_type}, {lookback})",
            params
        )
        
        # Zトレンドのインスタンス化
        self._z_trend = ZTrend(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            # CERのドミナントサイクル検出器用パラメータ
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            
            # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            
            # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            
            # ZATR用ドミナントサイクル検出器のパラメータ
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            
            # パーセンタイル乗数
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            
            
            # 動的乗数の範囲
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # その他の設定
            smoother_type=smoother_type,
            src_type=src_type
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # closeカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合はcloseだけハッシュ
                data_hash = hash(tuple(data[:, 3]))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
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
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # Zトレンドの計算
            result = self._z_trend.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                return self._signals
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドとトレンド方向の取得
            upper, lower = self._z_trend.get_bands()
            trend = self._z_trend.get_trend()
            
            # ブレイクアウトシグナルの計算（高速化版）
            lookback = self._params['lookback']
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                trend,
                lookback
            )
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"ZTrendBreakoutEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._signals = np.zeros(len(data), dtype=np.int8)
            return self._signals
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zトレンドのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_bands()
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇トレンド、-1=下降トレンド）
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_trend()
    
    def get_percentiles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        平滑化されたパーセンタイル値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上側パーセンタイル, 下側パーセンタイル)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_percentiles()
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_cycle_er()
    
    def get_dynamic_parameters(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (動的乗数, 動的パーセンタイル期間)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_dynamic_parameters()
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZATRの値
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_z_atr()
    
    def get_dominant_cycles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ドミナントサイクルの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最大パーセンタイル期間用ドミナントサイクル, 最小パーセンタイル期間用ドミナントサイクル)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_trend.get_dominant_cycles()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._z_trend.reset() if hasattr(self._z_trend, 'reset') else None
        self._signals = None
        self._data_hash = None 