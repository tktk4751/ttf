#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_v_channel import ZVChannel


@njit(fastmath=True, parallel=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
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
        if close[i] > upper[i-lookback]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback]:
            signals[i] = -1
    
    return signals


class ZVChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ZVチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - ボリンジャーバンドとZチャネルの特性を組み合わせた高度なアダプティブチャネル
    - サイクル効率比（CER）に基づく動的な適応性
    - 平滑化アルゴリズム（ALMAまたはハイパースムーサー）を使用したZATR
    - ボラティリティと価格変動に応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # ボリンジャーバンドパラメータ
        bb_max_multiplier: float = 2.5,
        bb_min_multiplier: float = 1.0,
        
        # ZBBの標準偏差計算用パラメータ
        bb_max_cycle_part: float = 0.5,    # 標準偏差最大期間用サイクル部分
        bb_max_max_cycle: int = 144,       # 標準偏差最大期間用最大サイクル
        bb_max_min_cycle: int = 10,        # 標準偏差最大期間用最小サイクル
        bb_max_max_output: int = 89,       # 標準偏差最大期間用最大出力値
        bb_max_min_output: int = 13,       # 標準偏差最大期間用最小出力値
        bb_min_cycle_part: float = 0.25,   # 標準偏差最小期間用サイクル部分
        bb_min_max_cycle: int = 55,        # 標準偏差最小期間用最大サイクル
        bb_min_min_cycle: int = 5,         # 標準偏差最小期間用最小サイクル
        bb_min_max_output: int = 21,       # 標準偏差最小期間用最大出力値
        bb_min_min_output: int = 5,        # 標準偏差最小期間用最小出力値
        
        # Zチャネルパラメータ
        kc_max_multiplier: float = 3.0,
        kc_min_multiplier: float = 1.5,
        kc_smoother_type: str = 'alma',
        
        # ZChannel ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 144,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 89,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        zma_max_slow_period: int = 34,          # ZMA: 遅い移動平均の最大期間
        zma_min_slow_period: int = 9,           # ZMA: 遅い移動平均の最小期間
        zma_max_fast_period: int = 8,           # ZMA: 速い移動平均の最大期間
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間（0=平滑化しない）
        
        # ZChannel ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.5,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 55,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 55,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.25,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3,        # ZATR: 最小期間用ドミナントサイクル計算用
        
        # 共通パラメータ
        src_type: str = 'hlc3',
        lookback: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            
            # ボリンジャーバンドパラメータ
            bb_max_multiplier: ボリンジャーバンド標準偏差乗数の最大値（デフォルト: 2.5）
            bb_min_multiplier: ボリンジャーバンド標準偏差乗数の最小値（デフォルト: 1.0）
            
            # ZBBの標準偏差計算用パラメータ
            bb_max_cycle_part: 標準偏差最大期間用サイクル部分
            bb_max_max_cycle: 標準偏差最大期間用最大サイクル
            bb_max_min_cycle: 標準偏差最大期間用最小サイクル
            bb_max_max_output: 標準偏差最大期間用最大出力値
            bb_max_min_output: 標準偏差最大期間用最小出力値
            bb_min_cycle_part: 標準偏差最小期間用サイクル部分
            bb_min_max_cycle: 標準偏差最小期間用最大サイクル
            bb_min_min_cycle: 標準偏差最小期間用最小サイクル
            bb_min_max_output: 標準偏差最小期間用最大出力値
            bb_min_min_output: 標準偏差最小期間用最小出力値
            
            # Zチャネルパラメータ
            kc_max_multiplier: ZチャネルATR乗数の最大値（デフォルト: 3.0）
            kc_min_multiplier: ZチャネルATR乗数の最小値（デフォルト: 1.5）
            kc_smoother_type: Zチャネル平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
                
            # ZChannel ZMA用パラメータ
            zma_max_dc_cycle_part: ZMA最大期間用ドミナントサイクル計算用のサイクル部分
            zma_max_dc_max_cycle: ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間
            zma_max_dc_min_cycle: ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間
            zma_max_dc_max_output: ZMA最大期間用ドミナントサイクル計算用の最大出力値
            zma_max_dc_min_output: ZMA最大期間用ドミナントサイクル計算用の最小出力値
            
            zma_min_dc_cycle_part: ZMA最小期間用ドミナントサイクル計算用のサイクル部分
            zma_min_dc_max_cycle: ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間
            zma_min_dc_min_cycle: ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間
            zma_min_dc_max_output: ZMA最小期間用ドミナントサイクル計算用の最大出力値
            zma_min_dc_min_output: ZMA最小期間用ドミナントサイクル計算用の最小出力値
            
            zma_max_slow_period: ZMA遅い移動平均の最大期間
            zma_min_slow_period: ZMA遅い移動平均の最小期間
            zma_max_fast_period: ZMA速い移動平均の最大期間
            zma_min_fast_period: ZMA速い移動平均の最小期間
            zma_hyper_smooth_period: ZMAハイパースムーサーの平滑化期間（0=平滑化しない）
            
            # ZChannel ZATR用パラメータ
            zatr_max_dc_cycle_part: ZATR最大期間用ドミナントサイクル計算用のサイクル部分
            zatr_max_dc_max_cycle: ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間
            zatr_max_dc_min_cycle: ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間
            zatr_max_dc_max_output: ZATR最大期間用ドミナントサイクル計算用の最大出力値
            zatr_max_dc_min_output: ZATR最大期間用ドミナントサイクル計算用の最小出力値
            
            zatr_min_dc_cycle_part: ZATR最小期間用ドミナントサイクル計算用のサイクル部分
            zatr_min_dc_max_cycle: ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間
            zatr_min_dc_min_cycle: ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間
            zatr_min_dc_max_output: ZATR最小期間用ドミナントサイクル計算用の最大出力値
            zatr_min_dc_min_output: ZATR最小期間用ドミナントサイクル計算用の最小出力値
            
            src_type: ソースタイプ （デフォルト: 'hlc3'）
                'close' - 終値のみ使用
                'hlc3' - (高値+安値+終値)/3
                'hl2' - (高値+安値)/2
                'ohlc4' - (始値+高値+安値+終値)/4
            lookback: 過去のバンドを参照する期間（デフォルト: 1）
        """
        # すべてのパラメータを辞書に格納
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            
            'bb_max_multiplier': bb_max_multiplier,
            'bb_min_multiplier': bb_min_multiplier,
            
            'bb_max_cycle_part': bb_max_cycle_part,
            'bb_max_max_cycle': bb_max_max_cycle,
            'bb_max_min_cycle': bb_max_min_cycle,
            'bb_max_max_output': bb_max_max_output,
            'bb_max_min_output': bb_max_min_output,
            'bb_min_cycle_part': bb_min_cycle_part,
            'bb_min_max_cycle': bb_min_max_cycle,
            'bb_min_min_cycle': bb_min_min_cycle,
            'bb_min_max_output': bb_min_max_output,
            'bb_min_min_output': bb_min_min_output,
            
            'kc_max_multiplier': kc_max_multiplier,
            'kc_min_multiplier': kc_min_multiplier,
            'kc_smoother_type': kc_smoother_type,
            
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_max_slow_period': zma_max_slow_period,
            'zma_min_slow_period': zma_min_slow_period,
            'zma_max_fast_period': zma_max_fast_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
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
            
            'src_type': src_type,
            'lookback': lookback
        }
        
        super().__init__(
            f"ZVChannelBreakout({cycle_detector_type}, BB:{bb_max_multiplier}-{bb_min_multiplier}, KC:{kc_max_multiplier}-{kc_min_multiplier}, {kc_smoother_type}, {lookback})",
            params
        )
        
        # ZVチャネルのインスタンス化
        self._z_v_channel = ZVChannel(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            bb_max_multiplier=bb_max_multiplier,
            bb_min_multiplier=bb_min_multiplier,
            
            bb_max_cycle_part=bb_max_cycle_part,
            bb_max_max_cycle=bb_max_max_cycle,
            bb_max_min_cycle=bb_max_min_cycle,
            bb_max_max_output=bb_max_max_output,
            bb_max_min_output=bb_max_min_output,
            bb_min_cycle_part=bb_min_cycle_part,
            bb_min_max_cycle=bb_min_max_cycle,
            bb_min_min_cycle=bb_min_min_cycle,
            bb_min_max_output=bb_min_max_output,
            bb_min_min_output=bb_min_min_output,
            
            kc_max_multiplier=kc_max_multiplier,
            kc_min_multiplier=kc_min_multiplier,
            kc_smoother_type=kc_smoother_type,
            
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            zma_max_slow_period=zma_max_slow_period,
            zma_min_slow_period=zma_min_slow_period,
            zma_max_fast_period=zma_max_fast_period,
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
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
            
            # ZVチャネルの計算
            result = self._z_v_channel.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                return self._signals
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドの取得
            _, upper, lower = self._z_v_channel.get_bands()
            
            # ブレイクアウトシグナルの計算（高速化版）
            lookback = self._params['lookback']
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                lookback
            )
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"ZVChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            self._signals = np.zeros(len(data), dtype=np.int8)
            return self._signals
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZVチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_v_channel.get_bands()
    
    def get_bb_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部ボリンジャーバンドの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_v_channel.get_bb_bands()
    
    def get_kc_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部Zチャネル（ケルトナーチャネル）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._z_v_channel.get_kc_bands()
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.generate(data)
            
        return self._z_v_channel.get_cycle_er()
    
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._z_v_channel.reset() if hasattr(self._z_v_channel, 'reset') else None
        self._signals = None
        self._data_hash = None 