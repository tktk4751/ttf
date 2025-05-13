#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_channel import ZChannel


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
    for i in prange(lookback + 1, length):
        # 終値とバンドの値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 前回の終値が前回のアッパーバンドを上回っていないかつ現在の終値が現在のアッパーバンドを上回る
        if close[i-1] <= upper[i-1] and close[i] > upper[i]:
            signals[i] = 1
        # ショートエントリー: 前回の終値が前回のロワーバンドを下回っていないかつ現在の終値が現在のロワーバンドを下回る
        elif close[i-1] >= lower[i-1] and close[i] < lower[i]:
            signals[i] = -1
    
    return signals


class ZChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    Zチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - サイクル効率比（CER）に基づく動的な適応性
    - 平滑化アルゴリズム（ALMAまたはハイパースムーサー）を使用したZATR
    - ボラティリティに応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # Zチャネル共通パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # CER用パラメータ
        cer_max_cycle: int = 144,       # CER用の最大サイクル期間
        cer_min_cycle: int = 5,         # CER用の最小サイクル期間
        cer_max_output: int = 89,       # CER用の最大出力値
        cer_min_output: int = 5,        # CER用の最小出力値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,

        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.7,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 77,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 35,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.5,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3         # ZATR: 最小期間用ドミナントサイクル計算用
    ):
        """
        初期化
        
        Args:
            detector_type: 検出器タイプ（ZMAとZATRに使用）
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積（デフォルト）
                - 'dft': 離散フーリエ変換
            cer_detector_type: CER用の検出器タイプ（指定しない場合はdetector_typeと同じ）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 55）
            cycle_part: サイクル部分の倍率（デフォルト: 0.7）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 6.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # CER用パラメータ
            cer_max_cycle: CER用の最大サイクル期間（デフォルト: 144）
            cer_min_cycle: CER用の最小サイクル期間（デフォルト: 5）
            cer_max_output: CER用の最大出力値（デフォルト: 89）
            cer_min_output: CER用の最小出力値（デフォルト: 5）
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part: ZMA最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_max_dc_max_cycle: ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 100）
            zma_max_dc_min_cycle: ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_max_dc_max_output: ZMA最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 120）
            zma_max_dc_min_output: ZMA最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            zma_min_dc_cycle_part: ZMA最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            zma_min_dc_max_cycle: ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            zma_min_dc_min_cycle: ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_min_dc_max_output: ZMA最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zma_min_dc_min_output: ZMA最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part: ZMA動的Slow最大用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_slow_max_dc_max_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            zma_slow_max_dc_min_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_slow_max_dc_max_output: ZMA動的Slow最大用ドミナントサイクル計算用の最大出力値（デフォルト: 89）
            zma_slow_max_dc_min_output: ZMA動的Slow最大用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part: ZMA動的Slow最小用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_slow_min_dc_max_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 89）
            zma_slow_min_dc_min_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_slow_min_dc_max_output: ZMA動的Slow最小用ドミナントサイクル計算用の最大出力値（デフォルト: 21）
            zma_slow_min_dc_min_output: ZMA動的Slow最小用ドミナントサイクル計算用の最小出力値（デフォルト: 8）
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part: ZMA動的Fast最大用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_fast_max_dc_max_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            zma_fast_max_dc_min_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_fast_max_dc_max_output: ZMA動的Fast最大用ドミナントサイクル計算用の最大出力値（デフォルト: 15）
            zma_fast_max_dc_min_output: ZMA動的Fast最大用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            zma_min_fast_period: ZMA速い移動平均の最小期間（デフォルト: 2）
            zma_hyper_smooth_period: ZMAハイパースムーサーの平滑化期間（デフォルト: 0）
            
            # ZATR用パラメータ
            zatr_max_dc_cycle_part: ZATR最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.7）
            zatr_max_dc_max_cycle: ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 77）
            zatr_max_dc_min_cycle: ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zatr_max_dc_max_output: ZATR最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 35）
            zatr_max_dc_min_output: ZATR最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            
            zatr_min_dc_cycle_part: ZATR最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zatr_min_dc_max_cycle: ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 3）
            zatr_min_dc_max_output: ZATR最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zatr_min_dc_min_output: ZATR最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
        """
        super().__init__(
            f"ZChannelBreakoutEntrySignal({detector_type}, {max_max_multiplier}, {min_min_multiplier}, {band_lookback})"
        )
        
        # CERの検出器タイプが指定されていない場合、detector_typeと同じ値を使用
        if cer_detector_type is None:
            cer_detector_type = detector_type
            
        # パラメータの保存
        self._params = {
            # Zチャネル共通パラメータ
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # CER用パラメータ
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            
            # ZMA用パラメータ
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
            'zma_slow_max_dc_cycle_part': zma_slow_max_dc_cycle_part,
            'zma_slow_max_dc_max_cycle': zma_slow_max_dc_max_cycle,
            'zma_slow_max_dc_min_cycle': zma_slow_max_dc_min_cycle,
            'zma_slow_max_dc_max_output': zma_slow_max_dc_max_output,
            'zma_slow_max_dc_min_output': zma_slow_max_dc_min_output,
            'zma_slow_min_dc_cycle_part': zma_slow_min_dc_cycle_part,
            'zma_slow_min_dc_max_cycle': zma_slow_min_dc_max_cycle,
            'zma_slow_min_dc_min_cycle': zma_slow_min_dc_min_cycle,
            'zma_slow_min_dc_max_output': zma_slow_min_dc_max_output,
            'zma_slow_min_dc_min_output': zma_slow_min_dc_min_output,
            'zma_fast_max_dc_cycle_part': zma_fast_max_dc_cycle_part,
            'zma_fast_max_dc_max_cycle': zma_fast_max_dc_max_cycle,
            'zma_fast_max_dc_min_cycle': zma_fast_max_dc_min_cycle,
            'zma_fast_max_dc_max_output': zma_fast_max_dc_max_output,
            'zma_fast_max_dc_min_output': zma_fast_max_dc_min_output,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # ZATR用パラメータ
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output
        }
            
        # ZチャネルZMAの初期化
        self.z_channel = ZChannel(
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            # 動的乗数の範囲パラメータ
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # CER用パラメータ
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            
            # ZMA用パラメータ - すべて
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
            
            # 動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part=zma_slow_max_dc_cycle_part,
            zma_slow_max_dc_max_cycle=zma_slow_max_dc_max_cycle,
            zma_slow_max_dc_min_cycle=zma_slow_max_dc_min_cycle,
            zma_slow_max_dc_max_output=zma_slow_max_dc_max_output,
            zma_slow_max_dc_min_output=zma_slow_max_dc_min_output,
            
            # 動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part=zma_slow_min_dc_cycle_part,
            zma_slow_min_dc_max_cycle=zma_slow_min_dc_max_cycle,
            zma_slow_min_dc_min_cycle=zma_slow_min_dc_min_cycle,
            zma_slow_min_dc_max_output=zma_slow_min_dc_max_output,
            zma_slow_min_dc_min_output=zma_slow_min_dc_min_output,
            
            # 動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part=zma_fast_max_dc_cycle_part,
            zma_fast_max_dc_max_cycle=zma_fast_max_dc_max_cycle,
            zma_fast_max_dc_min_cycle=zma_fast_max_dc_min_cycle,
            zma_fast_max_dc_max_output=zma_fast_max_dc_max_output,
            zma_fast_max_dc_min_output=zma_fast_max_dc_min_output,
            
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # ZATR用パラメータ - すべて
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output
        )
        
        # 参照期間の設定
        self.band_lookback = band_lookback
        
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
                
            # Zチャネルの計算
            result = self.z_channel.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドの取得
            _, upper, lower = self.z_channel.get_bands()
            
            # ブレイクアウトシグナルの計算（高速化版）
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                self.band_lookback
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"ZChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self.z_channel.get_bands()
    
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
            
        return self.z_channel.get_cycle_er()
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        if data is not None:
            self.generate(data)
            
        return self.z_channel.get_dynamic_multiplier()
    
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
            
        return self.z_channel.get_z_atr()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.z_channel.reset() if hasattr(self.z_channel, 'reset') else None
        self._signals_cache = {}
    