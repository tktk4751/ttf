#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, prange

from .indicator import Indicator
from .z_bollinger_bands import ZBollingerBands
from .z_channel import ZChannel


@dataclass
class ZVChannelResult:
    """ZVチャネルの計算結果"""
    middle: np.ndarray        # 中心線（ZMA）
    upper: np.ndarray         # 上限バンド（BB上限とKC上限の平均）
    lower: np.ndarray         # 下限バンド（BB下限とKC下限の平均）
    bb_upper: np.ndarray      # ボリンジャーバンド上限
    bb_lower: np.ndarray      # ボリンジャーバンド下限
    kc_upper: np.ndarray      # Zチャネル上限
    kc_lower: np.ndarray      # Zチャネル下限
    cer: np.ndarray           # サイクル効率比


@njit(fastmath=True, parallel=True, cache=True)
def calculate_combined_bands(
    bb_middle: np.ndarray,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    kc_middle: np.ndarray,
    kc_upper: np.ndarray,
    kc_lower: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ボリンジャーバンドとZチャネルを組み合わせたバンドを計算する（高速化版）
    
    Args:
        bb_middle: ボリンジャーバンド中心線
        bb_upper: ボリンジャーバンド上限
        bb_lower: ボリンジャーバンド下限
        kc_middle: Zチャネル中心線
        kc_upper: Zチャネル上限
        kc_lower: Zチャネル下限
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(bb_middle)
    
    # 中心線は両者の平均を使用
    middle = np.full_like(bb_middle, np.nan, dtype=np.float64)
    upper = np.full_like(bb_middle, np.nan, dtype=np.float64)
    lower = np.full_like(bb_middle, np.nan, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(length):
        # 中心線はボリンジャーの中心を使用（ZMAベース）
        middle[i] = bb_middle[i]
        
        # 上限と下限は両者の平均を計算
        if not (np.isnan(bb_upper[i]) or np.isnan(kc_upper[i])):
            upper[i] = (bb_upper[i] + kc_upper[i]) / 2.0
        
        if not (np.isnan(bb_lower[i]) or np.isnan(kc_lower[i])):
            lower[i] = (bb_lower[i] + kc_lower[i]) / 2.0
    
    return middle, upper, lower


class ZVChannel(Indicator):
    """
    ZVチャネル（ZV Channel）インジケーター
    
    特徴:
    - ボリンジャーバンド（ZBollingerBands）とZチャネル（ZChannel）を組み合わせたハイブリッドチャネル
    - 上限バンドは両者の上限の平均
    - 下限バンドは両者の下限の平均
    - 中心線はZMA（Z Moving Average）を共有
    
    利点:
    - ボリンジャーバンドの価格変動に基づく揺れと、ZチャネルのATRベースの揺れを組み合わせ
    - より安定したチャネルを形成しつつ、両方のメリットを活用
    - 市場状態に適応して変化するバンド幅
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
        zma_max_dc_cycle_part: float = 0.5,     # ZMA最大期間用ドミナントサイクル計算用のサイクル部分
        zma_max_dc_max_cycle: int = 144,        # ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間
        zma_max_dc_min_cycle: int = 5,          # ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間
        zma_max_dc_max_output: int = 89,        # ZMA最大期間用ドミナントサイクル計算用の最大出力値
        zma_max_dc_min_output: int = 22,        # ZMA最大期間用ドミナントサイクル計算用の最小出力値
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA最小期間用ドミナントサイクル計算用のサイクル部分
        zma_min_dc_max_cycle: int = 55,         # ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間
        zma_min_dc_min_cycle: int = 5,          # ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間
        zma_min_dc_max_output: int = 13,        # ZMA最小期間用ドミナントサイクル計算用の最大出力値
        zma_min_dc_min_output: int = 3,         # ZMA最小期間用ドミナントサイクル計算用の最小出力値
        
        zma_max_slow_period: int = 34,          # ZMA遅い移動平均の最大期間
        zma_min_slow_period: int = 9,           # ZMA遅い移動平均の最小期間
        zma_max_fast_period: int = 8,           # ZMA速い移動平均の最大期間
        zma_min_fast_period: int = 2,           # ZMA速い移動平均の最小期間
        zma_hyper_smooth_period: int = 0,       # ZMAハイパースムーサーの平滑化期間（0=平滑化しない）
        
        # ZChannel ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.5,    # ZATR最大期間用ドミナントサイクル計算用のサイクル部分
        zatr_max_dc_max_cycle: int = 55,        # ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間
        zatr_max_dc_min_cycle: int = 5,         # ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間
        zatr_max_dc_max_output: int = 55,       # ZATR最大期間用ドミナントサイクル計算用の最大出力値
        zatr_max_dc_min_output: int = 5,        # ZATR最大期間用ドミナントサイクル計算用の最小出力値
        
        zatr_min_dc_cycle_part: float = 0.25,   # ZATR最小期間用ドミナントサイクル計算用のサイクル部分
        zatr_min_dc_max_cycle: int = 34,        # ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間
        zatr_min_dc_min_cycle: int = 3,         # ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間
        zatr_min_dc_max_output: int = 13,       # ZATR最小期間用ドミナントサイクル計算用の最大出力値
        zatr_min_dc_min_output: int = 3,        # ZATR最小期間用ドミナントサイクル計算用の最小出力値
        
        # 共通パラメータ
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機（デフォルト）
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
                
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        super().__init__(
            f"ZVChannel({cycle_detector_type}, BB:{bb_max_multiplier}-{bb_min_multiplier}, KC:{kc_max_multiplier}-{kc_min_multiplier})"
        )
        
        # 基本パラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.src_type = src_type
        
        # ボリンジャーバンドパラメータ
        self.bb_max_multiplier = bb_max_multiplier
        self.bb_min_multiplier = bb_min_multiplier
        
        # ZBBの標準偏差計算用パラメータ
        self.bb_max_cycle_part = bb_max_cycle_part
        self.bb_max_max_cycle = bb_max_max_cycle
        self.bb_max_min_cycle = bb_max_min_cycle
        self.bb_max_max_output = bb_max_max_output
        self.bb_max_min_output = bb_max_min_output
        self.bb_min_cycle_part = bb_min_cycle_part
        self.bb_min_max_cycle = bb_min_max_cycle
        self.bb_min_min_cycle = bb_min_min_cycle
        self.bb_min_max_output = bb_min_max_output
        self.bb_min_min_output = bb_min_min_output
        
        # Zチャネルパラメータ
        self.kc_max_multiplier = kc_max_multiplier
        self.kc_min_multiplier = kc_min_multiplier
        self.kc_smoother_type = kc_smoother_type
        
        # ZChannel ZMA用パラメータ
        self.zma_max_dc_cycle_part = zma_max_dc_cycle_part
        self.zma_max_dc_max_cycle = zma_max_dc_max_cycle
        self.zma_max_dc_min_cycle = zma_max_dc_min_cycle
        self.zma_max_dc_max_output = zma_max_dc_max_output
        self.zma_max_dc_min_output = zma_max_dc_min_output
        self.zma_min_dc_cycle_part = zma_min_dc_cycle_part
        self.zma_min_dc_max_cycle = zma_min_dc_max_cycle
        self.zma_min_dc_min_cycle = zma_min_dc_min_cycle
        self.zma_min_dc_max_output = zma_min_dc_max_output
        self.zma_min_dc_min_output = zma_min_dc_min_output
        self.zma_max_slow_period = zma_max_slow_period
        self.zma_min_slow_period = zma_min_slow_period
        self.zma_max_fast_period = zma_max_fast_period
        self.zma_min_fast_period = zma_min_fast_period
        self.zma_hyper_smooth_period = zma_hyper_smooth_period
        
        # ZChannel ZATR用パラメータ
        self.zatr_max_dc_cycle_part = zatr_max_dc_cycle_part
        self.zatr_max_dc_max_cycle = zatr_max_dc_max_cycle
        self.zatr_max_dc_min_cycle = zatr_max_dc_min_cycle
        self.zatr_max_dc_max_output = zatr_max_dc_max_output
        self.zatr_max_dc_min_output = zatr_max_dc_min_output
        self.zatr_min_dc_cycle_part = zatr_min_dc_cycle_part
        self.zatr_min_dc_max_cycle = zatr_min_dc_max_cycle
        self.zatr_min_dc_min_cycle = zatr_min_dc_min_cycle
        self.zatr_min_dc_max_output = zatr_min_dc_max_output
        self.zatr_min_dc_min_output = zatr_min_dc_min_output
        
        # 内部インジケーターの初期化
        self.bollinger_bands = ZBollingerBands(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=bb_max_multiplier,
            min_multiplier=bb_min_multiplier,
            max_cycle_part=bb_max_cycle_part,
            max_max_cycle=bb_max_max_cycle,
            max_min_cycle=bb_max_min_cycle,
            max_max_output=bb_max_max_output,
            max_min_output=bb_max_min_output,
            min_cycle_part=bb_min_cycle_part,
            min_max_cycle=bb_min_max_cycle,
            min_min_cycle=bb_min_min_cycle,
            min_max_output=bb_min_max_output,
            min_min_output=bb_min_min_output,
            src_type=src_type
        )
        
        self.keltner_channel = ZChannel(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=kc_max_multiplier,
            min_multiplier=kc_min_multiplier,
            smoother_type=kc_smoother_type,
            src_type=src_type,
            
            # ZMA用パラメータ
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
            
            # ZATR用パラメータ
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
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする（高速化）
            cols = ['high', 'low', 'close']
            if self.src_type == 'ohlc4' and 'open' in data.columns:
                cols.append('open')
            # NumPyでの高速ハッシュ計算
            data_values = np.vstack([data[col].values for col in cols if col in data.columns])
            data_hash = hash(data_values.tobytes())
        else:
            # NumPy配列の場合は全体をハッシュする（高速化）
            data_hash = hash(data.tobytes() if isinstance(data, np.ndarray) else str(data))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = (
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"bb_{self.bb_max_multiplier}_{self.bb_min_multiplier}_"
            f"{self.bb_max_cycle_part}_{self.bb_max_max_cycle}_{self.bb_max_min_cycle}_"
            f"{self.bb_max_max_output}_{self.bb_max_min_output}_"
            f"{self.bb_min_cycle_part}_{self.bb_min_max_cycle}_{self.bb_min_min_cycle}_"
            f"{self.bb_min_max_output}_{self.bb_min_min_output}_"
            f"kc_{self.kc_max_multiplier}_{self.kc_min_multiplier}_{self.kc_smoother_type}_"
            f"zma_{self.zma_max_dc_cycle_part}_{self.zma_max_dc_max_cycle}_{self.zma_max_dc_min_cycle}_"
            f"{self.zma_max_dc_max_output}_{self.zma_max_dc_min_output}_"
            f"{self.zma_min_dc_cycle_part}_{self.zma_min_dc_max_cycle}_{self.zma_min_dc_min_cycle}_"
            f"{self.zma_min_dc_max_output}_{self.zma_min_dc_min_output}_"
            f"{self.zma_max_slow_period}_{self.zma_min_slow_period}_{self.zma_max_fast_period}_"
            f"{self.zma_min_fast_period}_{self.zma_hyper_smooth_period}_"
            f"zatr_{self.zatr_max_dc_cycle_part}_{self.zatr_max_dc_max_cycle}_{self.zatr_max_dc_min_cycle}_"
            f"{self.zatr_max_dc_max_output}_{self.zatr_max_dc_min_output}_"
            f"{self.zatr_min_dc_cycle_part}_{self.zatr_min_dc_max_cycle}_{self.zatr_min_dc_min_cycle}_"
            f"{self.zatr_min_dc_max_output}_{self.zatr_min_dc_min_output}_"
            f"{self.src_type}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ZVチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            中心線の値（ZMA）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.middle
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # ボリンジャーバンドの計算
            self.bollinger_bands.calculate(data)
            bb_middle, bb_upper, bb_lower = self.bollinger_bands.get_bands()
            
            # Zチャネルの計算
            self.keltner_channel.calculate(data)
            kc_middle, kc_upper, kc_lower = self.keltner_channel.get_bands()
            
            # サイクル効率比はボリンジャーバンドから取得（どちらからでも同じはず）
            cer = self.bollinger_bands.get_cycle_er()
            
            # 組み合わせたバンドの計算（Numba高速化関数を使用）
            middle, upper, lower = calculate_combined_bands(
                bb_middle, bb_upper, bb_lower,
                kc_middle, kc_upper, kc_lower
            )
            
            # 結果の保存
            self._result = ZVChannelResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                bb_upper=np.copy(bb_upper),
                bb_lower=np.copy(bb_lower),
                kc_upper=np.copy(kc_upper),
                kc_lower=np.copy(kc_lower),
                cer=np.copy(cer)
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZVChannel計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は前回の結果を維持する（nullではなく）
            if self._result is None:
                # 初回エラー時は空の結果を作成
                empty_array = np.array([])
                self._result = ZVChannelResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    bb_upper=empty_array,
                    bb_lower=empty_array,
                    kc_upper=empty_array,
                    kc_lower=empty_array,
                    cer=empty_array
                )
                self._values = empty_array
            
            return self._values
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ZVチャネルのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_bb_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部で計算したボリンジャーバンドの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.bb_upper, self._result.bb_lower
    
    def get_kc_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        内部で計算したZチャネル（ケルトナーチャネル）の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.kc_upper, self._result.kc_lower
    
    def get_cycle_er(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.cer
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.bollinger_bands.reset()
        self.keltner_channel.reset() 