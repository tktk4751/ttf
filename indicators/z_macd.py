#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, vectorize, prange

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .cycle.ehlers_unified_dc import EhlersUnifiedDC
from .z_ma import ZMA, calculate_z_ma
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZMACDResult:
    """ZMACDの計算結果"""
    macd: np.ndarray       # MACD線
    signal: np.ndarray     # シグナル線
    histogram: np.ndarray  # ヒストグラム
    er: np.ndarray         # 効率比
    dc_values: np.ndarray  # ドミナントサイクル値
    fast_dynamic_period: np.ndarray  # 動的Fast期間
    slow_dynamic_period: np.ndarray  # 動的Slow期間
    signal_dynamic_period: np.ndarray  # 動的Signal期間


@njit(fastmath=True)
def calculate_z_macd(
    close: np.ndarray,
    er: np.ndarray,
    er_period: int,
    fast_kama_period: np.ndarray,
    slow_kama_period: np.ndarray,
    signal_kama_period: np.ndarray,
    fast_constants: np.ndarray,
    slow_constants: np.ndarray,
    signal_constants: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ZMACDを計算する（高速化版）
    
    Args:
        close: 終値の配列
        er: 効率比の配列
        er_period: 効率比の計算期間
        fast_kama_period: 短期ZMAの動的期間配列
        slow_kama_period: 長期ZMAの動的期間配列
        signal_kama_period: シグナルZMAの動的期間配列
        fast_constants: 短期ZMAの定数配列 [fast_constant, slow_constant]
        slow_constants: 長期ZMAの定数配列 [fast_constant, slow_constant]
        signal_constants: シグナルZMAの定数配列 [fast_constant, slow_constant]
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: MACD線、シグナル線、ヒストグラムの配列
    """
    # 短期と長期のZMAを計算
    fast_zma = calculate_z_ma(
        close, 
        er, 
        er_period, 
        fast_kama_period, 
        fast_constants[:, 0], 
        fast_constants[:, 1]
    )
    
    slow_zma = calculate_z_ma(
        close, 
        er, 
        er_period, 
        slow_kama_period, 
        slow_constants[:, 0], 
        slow_constants[:, 1]
    )
    
    # MACD線を計算
    macd_line = fast_zma - slow_zma
    
    # シグナル線を計算（通常のEMAを使用）
    size = len(macd_line)
    signal_line = np.zeros(size)
    
    # 最初の有効なMACD値のインデックスを見つける
    start_idx = 0
    for i in range(size):
        if not np.isnan(macd_line[i]):
            start_idx = i
            break
    
    # 初期値を設定
    if start_idx < size:
        signal_line[start_idx] = macd_line[start_idx]
    
    # 固定期間のEMAを計算
    if len(signal_kama_period) > 0:
        # 各ポイントでの期間を使用
        for i in range(start_idx + 1, size):
            period = int(signal_kama_period[i])
            if period < 1:
                period = 1
            alpha = 2.0 / (period + 1.0)
            signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i-1]
    else:
        # 固定期間を使用（デフォルト: 9）
        period = 9
        alpha = 2.0 / (period + 1.0)
        for i in range(start_idx + 1, size):
            signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i-1]
    
    # ヒストグラムを計算
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class ZMACD(Indicator):
    """
    ZMACD インディケーター
    
    Alpha MACDの拡張版。ZMAを使用し、サイクル効率比とドミナントサイクルを用いて動的に期間を調整します。
    
    特徴:
    - ZMAを使用した高度な平滑化
    - ドミナントサイクルを用いた動的な期間計算
    - サイクル効率比による細かな調整
    - トレンドが強い時：短い期間で素早く反応
    - レンジ相場時：長い期間でノイズを除去
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,          # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 144,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 5,              # 最大期間用ドミナントサイクル計算用
        
        # 短期線用パラメータ
        fast_max_dc_max_output: int = 21,        # 短期線用最大期間出力値
        fast_max_dc_min_output: int = 5,         # 短期線用最大期間出力の最小値
        
        # 長期線用パラメータ
        slow_max_dc_max_output: int = 55,       # 長期線用最大期間出力値
        slow_max_dc_min_output: int = 13,        # 長期線用最大期間出力の最小値
        
        # シグナル線用パラメータ
        signal_max_dc_max_output: int = 21,      # シグナル線用最大期間出力値
        signal_max_dc_min_output: int = 5,       # シグナル線用最大期間出力の最小値
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 55,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 3,               # 最小期間用ドミナントサイクル計算用
        
        # 短期線用最小期間パラメータ
        fast_min_dc_max_output: int = 13,        # 短期線用最小期間出力値
        fast_min_dc_min_output: int = 2,         # 短期線用最小期間出力の最小値
        
        # 長期線用最小期間パラメータ
        slow_min_dc_max_output: int = 34,        # 長期線用最小期間出力値
        slow_min_dc_min_output: int = 8,         # 長期線用最小期間出力の最小値
        
        # シグナル線用最小期間パラメータ
        signal_min_dc_max_output: int = 8,       # シグナル線用最小期間出力値
        signal_min_dc_min_output: int = 2,       # シグナル線用最小期間出力の最小値
        
        # 共通パラメータ
        max_slow_period: int = 34,               # 遅い移動平均の最大期間
        min_slow_period: int = 13,               # 遅い移動平均の最小期間
        max_fast_period: int = 8,               # 速い移動平均の最大期間
        min_fast_period: int = 2,                # 速い移動平均の最小期間
        er_period: int = 21,                     # 効率比の計算期間
        hyper_smooth_period: int = 0,            # ハイパースムーサー期間（0=無効）
        src_type: str = 'close',                 # ソースタイプ
        detector_type: str = 'hody'              # 検出器タイプ
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル検出の最大期間（デフォルト: 144）
            max_dc_min_cycle: 最大期間用ドミナントサイクル検出の最小期間（デフォルト: 5）
            
            fast_max_dc_max_output: 短期線用最大期間出力値（デフォルト: 89）
            fast_max_dc_min_output: 短期線用最大期間出力の最小値（デフォルト: 8）
            
            slow_max_dc_max_output: 長期線用最大期間出力値（デフォルト: 144）
            slow_max_dc_min_output: 長期線用最大期間出力の最小値（デフォルト: 21）
            
            signal_max_dc_max_output: シグナル線用最大期間出力値（デフォルト: 55）
            signal_max_dc_min_output: シグナル線用最大期間出力の最小値（デフォルト: 5）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル検出の最大期間（デフォルト: 55）
            min_dc_min_cycle: 最小期間用ドミナントサイクル検出の最小期間（デフォルト: 3）
            
            fast_min_dc_max_output: 短期線用最小期間出力値（デフォルト: 13）
            fast_min_dc_min_output: 短期線用最小期間出力の最小値（デフォルト: 2）
            
            slow_min_dc_max_output: 長期線用最小期間出力値（デフォルト: 34）
            slow_min_dc_min_output: 長期線用最小期間出力の最小値（デフォルト: 8）
            
            signal_min_dc_max_output: シグナル線用最小期間出力値（デフォルト: 8）
            signal_min_dc_min_output: シグナル線用最小期間出力の最小値（デフォルト: 2）
            
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 89）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 30）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 13）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
            er_period: 効率比の計算期間（デフォルト: 21）
            hyper_smooth_period: ハイパースムーサー期間（0=無効）（デフォルト: 0）
            src_type: ソースタイプ（デフォルト: 'close'）
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
        """
        super().__init__(
            f"ZMACD({fast_max_dc_max_output}-{slow_max_dc_max_output}-{signal_max_dc_max_output})"
        )
        
        # ドミナントサイクルの検出器 - 短期線用
        self.fast_max_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=fast_max_dc_max_output,
            min_output=fast_max_dc_min_output,
            src_type=src_type
        )
        
        self.fast_min_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=fast_min_dc_max_output,
            min_output=fast_min_dc_min_output,
            src_type=src_type
        )
        
        # ドミナントサイクルの検出器 - 長期線用
        self.slow_max_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=slow_max_dc_max_output,
            min_output=slow_max_dc_min_output,
            src_type=src_type
        )
        
        self.slow_min_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=slow_min_dc_max_output,
            min_output=slow_min_dc_min_output,
            src_type=src_type
        )
        
        # ドミナントサイクルの検出器 - シグナル線用
        self.signal_max_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=max_dc_cycle_part,
            max_cycle=max_dc_max_cycle,
            min_cycle=max_dc_min_cycle,
            max_output=signal_max_dc_max_output,
            min_output=signal_max_dc_min_output,
            src_type=src_type
        )
        
        self.signal_min_dc = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=min_dc_cycle_part,
            max_cycle=min_dc_max_cycle,
            min_cycle=min_dc_min_cycle,
            max_output=signal_min_dc_max_output,
            min_output=signal_min_dc_min_output,
            src_type=src_type
        )
        
        # ZMA設定
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        
        self.er_period = er_period
        self.hyper_smooth_period = hyper_smooth_period
        self.src_type = src_type
        self.detector_type = detector_type
        
        # ZMA初期化
        self.fast_zma = ZMA(
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=fast_max_dc_max_output,
            max_dc_min_output=fast_max_dc_min_output,
            
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=fast_min_dc_max_output,
            min_dc_min_output=fast_min_dc_min_output,
            
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            hyper_smooth_period=hyper_smooth_period,
            src_type=src_type,
            detector_type=detector_type
        )
        
        self.slow_zma = ZMA(
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=slow_max_dc_max_output,
            max_dc_min_output=slow_max_dc_min_output,
            
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=slow_min_dc_max_output,
            min_dc_min_output=slow_min_dc_min_output,
            
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            hyper_smooth_period=hyper_smooth_period,
            src_type=src_type,
            detector_type=detector_type
        )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if self.src_type == 'close' and 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            elif set(['high', 'low', 'close']).issubset(data.columns):
                data_hash = hash(tuple(map(tuple, data[['high', 'low', 'close']].values)))
            else:
                # 必要なカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合は必要な列だけハッシュ
                if self.src_type == 'close':
                    data_hash = hash(tuple(data[:, 3]))  # close
                else:
                    data_hash = hash(tuple(map(tuple, data[:, 1:4])))  # high, low, close
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # 外部ERがある場合はそのハッシュも含める
        external_er_hash = "no_external_er"
        if external_er is not None:
            external_er_hash = hash(tuple(external_er))
        
        # パラメータ値を含める
        param_str = (
            f"{self.fast_max_dc.cycle_part}_{self.fast_max_dc.max_cycle}_{self.fast_max_dc.min_cycle}_"
            f"{self.fast_max_dc.max_output}_{self.fast_max_dc.min_output}_"
            f"{self.slow_max_dc.max_output}_{self.slow_max_dc.min_output}_"
            f"{self.signal_max_dc.max_output}_{self.signal_max_dc.min_output}_"
            f"{self.detector_type}_{self.er_period}_{self.hyper_smooth_period}_{self.src_type}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ZMACDを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したsrc_typeに必要なカラムが必要
            external_er: 外部から提供される効率比（オプション）
        
        Returns:
            ZMACD値の配列（ヒストグラム）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.histogram
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # 価格データを取得
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    # 適切なソースを使用
                    prices = self.fast_zma.calculate_source_values(data, self.src_type)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    prices = data[:, 3]  # close
                else:
                    prices = data
            
            # データ長の検証
            data_length = len(prices)
            
            # 効率比の計算（外部から提供されない場合は計算）
            if external_er is None:
                # CycleEfficiencyRatioを使用して効率比を計算
                from .cycle_efficiency_ratio import CycleEfficiencyRatio
                cycle_er = CycleEfficiencyRatio(
                    cycle_detector_type=self.detector_type,
                    lp_period=5,
                    hp_period=144,
                    cycle_part=0.5,
                    max_cycle=144,
                    min_cycle=5,
                    max_output=89,
                    min_output=5,
                    src_type=self.src_type
                )
                er = cycle_er.calculate(data)
            else:
                # 外部から提供されるERをそのまま使用
                er = external_er
                # 外部ERの長さ検証
                if len(external_er) != data_length:
                    raise ValueError(f"外部ERの長さ({len(external_er)})がデータ長({data_length})と一致しません")
            
            # ZMAを使用して短期・長期線を計算（サイクル効率比を渡す）
            fast_zma_values = self.fast_zma.calculate(data, er)
            slow_zma_values = self.slow_zma.calculate(data, er)
            
            # MACD線を計算
            macd_line = fast_zma_values - slow_zma_values
            
            # シグナル線用のドミナントサイクルを計算
            signal_max_periods = self.signal_max_dc.calculate(data)
            signal_min_periods = self.signal_min_dc.calculate(data)
            
            # 動的なシグナル期間を計算
            from .z_ma import calculate_dynamic_kama_period
            signal_dynamic_period = calculate_dynamic_kama_period(er, signal_max_periods, signal_min_periods)
            
            # 定数を計算
            from .z_ma import calculate_dynamic_kama_constants
            _, _, signal_fast_constants, signal_slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            # シグナル線を計算（通常のEMAを使用）
            signal_period = 9  # 標準的なMACDのシグナル期間
            
            # EMAの計算
            signal_line = np.zeros(data_length)
            
            # 最初の有効なMACD値のインデックスを見つける
            start_idx = 0
            for i in range(data_length):
                if not np.isnan(macd_line[i]):
                    start_idx = i
                    break
                    
            # 初期値を設定
            if start_idx < data_length:
                signal_line[start_idx] = macd_line[start_idx]
                
            # シグナル線のEMAを計算
            alpha = 2.0 / (signal_period + 1.0)
            for i in range(start_idx + 1, data_length):
                signal_line[i] = alpha * macd_line[i] + (1.0 - alpha) * signal_line[i-1]
            
            # ヒストグラムを計算
            histogram = macd_line - signal_line
            
            # 結果を保存
            self._result = ZMACDResult(
                macd=macd_line,
                signal=signal_line,
                histogram=histogram,
                er=er,
                dc_values=signal_max_periods,  # シグナル用の最大ドミナントサイクル値を保存
                fast_dynamic_period=self.fast_zma._result.dynamic_kama_period if (hasattr(self.fast_zma, '_result') and self.fast_zma._result is not None) else np.array([]),
                slow_dynamic_period=self.slow_zma._result.dynamic_kama_period if (hasattr(self.slow_zma, '_result') and self.slow_zma._result is not None) else np.array([]),
                signal_dynamic_period=signal_dynamic_period
            )
            
            self._values = histogram  # 基底クラスの要件を満たすため
            
            return histogram
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZMACD計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_lines(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD線、シグナル線、ヒストグラムを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (MACD線, シグナル線, ヒストグラム)のタプル
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty, empty
        return self._result.macd, self._result.signal, self._result.histogram
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_dc_values(self) -> np.ndarray:
        """
        ドミナントサイクル値を取得する
        
        Returns:
            np.ndarray: ドミナントサイクル値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的な期間の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (Fast期間, Slow期間, Signal期間)の値
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty, empty
        return (
            self._result.fast_dynamic_period,
            self._result.slow_dynamic_period,
            self._result.signal_dynamic_period
        )
    
    def get_crossover_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        クロスオーバー・クロスアンダーのシグナルを取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (クロスオーバーシグナル, クロスアンダーシグナル)のタプル
                シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        
        # MACDとシグナルラインの取得
        macd = self._result.macd
        signal = self._result.signal
        
        # 1つ前の値を取得
        prev_macd = np.roll(macd, 1)
        prev_signal = np.roll(signal, 1)
        prev_macd[0] = macd[0]
        prev_signal[0] = signal[0]
        
        # クロスオーバー: 前の値がシグナル線未満で、現在の値がシグナル線以上
        crossover = np.where(
            (prev_macd < prev_signal) & (macd >= signal),
            1, 0
        )
        
        # クロスアンダー: 前の値がシグナル線以上で、現在の値がシグナル線未満
        crossunder = np.where(
            (prev_macd >= prev_signal) & (macd < signal),
            1, 0
        )
        
        return crossover, crossunder
    
    def get_zero_crossover_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ゼロラインのクロスオーバー・クロスアンダーのシグナルを取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (ゼロラインクロスオーバーシグナル, ゼロラインクロスアンダーシグナル)のタプル
                シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty
        
        # MACDラインの取得
        macd = self._result.macd
        
        # 1つ前の値を取得
        prev_macd = np.roll(macd, 1)
        prev_macd[0] = macd[0]
        
        # ゼロラインクロスオーバー: 前の値が0未満で、現在の値が0以上
        zero_crossover = np.where(
            (prev_macd < 0) & (macd >= 0),
            1, 0
        )
        
        # ゼロラインクロスアンダー: 前の値が0以上で、現在の値が0未満
        zero_crossunder = np.where(
            (prev_macd >= 0) & (macd < 0),
            1, 0
        )
        
        return zero_crossover, zero_crossunder
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.fast_zma.reset()
        self.slow_zma.reset()
        self.fast_max_dc.reset()
        self.fast_min_dc.reset()
        self.slow_max_dc.reset()
        self.slow_min_dc.reset()
        self.signal_max_dc.reset()
        self.signal_min_dc.reset() 