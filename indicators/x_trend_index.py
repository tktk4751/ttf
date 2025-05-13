#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from .indicator import Indicator
from .hyper_smoother import hyper_smoother
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .c_atr import CATR
from .ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class XTrendIndexResult:
    """Xトレンドインデックスの計算結果"""
    values: np.ndarray          # Xトレンドインデックスの値（0-1の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    dominant_cycle: np.ndarray  # ドミナントサイクル値（チョピネス期間として使用）
    dynamic_atr_period: np.ndarray   # 動的ATR期間 (CATRから取得)
    choppiness_index: np.ndarray # Choppiness Index（元の値）
    range_index: np.ndarray     # Range Index（元の値）
    stddev_factor: np.ndarray   # 標準偏差係数 (固定期間で計算)
    tr: np.ndarray              # True Range (CATR計算時に内部で計算されるが、結果として保持)
    atr: np.ndarray             # Average True Range (CATRの絶対値)
    dynamic_threshold: np.ndarray  # 動的しきい値
    trend_state: np.ndarray     # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)


@njit(fastmath=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True, parallel=True)
def calculate_choppiness_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """
    動的期間によるチョピネス指数を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 動的な期間の配列
        tr: True Rangeの配列

    
    Returns:
        チョピネス指数の配列（0-100の範囲）
    """
    length = len(high)
    chop = np.zeros(length, dtype=np.float64)
    
    for i in prange(1, length):
        curr_period = int(period[i])
        if curr_period < 2:
            curr_period = 2
        
        if i < curr_period:
            continue
        
        # True Range の合計
        tr_sum = np.sum(tr[i - curr_period + 1:i + 1])
        
        # 期間内の最高値と最安値を取得
        idx_start = i - curr_period + 1
        period_high = np.max(high[idx_start:i + 1])
        period_low = np.min(low[idx_start:i + 1])
        
        # 価格レンジの計算
        price_range = period_high - period_low
        
        # チョピネス指数の計算
        if price_range > 0 and curr_period > 1 and tr_sum > 0:
            log_period = np.log10(float(curr_period))
            chop_val = 100.0 * np.log10(tr_sum / price_range) / log_period
            # 値を0-100の範囲に制限
            chop[i] = max(0.0, min(100.0, chop_val))
        else:
            chop[i] = 0.0
    
    return chop


@njit(fastmath=True, parallel=True)
def calculate_stddev_factor(atr: np.ndarray) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する (固定期間: 期間=14, ルックバック=14)

    Args:
        atr: ATR配列
    Returns:
        標準偏差係数
    """
    n = len(atr)
    fixed_period = 14
    fixed_lookback = 14
    stddev = np.zeros(n, dtype=np.float64)
    lowest_stddev = np.full(n, np.inf, dtype=np.float64)
    stddev_factor = np.ones(n, dtype=np.float64)

    for i in prange(n):
        if i >= fixed_period - 1:
            start_idx = i - fixed_period + 1
            atr_window = atr[start_idx:i+1]

            # numpy.stdを使用して標準偏差を計算 (ddof=1で不偏標準偏差)
            # stddev[i] = np.std(atr_window, ddof=1) # np.stdはNumbaのparallel=Trueでは推奨されない場合がある

            # PineScriptのSMAを使用した計算方法を維持
            stddev_a = np.mean(np.power(atr_window, 2))
            stddev_b = np.power(np.sum(atr_window), 2) / np.power(len(atr_window), 2)
            curr_stddev = np.sqrt(max(0.0, stddev_a - stddev_b))
            stddev[i] = curr_stddev

            # 最小標準偏差の更新（固定ルックバック期間内で）
            lowest_lookback_start = max(0, i - fixed_lookback + 1)
            # windowがlookback期間より短い場合も考慮
            valid_stddev_window = stddev[lowest_lookback_start : i + 1]
            # infを除外して最小値を計算
            valid_stddev_window_finite = valid_stddev_window[np.isfinite(valid_stddev_window)]
            if len(valid_stddev_window_finite) > 0:
                 lowest_stddev[i] = np.min(valid_stddev_window_finite)
            else:
                 # 期間内に有効な標準偏差がない場合は現在の値を使用
                 lowest_stddev[i] = stddev[i] if np.isfinite(stddev[i]) else np.inf


            # 標準偏差係数の計算
            if stddev[i] > 0 and np.isfinite(lowest_stddev[i]):
                stddev_factor[i] = lowest_stddev[i] / stddev[i]
            elif i > 0:
                 stddev_factor[i] = stddev_factor[i-1] # 前の値を使用
            else:
                 stddev_factor[i] = 1.0 # 初期値

        elif i > 0:
            # データ不足の場合は前の値を使用
            stddev[i] = stddev[i-1]
            lowest_stddev[i] = lowest_stddev[i-1]
            stddev_factor[i] = stddev_factor[i-1]
        else:
            # 最初の要素はNaNまたはデフォルト値
             stddev[i] = np.nan
             lowest_stddev[i] = np.inf
             stddev_factor[i] = 1.0

    return stddev_factor


@njit(fastmath=True)
def calculate_x_trend_index(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    Xトレンドインデックスを計算する
    
    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        Xトレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    range_index = chop * stddev_factor
    
    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)
    
    # 値を0-1の範囲にクリップ
    trend_index = np.maximum(0.0, np.minimum(1.0, trend_index))
    
    return trend_index


@njit(fastmath=True)
def calculate_x_trend_index_batch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dominant_cycle: np.ndarray,
    atr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Xトレンドインデックスを一括計算する

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        dominant_cycle: ドミナントサイクル値の配列（チョピネス期間として使用）
        atr: CATR (金額ベース) の配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (Xトレンドインデックス, チョピネス指数, 標準偏差係数)
    """
    # True Rangeの計算
    tr = calculate_tr(high, low, close)

    # チョピネスインデックスの計算 (DC値を期間として使用)
    # DC値が0や負にならないように最小値を2にクリップ
    chop_period = np.maximum(2, dominant_cycle).astype(np.int32)
    chop_index = calculate_choppiness_index(high, low, close, chop_period, tr)

    # 標準偏差係数の計算 (固定期間を使用)
    stddev_factor = calculate_stddev_factor(atr)

    # トレンドインデックスの計算
    trend_index = calculate_x_trend_index(chop_index, stddev_factor)

    return trend_index, chop_index, stddev_factor


@njit(fastmath=True)
def calculate_dynamic_threshold(
    er: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する
    
    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        動的なしきい値の配列
    """
    length = len(er)
    threshold = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er[i]):
            threshold[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # ERが高いほど（トレンドが強いほど）しきい値は高く
        # ERが低いほど（レンジ相場ほど）しきい値は低く
        threshold[i] = min_threshold + er_abs * (max_threshold - min_threshold)
    
    return threshold


class XTrendIndex(Indicator):
    """
    Xトレンドインデックス（X Trend Index）インジケーター
    
    サイクル効率比（CER）とEhlersUnifiedDCを使用してチョピネス期間を動的に決定し、
    CATRと固定期間の標準偏差係数を組み合わせてトレンド/レンジを検出する指標。
    ZTrendIndexをベースに、チョピネス期間の決定方法とATR計算、標準偏差期間を変更。
    
    特徴:
    - サイクル効率比（CER）を使用して、現在のサイクルに基づいた適応的な計算
    - EhlersUnifiedDCを使用してチョピネス期間を動的に決定
    - CATRを使用したサイクル適応型ボラティリティ測定
    - チョピネスインデックスと固定期間の標準偏差係数を組み合わせて正規化したトレンド指標
    - 市場状態に応じてチョピネス期間とATR期間が自動調整される
    - 0-1の範囲で表示（1に近いほど強いトレンド、0に近いほど強いレンジ）
    - 動的しきい値によるトレンド/レンジ状態の判定
    """
    
    def __init__(
        self,
        # EhlersUnifiedDC パラメータ
        detector_type: str = 'phac_e',
        cycle_part: float = 0.5,
        max_cycle: int = 55, # CATRと合わせる例
        min_cycle: int = 5,  # CATRと合わせる例
        max_output: int = 34, # CATRと合わせる例
        min_output: int = 5,  # CATRと合わせる例
        src_type: str = 'hlc3', # DC計算ソース
        lp_period: int = 5,    # 拡張DC用
        hp_period: int = 55,   # 拡張DC用

        # CATR パラメータ (DCパラメータは上記と共有可能)
        smoother_type: str = 'alma', # CATRの平滑化タイプ

        # CycleEfficiencyRatio パラメータ (CATRに必要)
        cer_detector_type: str = 'phac_e', # CER用DCタイプ
        cer_lp_period: int = 5,       # CER用LP期間
        cer_hp_period: int = 144,      # CER用HP期間
        cer_cycle_part: float = 0.5,   # CER用サイクル部分

        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: EhlersUnifiedDCで使用する検出器タイプ (デフォルト: 'hody')
            cycle_part: DCのサイクル部分の倍率 (デフォルト: 0.5)
            max_cycle: DCの最大サイクル期間 (デフォルト: 55)
            min_cycle: DCの最小サイクル期間 (デフォルト: 5)
            max_output: DCの最大出力値 (デフォルト: 34)
            min_output: DCの最小出力値 (デフォルト: 5)
            src_type: DC計算に使用する価格ソース ('close', 'hlc3', etc.) (デフォルト: 'hlc3')
            lp_period: 拡張DC用のローパスフィルター期間 (デフォルト: 5)
            hp_period: 拡張DC用のハイパスフィルター期間 (デフォルト: 55)
            smoother_type: CATRで使用する平滑化アルゴリズム ('alma' or 'hyper') (デフォルト: 'alma')
            cer_detector_type: CycleEfficiencyRatioで使用する検出器タイプ (デフォルト: 'hody')
            cer_lp_period: CER用のローパスフィルター期間 (デフォルト: 5)
            cer_hp_period: CER用のハイパスフィルター期間 (デフォルト: 144)
            cer_cycle_part: CER用のサイクル部分の倍率 (デフォルト: 0.5)
            max_threshold: 動的しきい値の最大値 (デフォルト: 0.75)
            min_threshold: 動的しきい値の最小値 (デフォルト: 0.55)
        """
        super().__init__(
            f"XTrendIndex({detector_type}, {max_output}, {min_output}, {smoother_type})"
        )
        
        # ドミナントサイクル検出器 (EhlersUnifiedDC) のパラメータ
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        self.lp_period = lp_period
        self.hp_period = hp_period

        # CATRパラメータ
        self.smoother_type = smoother_type

        # CycleEfficiencyRatio パラメータ (CATRとThresholdに必要)
        self.cer_detector_type = cer_detector_type
        self.cer_lp_period = cer_lp_period
        self.cer_hp_period = cer_hp_period
        self.cer_cycle_part = cer_cycle_part

        # 動的しきい値のパラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

        # ドミナントサイクル検出器の初期化 (EhlersUnifiedDCを使用)
        self.dc_detector = EhlersUnifiedDC(
            detector_type=self.detector_type,
            cycle_part=self.cycle_part,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            src_type=self.src_type,
            lp_period=self.lp_period,
            hp_period=self.hp_period
        )

        # サイクル効率比(CER)のインスタンス化 (CATRとThresholdに必要)
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=self.cer_detector_type,
            lp_period=self.cer_lp_period,
            hp_period=self.cer_hp_period,
            cycle_part=self.cer_cycle_part
        )

        # CATRのインスタンス化 (DC検出器は別パラメータを持つ可能性があるので注意)
        # CATR内部のDC検出器は、CATR自身のパラメータで初期化される
        self.c_atr_indicator = CATR(
             detector_type=self.detector_type, # XTrendIndexと同じDC設定を使う場合
             cycle_part=self.cycle_part,
             lp_period=self.lp_period,
             hp_period=self.hp_period,
             max_cycle=self.max_cycle,
             min_cycle=self.min_cycle,
             max_output=self.max_output,
             min_output=self.min_output,
             smoother_type=self.smoother_type
        )

        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            cols = ['open', 'high', 'low', 'close']
            data_hash_part = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            data_hash_part = hash(tuple(map(tuple, data)))

        # パラメータ文字列を生成 (XTrendIndexの全パラメータを含む)
        param_str = (
            f"{self.detector_type}_{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_"
            f"{self.max_output}_{self.min_output}_{self.src_type}_{self.lp_period}_{self.hp_period}_"
            f"{self.smoother_type}_"
            f"{self.cer_detector_type}_{self.cer_lp_period}_{self.cer_hp_period}_{self.cer_cycle_part}_"
            f"{self.max_threshold}_{self.min_threshold}"
        )
        return f"{data_hash_part}_{hash(param_str)}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XTrendIndexResult:
        """
        Xトレンドインデックスを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            XTrendIndexResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result

            self._data_hash = data_hash

            # データ検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                # openがない場合はcloseで代用するか、エラーにするか選択。ここでは0埋め。
                o = np.asarray(data.get('open', data['close']).values, dtype=np.float64)
                h = np.asarray(data['high'].values, dtype=np.float64)
                l = np.asarray(data['low'].values, dtype=np.float64)
                c = np.asarray(data['close'].values, dtype=np.float64)
                # DataFrameを渡す必要がある場合
                df_data = data[['open', 'high', 'low', 'close']] if 'open' in data.columns else data[['high', 'low', 'close']]

            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 4:
                o = np.asarray(data[:, 0], dtype=np.float64)
                h = np.asarray(data[:, 1], dtype=np.float64)
                l = np.asarray(data[:, 2], dtype=np.float64)
                c = np.asarray(data[:, 3], dtype=np.float64)
                 # DataFrameが必要なインジケータ用に一時的に作成
                df_data = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c})
            else:
                raise ValueError("データはPandas DataFrameまたは4列以上のNumPy配列である必要があります")

            # ドミナントサイクルの計算 (EhlersUnifiedDCを使用)
            # このDC値がチョピネス期間として使われる
            dominant_cycle = self.dc_detector.calculate(df_data) # UnifiedDCはDataFrame/ndarrayを受け付ける

            # サイクル効率比(CER)の計算 (CATRとThresholdに必要)
            er = self.cycle_er.calculate(df_data) # CycleEfficiencyRatioはDataFrameを期待

            # CATRの計算 (CERが必要)
            # CATRは内部で自身のDC設定に基づきATR期間を計算する
            self.c_atr_indicator.calculate(df_data, external_er=er) # CATRはDataFrameとerを期待
            atr = self.c_atr_indicator.get_absolute_atr() # 金額ベースATR
            dynamic_atr_period = self.c_atr_indicator.get_atr_period() # CATRが計算したATR期間

            # 一括計算 (X Trend Index, Choppiness, StdDev Factor)
            trend_index, chop_index, stddev_factor = calculate_x_trend_index_batch(
                h, l, c,
                dominant_cycle, # チョピネス期間としてDC値を使用
                atr             # CATR (金額ベース)
            )

            # True Rangeの計算 (結果オブジェクト用)
            tr = calculate_tr(h, l, c)

            # 動的しきい値の計算
            dynamic_threshold = calculate_dynamic_threshold(
                er, self.max_threshold, self.min_threshold
            )

            # トレンド状態の計算
            length = len(trend_index)
            trend_state = np.full(length, np.nan)
            for i in range(length):
                if np.isnan(trend_index[i]) or np.isnan(dynamic_threshold[i]):
                    continue
                trend_state[i] = 1.0 if trend_index[i] >= dynamic_threshold[i] else 0.0

            # 結果オブジェクトを作成 (XTrendIndexResult)
            result = XTrendIndexResult(
                values=trend_index,
                er=er,
                dominant_cycle=dominant_cycle, # チョピネスに使ったDC値
                dynamic_atr_period=dynamic_atr_period, # CATRが計算したATR期間
                choppiness_index=chop_index,
                range_index=100.0 - chop_index, # レンジインデックス (0-100)
                stddev_factor=stddev_factor,
                tr=tr,
                atr=atr, # CATRの絶対値
                dynamic_threshold=dynamic_threshold,
                trend_state=trend_state
            )

            self._result = result
            self._values = trend_index # Indicatorクラスの標準出力

            return result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"XTrendIndex計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時はNaN配列を返すか、以前の結果を返すか選択。ここでは空配列。
            n = len(data) if hasattr(data, '__len__') else 0
            empty_result = XTrendIndexResult(
                 values=np.full(n, np.nan), er=np.full(n, np.nan), dominant_cycle=np.full(n, np.nan),
                 dynamic_atr_period=np.full(n, np.nan), choppiness_index=np.full(n, np.nan),
                 range_index=np.full(n, np.nan), stddev_factor=np.full(n, np.nan), tr=np.full(n, np.nan),
                 atr=np.full(n, np.nan), dynamic_threshold=np.full(n, np.nan), trend_state=np.full(n, np.nan)
            )
            # エラー発生時はNoneを返し、キャッシュもクリア
            self._result = None
            self._values = np.full(n, np.nan)
            self._data_hash = None # ハッシュもクリアして次回再計算を強制
            return empty_result # 結果クラスの空インスタンスを返す

    # --- Getter Methods ---
    def get_result(self) -> Optional[XTrendIndexResult]:
        """計算結果全体を取得する"""
        return self._result

    def get_dominant_cycle(self) -> np.ndarray:
        """チョピネス期間の計算に使用したドミナントサイクル値を取得する"""
        if self._result is None: return np.array([])
        return self._result.dominant_cycle

    def get_dynamic_atr_period(self) -> np.ndarray:
        """CATRが計算した動的ATR期間を取得する"""
        if self._result is None: return np.array([])
        return self._result.dynamic_atr_period

    def get_efficiency_ratio(self) -> np.ndarray:
        """サイクル効率比（CER）を取得する"""
        if self._result is None: return np.array([])
        return self._result.er

    def get_stddev_factor(self) -> np.ndarray:
        """標準偏差係数の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.stddev_factor

    def get_choppiness_index(self) -> np.ndarray:
        """チョピネス指数の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.choppiness_index

    def get_range_index(self) -> np.ndarray:
        """レンジインデックスの値を取得する (0-100)"""
        if self._result is None: return np.array([])
        return self._result.range_index

    def get_true_range(self) -> np.ndarray:
        """True Rangeの値を取得する"""
        if self._result is None: return np.array([])
        return self._result.tr

    def get_absolute_atr(self) -> np.ndarray:
        """CATR（金額ベース）の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.atr

    def get_dynamic_threshold(self) -> np.ndarray:
        """動的しきい値を取得する"""
        if self._result is None: return np.array([])
        return self._result.dynamic_threshold

    def get_trend_state(self) -> np.ndarray:
        """トレンド状態を取得する（1=トレンド、0=レンジ、NaN=不明）"""
        if self._result is None: return np.array([])
        return self._result.trend_state

    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        self.dc_detector.reset()
        self.cycle_er.reset()
        self.c_atr_indicator.reset()
        self._result = None
        self._data_hash = None 