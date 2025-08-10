#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from indicators.kalman.unified_kalman import UnifiedKalman


@jit(nopython=True)
def calculate_hody_dce_numba(
    price: np.ndarray,
    lp_period: int = 10,
    hp_period: int = 48,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    バンドパスフィルターを使った拡張ホモダイン判別機によるドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        lp_period: ローパスフィルターの期間
        hp_period: ハイパスフィルターの期間
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化 - Pineスクリプトと同じ形式
    alpha1 = 0.0
    hp = np.zeros(n)
    a1 = 0.0
    b1 = 0.0
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    filt = np.zeros(n)
    i_peak = np.zeros(n)
    q_peak = np.zeros(n)
    real = np.zeros(n)
    quad = np.zeros(n)
    imag = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    dom_cycle = np.zeros(n)
    dc = np.zeros(n)
    
    # メインループ
    for i in range(n):
        # Highpass filter cyclic components whose periods are shorter than HPPeriod bars
        alpha1 = (np.cos(0.707 * 2 * pi / hp_period) + np.sin(0.707 * 2 * pi / hp_period) - 1) / np.cos(0.707 * 2 * pi / hp_period)
        
        # HP計算
        price_1 = price[i-1] if i >= 1 else 0.0
        price_2 = price[i-2] if i >= 2 else 0.0
        hp_1 = hp[i-1] if i >= 1 else 0.0
        hp_2 = hp[i-2] if i >= 2 else 0.0
        
        hp[i] = ((1 - alpha1 / 2) * (1 - alpha1 / 2) * (price[i] - 2 * price_1 + price_2) + 
                 2 * (1 - alpha1) * hp_1 - (1 - alpha1) * (1 - alpha1) * hp_2)
        
        # Smooth with a Super Smoother Filter
        a1 = np.exp(-1.414 * 3.14159 / lp_period)  # PineスクリプトでのpiValue（3.14159）を使用
        b1 = 2 * a1 * np.cos(1.414 * pi / lp_period)  # ここはpiを使用
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        # Filt計算
        hp_1 = hp[i-1] if i >= 1 else 0.0
        filt_1 = filt[i-1] if i >= 1 else 0.0
        filt_2 = filt[i-2] if i >= 2 else 0.0
        
        filt[i] = c1 * (hp[i] + hp_1) / 2 + c2 * filt_1 + c3 * filt_2
        
        # IPeak計算
        i_peak_1 = i_peak[i-1] if i >= 1 else 0.0
        i_peak[i] = 0.991 * i_peak_1
        if abs(filt[i]) > i_peak[i]:
            i_peak[i] = abs(filt[i])
        
        # Real計算
        if i_peak[i] != 0:
            real[i] = filt[i] / i_peak[i]
        else:
            real[i] = 0.0
        
        # Quad計算
        real_1 = real[i-1] if i >= 1 else 0.0
        quad[i] = real[i] - real_1
        
        # QPeak計算
        q_peak_1 = q_peak[i-1] if i >= 1 else 0.0
        q_peak[i] = 0.991 * q_peak_1
        if abs(quad[i]) > q_peak[i]:
            q_peak[i] = abs(quad[i])
        
        # Imag計算
        if q_peak[i] != 0:
            imag[i] = quad[i] / q_peak[i]
        else:
            imag[i] = 0.0
        
        # ホモダイン判別器の実装 - Pineスクリプトと同じロジック
        # Re := Real * nz(Real[1]) + Imag * nz(Imag[1])
        # Im := nz(Real[1]) * Imag - Real * nz(Imag[1])
        imag_1 = imag[i-1] if i >= 1 else 0.0
        
        re[i] = real[i] * real_1 + imag[i] * imag_1
        im[i] = real_1 * imag[i] - real[i] * imag_1
        
        # Period計算 - Pineスクリプトと同じ式
        # Period := Im != 0 and Re != 0 ? 6.28318 / math.abs(Im / Re) : Period
        if im[i] != 0 and re[i] != 0:
            period[i] = 6.28318 / abs(im[i] / re[i])
        else:
            # 分母がゼロの場合は前回の値を維持
            period[i] = period[i-1] if i >= 1 else lp_period
        
        # Period制限
        if period[i] < lp_period:
            period[i] = lp_period
        if period[i] > hp_period:
            period[i] = hp_period
        
        # DomCycle計算 - スーパースムーサーフィルター
        period_1 = period[i-1] if i >= 1 else 0.0
        dom_cycle_1 = dom_cycle[i-1] if i >= 1 else 0.0
        dom_cycle_2 = dom_cycle[i-2] if i >= 2 else 0.0
        
        dom_cycle[i] = c1 * (period[i] + period_1) / 2 + c2 * dom_cycle_1 + c3 * dom_cycle_2
        
        # DC計算 - Pineスクリプトと同じロジック
        cycle_value = np.ceil(dom_cycle[i] * cycle_part)
        if cycle_value > max_output:
            dc[i] = max_output
        elif cycle_value < min_output:
            dc[i] = min_output
        else:
            dc[i] = cycle_value
    
    # 生の周期値と平滑化周期値を保存
    raw_period = np.copy(period)
    smooth_period = np.copy(dom_cycle)
    
    return dc, raw_period, smooth_period


class EhlersHoDyDCE(EhlersDominantCycle):
    """
    エーラーズの拡張ホモダイン判別機（Homodyne Discriminator with Bandpass Filter）
    
    このアルゴリズムはバンドパスフィルターを使用して価格データをフィルタリングし、
    ホモダイン判別機を使用して周期を検出します。バンドパスフィルターの使用により
    ノイズ除去性能が向上しています。
    
    特徴:
    - バンドパスフィルターによる高精度なノイズ除去
    - 適応型フィルターでさまざまな市場状況に対応
    - 周期の検出範囲を正確に制御可能
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        lp_period: int = 10,
        hp_period: int = 48,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ
        
        Args:
            lp_period: ローパスフィルターの期間（デフォルト: 10）
            hp_period: ハイパスフィルターの期間（デフォルト: 48）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            use_kalman_filter: Kalmanフィルターを使用するか（デフォルト: False）
            kalman_filter_type: Kalmanフィルターのタイプ（デフォルト: 'adaptive'）
        """
        super().__init__(
            f"EhlersHoDyDCE({lp_period}, {hp_period}, {cycle_part})",
            cycle_part,
            hp_period,  # max_cycle
            lp_period,  # min_cycle
            max_output,
            min_output
        )
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        
        # ソースタイプを保存
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # Kalmanフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = UnifiedKalman(
                filter_type=kalman_filter_type,
                src_type='close'  # 内部的にはcloseで使用
            )
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
        """
        指定されたソースタイプに基づいて価格データを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        
        Returns:
            計算された価格データの配列
        """
        if isinstance(data, pd.DataFrame):
            if src_type == 'close':
                if 'close' in data.columns:
                    return data['close'].values
                elif 'Close' in data.columns:
                    return data['Close'].values
                else:
                    raise ValueError("DataFrameには'close'または'Close'カラムが必要です")
            
            elif src_type == 'hlc3':
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    return (data['high'] + data['low'] + data['close']).values / 3
                elif all(col in data.columns for col in ['High', 'Low', 'Close']):
                    return (data['High'] + data['Low'] + data['Close']).values / 3
                else:
                    raise ValueError("hlc3の計算には'high', 'low', 'close'カラムが必要です")
            
            elif src_type == 'hl2':
                if all(col in data.columns for col in ['high', 'low']):
                    return (data['high'] + data['low']).values / 2
                elif all(col in data.columns for col in ['High', 'Low']):
                    return (data['High'] + data['Low']).values / 2
                else:
                    raise ValueError("hl2の計算には'high', 'low'カラムが必要です")
            
            elif src_type == 'ohlc4':
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    return (data['open'] + data['high'] + data['low'] + data['close']).values / 4
                elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    return (data['Open'] + data['High'] + data['Low'] + data['Close']).values / 4
                else:
                    raise ValueError("ohlc4の計算には'open', 'high', 'low', 'close'カラムが必要です")
            
            else:
                raise ValueError(f"無効なソースタイプです: {src_type}")
        
        else:  # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                if src_type == 'close':
                    return data[:, 3]  # close
                elif src_type == 'hlc3':
                    return (data[:, 1] + data[:, 2] + data[:, 3]) / 3  # high, low, close
                elif src_type == 'hl2':
                    return (data[:, 1] + data[:, 2]) / 2  # high, low
                elif src_type == 'ohlc4':
                    return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4  # open, high, low, close
                else:
                    raise ValueError(f"無効なソースタイプです: {src_type}")
            else:
                return data  # 1次元配列として扱う
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        拡張ホモダイン判別機を使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、指定したソースタイプに必要なカラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # ソースタイプに基づいて価格データを取得
            price = self.calculate_source_values(data, self.src_type)
            
            # Numba関数を使用してドミナントサイクルを計算
            dom_cycle, raw_period, smooth_period = calculate_hody_dce_numba(
                price,
                self.lp_period,
                self.hp_period,
                self.cycle_part,
                self.max_output,
                self.min_output
            )
            
            # 結果を保存
            self._result = DominantCycleResult(
                values=dom_cycle,
                raw_period=raw_period,
                smooth_period=smooth_period
            )
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersHoDyDCE計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 