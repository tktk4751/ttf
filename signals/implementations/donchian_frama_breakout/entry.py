#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.trend_filter.donchian_midline import DonchianMidline
from indicators.smoother.frama import FRAMA


@njit(fastmath=True)
def calculate_donchian_channel_breakout_signals(
    closes: np.ndarray,
    donchian_upper: np.ndarray,
    donchian_lower: np.ndarray,
    donchian_frama_signals: np.ndarray
) -> np.ndarray:
    """
    ドンチャンチャネルブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        closes: 終値配列
        donchian_upper: トリガー用ドンチャン上部バンド
        donchian_lower: トリガー用ドンチャン下部バンド
        donchian_frama_signals: ドンチャンFRAMA位置関係シグナル
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(closes)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(1, length):
        # 前回と現在の値が有効かチェック
        if (np.isnan(closes[i]) or np.isnan(closes[i-1]) or 
            np.isnan(donchian_upper[i-1]) or np.isnan(donchian_lower[i-1]) or
            np.isnan(donchian_frama_signals[i])):
            signals[i] = 0
            continue
        
        current_close = closes[i]
        prev_upper = donchian_upper[i-1]
        prev_lower = donchian_lower[i-1]
        frama_signal = donchian_frama_signals[i]
        
        # ドンチャンFRAMAシグナルが1（ロング傾向）の時のみブレイクアウトを考慮
        if frama_signal == 1:
            # 現在の終値が前回の上部バンドを超えた場合ロングエントリー
            if current_close > prev_upper:
                signals[i] = 1
        # ドンチャンFRAMAシグナルが-1（ショート傾向）の時のみブレイクアウトを考慮
        elif frama_signal == -1:
            # 現在の終値が前回の下部バンドを下回った場合ショートエントリー
            if current_close < prev_lower:
                signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_donchian_channel_exit_signals(
    closes: np.ndarray,
    donchian_upper: np.ndarray,
    donchian_lower: np.ndarray,
    position: int
) -> np.ndarray:
    """
    ドンチャンチャネルエグジットシグナルを計算する（高速化版）
    
    Args:
        closes: 終値配列
        donchian_upper: トリガー用ドンチャン上部バンド
        donchian_lower: トリガー用ドンチャン下部バンド
        position: ポジション方向（1: ロング, -1: ショート）
    
    Returns:
        エグジットシグナルの配列（1: エグジット, 0: ホールド）
    """
    length = len(closes)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(1, length):
        # 前回と現在の値が有効かチェック
        if (np.isnan(closes[i]) or np.isnan(donchian_upper[i-1]) or np.isnan(donchian_lower[i-1])):
            signals[i] = 0
            continue
        
        current_close = closes[i]
        prev_upper = donchian_upper[i-1]
        prev_lower = donchian_lower[i-1]
        
        if position == 1:  # ロングポジション
            # 現在の終値が前回の下部バンドを下回った場合ロング決済
            if current_close < prev_lower:
                signals[i] = 1
        elif position == -1:  # ショートポジション
            # 現在の終値が前回の上部バンドを超えた場合ショート決済
            if current_close > prev_upper:
                signals[i] = 1
    
    return signals


class DonchianFRAMABreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ドンチャンFRAMAブレイクアウトエントリーシグナル
    
    特徴:
    - 1つ目のドンチャンミッドライン: フィルター用（デフォルト期間200）
    - FRAMA: トレンド方向判定用
    - 2つ目のドンチャンチャネル: ブレイクアウトトリガー用（デフォルト期間60）
    
    ロジック:
    1. 3つのフィルター（HyperER, HyperTrendIndex, HyperADX）を通過
    2. ドンチャンFRAMA位置関係シグナルが1（ロング傾向）の時
    3. 2つ目のドンチャンの前回アッパーを現在終値が超えたらロングエントリー
    4. ドンチャンFRAMA位置関係シグナルが-1（ショート傾向）の時
    5. 2つ目のドンチャンの前回ロワーを現在終値が下回ったらショートエントリー
    """
    
    def __init__(
        self,
        # ドンチャンミッドラインパラメータ（フィルター用）
        donchian_midline_period: int = 200,
        donchian_midline_src_type: str = 'hlc3',
        
        # FRAMAパラメータ（トレンド方向判定用）
        frama_period: int = 16,
        frama_src_type: str = 'hlc3',
        frama_fc: int = 2,
        frama_sc: int = 198,
        frama_period_mode: str = 'fixed',
        
        # トリガー用ドンチャンチャネルパラメータ
        trigger_donchian_period: int = 60,
        trigger_donchian_src_type: str = 'hlc3',
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = False,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min: float = 1.0,
        frama_fc_max: float = 13.0,
        frama_sc_min: float = 60.0,
        frama_sc_max: float = 250.0,
        
        # ドンチャンミッドライン HyperER動的適応パラメータ
        donchian_midline_period_min: float = 55.0,
        donchian_midline_period_max: float = 250.0,
        
        # トリガー用ドンチャン HyperER動的適応パラメータ
        trigger_donchian_period_min: float = 20.0,
        trigger_donchian_period_max: float = 100.0,
        
        # シグナル設定
        signal_mode: str = 'position'  # 'position' または 'crossover'
    ):
        """
        初期化
        """
        # 動的適応文字列の作成
        adaptation_str = "_hyper_er" if enable_hyper_er_adaptation else ""
        
        super().__init__(
            f"DonchianFRAMABreakoutEntrySignal(midline={donchian_midline_period}({donchian_midline_src_type}), "
            f"frama={frama_period}({frama_src_type}), trigger={trigger_donchian_period}({trigger_donchian_src_type}){adaptation_str})"
        )
        
        # パラメータの保存
        self._params = {
            'donchian_midline_period': donchian_midline_period,
            'donchian_midline_src_type': donchian_midline_src_type,
            'frama_period': frama_period,
            'frama_src_type': frama_src_type,
            'frama_fc': frama_fc,
            'frama_sc': frama_sc,
            'frama_period_mode': frama_period_mode,
            'trigger_donchian_period': trigger_donchian_period,
            'trigger_donchian_src_type': trigger_donchian_src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'frama_fc_min': frama_fc_min,
            'frama_fc_max': frama_fc_max,
            'frama_sc_min': frama_sc_min,
            'frama_sc_max': frama_sc_max,
            'donchian_midline_period_min': donchian_midline_period_min,
            'donchian_midline_period_max': donchian_midline_period_max,
            'trigger_donchian_period_min': trigger_donchian_period_min,
            'trigger_donchian_period_max': trigger_donchian_period_max,
            'signal_mode': signal_mode
        }
        
        self.signal_mode = signal_mode
        
        # ドンチャンミッドラインインジケーター（フィルター用）
        self.donchian_midline = DonchianMidline(
            period=donchian_midline_period,
            src_type=donchian_midline_src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            period_min=donchian_midline_period_min,
            period_max=donchian_midline_period_max
        )
        
        # FRAMAインジケーター（トレンド方向判定用）
        self.frama = FRAMA(
            period=frama_period,
            src_type=frama_src_type,
            fc=frama_fc,
            sc=frama_sc,
            period_mode=frama_period_mode,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            fc_min=frama_fc_min,
            fc_max=frama_fc_max,
            sc_min=frama_sc_min,
            sc_max=frama_sc_max
        )
        
        # トリガー用ドンチャンチャネル
        self.trigger_donchian = DonchianMidline(
            period=trigger_donchian_period,
            src_type=trigger_donchian_src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            period_min=trigger_donchian_period_min,
            period_max=trigger_donchian_period_max
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """データハッシュを取得する"""
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
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
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
            
            # 各インジケーターを計算
            # 1. ドンチャンミッドライン（フィルター用）
            donchian_midline_result = self.donchian_midline.calculate(data)
            if donchian_midline_result is None or len(donchian_midline_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # 2. FRAMA（トレンド方向判定用）
            frama_result = self.frama.calculate(data)
            if frama_result is None or len(frama_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # 3. トリガー用ドンチャンチャネル
            trigger_donchian_result = self.trigger_donchian.calculate(data)
            if trigger_donchian_result is None or len(trigger_donchian_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ドンチャンFRAMA位置関係シグナル計算
            donchian_midline_values = donchian_midline_result.values
            frama_values = frama_result.values
            
            # データ長を合わせる
            min_length = min(len(donchian_midline_values), len(frama_values), len(data))
            if min_length == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ドンチャンFRAMAシグナル計算（位置関係またはクロスオーバー）
            if self.signal_mode == 'position':
                # 位置関係シグナル
                donchian_frama_signals = np.zeros(min_length, dtype=np.int8)
                for i in range(min_length):
                    if not np.isnan(donchian_midline_values[i]) and not np.isnan(frama_values[i]):
                        if frama_values[i] > donchian_midline_values[i]:
                            donchian_frama_signals[i] = 1
                        elif frama_values[i] < donchian_midline_values[i]:
                            donchian_frama_signals[i] = -1
            else:
                # クロスオーバーシグナル
                donchian_frama_signals = np.zeros(min_length, dtype=np.int8)
                for i in range(1, min_length):
                    if (not np.isnan(donchian_midline_values[i]) and not np.isnan(frama_values[i]) and
                        not np.isnan(donchian_midline_values[i-1]) and not np.isnan(frama_values[i-1])):
                        prev_donchian = donchian_midline_values[i-1]
                        prev_frama = frama_values[i-1]
                        curr_donchian = donchian_midline_values[i]
                        curr_frama = frama_values[i]
                        
                        # ゴールデンクロス
                        if prev_frama <= prev_donchian and curr_frama > curr_donchian:
                            donchian_frama_signals[i] = 1
                        # デッドクロス
                        elif prev_frama >= prev_donchian and curr_frama < curr_donchian:
                            donchian_frama_signals[i] = -1
            
            # 終値データ取得
            if isinstance(data, pd.DataFrame):
                closes = data['close'].values
            else:
                closes = data[:, 3]  # close price column
            
            # ブレイクアウトシグナル計算
            signals = calculate_donchian_channel_breakout_signals(
                closes[:min_length],
                trigger_donchian_result.upper_band[:min_length],
                trigger_donchian_result.lower_band[:min_length],
                donchian_frama_signals
            )
            
            # 元のデータ長に合わせて結果を拡張
            full_signals = np.zeros(len(data), dtype=np.int8)
            full_signals[:len(signals)] = signals
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = full_signals
            return full_signals
            
        except Exception as e:
            print(f"DonchianFRAMABreakoutEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int) -> np.ndarray:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: ポジション方向（1: ロング, -1: ショート）
        
        Returns:
            エグジットシグナルの配列 (1: エグジット, 0: ホールド)
        """
        try:
            # トリガー用ドンチャンチャネル計算
            trigger_donchian_result = self.trigger_donchian.calculate(data)
            if trigger_donchian_result is None or len(trigger_donchian_result.values) == 0:
                return np.zeros(len(data), dtype=np.int8)
            
            # 終値データ取得
            if isinstance(data, pd.DataFrame):
                closes = data['close'].values
            else:
                closes = data[:, 3]
            
            # エグジットシグナル計算
            exit_signals = calculate_donchian_channel_exit_signals(
                closes,
                trigger_donchian_result.upper_band,
                trigger_donchian_result.lower_band,
                position
            )
            
            return exit_signals
            
        except Exception as e:
            print(f"DonchianFRAMABreakoutEntrySignal エグジット計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_donchian_midline_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """ドンチャンミッドライン値を取得する"""
        if data is not None:
            self.generate(data)
        return self.donchian_midline.get_values() or np.array([])
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """FRAMA値を取得する"""
        if data is not None:
            self.generate(data)
        return self.frama.get_values() or np.array([])
    
    def get_trigger_donchian_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[tuple]:
        """トリガー用ドンチャンバンド（上部・下部）を取得する"""
        if data is not None:
            self.generate(data)
        
        upper = self.trigger_donchian.get_upper_band()
        lower = self.trigger_donchian.get_lower_band()
        
        if upper is not None and lower is not None:
            return (upper, lower)
        return None
    
    def reset(self) -> None:
        """シグナルの状態をリセットする"""
        super().reset()
        self.donchian_midline.reset() if hasattr(self.donchian_midline, 'reset') else None
        self.frama.reset() if hasattr(self.frama, 'reset') else None
        self.trigger_donchian.reset() if hasattr(self.trigger_donchian, 'reset') else None
        self._signals_cache = {}


# 便利関数
def create_donchian_frama_breakout_entry_signal(
    donchian_midline_period: int = 200,
    frama_period: int = 16,
    trigger_donchian_period: int = 60,
    **kwargs
) -> DonchianFRAMABreakoutEntrySignal:
    """
    ドンチャンFRAMAブレイクアウトエントリーシグナルを作成する便利関数
    """
    return DonchianFRAMABreakoutEntrySignal(
        donchian_midline_period=donchian_midline_period,
        frama_period=frama_period,
        trigger_donchian_period=trigger_donchian_period,
        **kwargs
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ドンチャンFRAMAブレイクアウトエントリーシグナルのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 300
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.002 + np.random.normal(0, 0.008)
        elif i < 150:  # レンジ相場
            change = np.random.normal(0, 0.010)
        elif i < 250:  # 強い上昇トレンド
            change = 0.004 + np.random.normal(0, 0.006)
        else:  # 下降トレンド  
            change = -0.003 + np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # ブレイクアウトシグナルをテスト
    print("\nドンチャンFRAMAブレイクアウトシグナルをテスト中...")
    breakout_signal = DonchianFRAMABreakoutEntrySignal(
        donchian_midline_period=200,
        frama_period=16,
        trigger_donchian_period=60
    )
    
    breakout_signals = breakout_signal.generate(df)
    
    long_signals = np.sum(breakout_signals == 1)
    short_signals = np.sum(breakout_signals == -1)
    neutral_signals = np.sum(breakout_signals == 0)
    
    print(f"  ロングシグナル: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"  ショートシグナル: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"  中立: {neutral_signals} ({neutral_signals/len(df)*100:.1f}%)")
    
    # エグジットシグナルをテスト
    print("\nエグジットシグナルをテスト中...")
    exit_signals_long = breakout_signal.generate_exit_signals(df, position=1)
    exit_signals_short = breakout_signal.generate_exit_signals(df, position=-1)
    
    long_exits = np.sum(exit_signals_long == 1)
    short_exits = np.sum(exit_signals_short == 1)
    
    print(f"  ロングエグジット: {long_exits} ({long_exits/len(df)*100:.1f}%)")
    print(f"  ショートエグジット: {short_exits} ({short_exits/len(df)*100:.1f}%)")
    
    print("\n=== テスト完了 ===")