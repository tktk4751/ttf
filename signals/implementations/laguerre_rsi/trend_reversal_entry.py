#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.laguerre_rsi import LaguerreRSI


@njit(fastmath=True, parallel=True)
def calculate_trend_reversal_signals(
    lrsi_values: np.ndarray, 
    buy_band: float = 0.2,
    sell_band: float = 0.8
) -> np.ndarray:
    """
    ラゲールRSIトレンドリバーサルシグナルを計算する（高速化版）
    
    ロジック（パインスクリプトのreverse=trueに対応）:
    - RSI < buy_band (0.2): ロングシグナル (1) - 売られすぎからの反転
    - RSI > sell_band (0.8): ショートシグナル (-1) - 買われすぎからの反転
    - それ以外: 前回のポジションを維持
    
    Args:
        lrsi_values: ラゲールRSI値の配列
        buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
        sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(lrsi_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前回のポジション状態を保持するための変数
    prev_position = 0
    
    # 並列処理化
    for i in prange(length):
        # ラゲールRSI値が有効かチェック
        if np.isnan(lrsi_values[i]):
            signals[i] = 0
            continue
            
        current_rsi = lrsi_values[i]
        
        # リバーサルロジック: 
        # RSI < buy_band (売られすぎ): ロングエントリー（反転狙い）
        # RSI > sell_band (買われすぎ): ショートエントリー（反転狙い）
        if current_rsi < buy_band:
            # 売られすぎ水準: ロングエントリー（リバーサル）
            signals[i] = 1
            prev_position = 1
        elif current_rsi > sell_band:
            # 買われすぎ水準: ショートエントリー（リバーサル）
            signals[i] = -1
            prev_position = -1
        else:
            # 閾値内の場合は前回ポジションを維持
            signals[i] = prev_position
    
    return signals


@njit(fastmath=True)
def calculate_crossover_trend_reversal_signals(
    lrsi_values: np.ndarray, 
    buy_band: float = 0.2,
    sell_band: float = 0.8
) -> np.ndarray:
    """
    ラゲールRSIクロスオーバーベースのトレンドリバーサルシグナルを計算する（高速化版）
    
    ロジック:
    - RSIが閾値を下抜けまたは上抜けした瞬間にのみシグナル発生
    - RSI < buy_band への下抜け: ロングシグナル (1) - 売られすぎ反転狙い
    - RSI > sell_band への上抜け: ショートシグナル (-1) - 買われすぎ反転狙い
    
    Args:
        lrsi_values: ラゲールRSI値の配列
        buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
        sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(lrsi_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if np.isnan(lrsi_values[i]) or np.isnan(lrsi_values[i-1]):
            signals[i] = 0
            continue
            
        prev_rsi = lrsi_values[i-1]
        curr_rsi = lrsi_values[i]
        
        # 売られすぎ水準の下抜け: 前回 >= buy_band、今回 < buy_band
        if prev_rsi >= buy_band and curr_rsi < buy_band:
            signals[i] = 1  # ロングエントリー（リバーサル）
        # 買われすぎ水準の上抜け: 前回 <= sell_band、今回 > sell_band
        elif prev_rsi <= sell_band and curr_rsi > sell_band:
            signals[i] = -1  # ショートエントリー（リバーサル）
    
    return signals


@njit(fastmath=True)
def calculate_mean_reversion_signals(
    lrsi_values: np.ndarray, 
    buy_band: float = 0.3,
    sell_band: float = 0.7,
    center_line: float = 0.5
) -> np.ndarray:
    """
    ラゲールRSI平均回帰シグナルを計算する（高速化版）
    
    ロジック:
    - RSI < buy_bandかつ上昇中: ロングシグナル
    - RSI > sell_bandかつ下降中: ショートシグナル
    - 中心線付近での反転を捉える
    
    Args:
        lrsi_values: ラゲールRSI値の配列
        buy_band: 買い閾値（デフォルト: 0.3）
        sell_band: 売り閾値（デフォルト: 0.7）
        center_line: 中心線（デフォルト: 0.5）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(lrsi_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較で方向性を確認
    for i in range(2, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(lrsi_values[i]) or np.isnan(lrsi_values[i-1]) or 
            np.isnan(lrsi_values[i-2])):
            signals[i] = 0
            continue
            
        prev_rsi = lrsi_values[i-1]
        curr_rsi = lrsi_values[i]
        prev2_rsi = lrsi_values[i-2]
        
        # 上昇傾向の確認: 2期間前 < 前期間 < 現在
        is_rising = (prev2_rsi < prev_rsi) and (prev_rsi < curr_rsi)
        # 下降傾向の確認: 2期間前 > 前期間 > 現在  
        is_falling = (prev2_rsi > prev_rsi) and (prev_rsi > curr_rsi)
        
        # 平均回帰シグナル
        # RSIが低水準で上昇に転じた場合: ロングシグナル（反転狙い）
        if curr_rsi < buy_band and is_rising:
            signals[i] = 1
        # RSIが高水準で下降に転じた場合: ショートシグナル（反転狙い）
        elif curr_rsi > sell_band and is_falling:
            signals[i] = -1
    
    return signals


class LaguerreRSITrendReversalEntrySignal(BaseSignal, IEntrySignal):
    """
    ラゲールRSI（Laguerre-based RSI）トレンドリバーサルエントリーシグナル
    
    特徴:
    - John Ehlers's Laguerre transform filterベースのRSI指標を使用
    - 従来のRSIよりも価格変動に敏感で、短期間のデータでも効果的
    - トレンドリバーサル戦略: RSIの極値からの反転を狙う
    - オプションでルーフィングフィルターによる前処理
    
    シグナル条件（パインスクリプトのreverse=true仕様準拠）:
    - position_mode=True: RSI < 0.2: ロング, RSI > 0.8: ショート, それ以外: 前回ポジション維持
    - position_mode=False: 閾値クロスオーバーでのみシグナル発生
    - mean_reversion_mode=True: 平均回帰モード（傾向変化を考慮）
    
    パラメータ:
    - gamma: ラゲール変換のガンマ値（0.0-0.9、感度調整）
    - buy_band/sell_band: エントリー閾値（リバーサル用、デフォルト: 0.2/0.8）
    - ルーフィングフィルター設定（オプション）
    """
    
    def __init__(
        self,
        # ラゲールRSIパラメータ
        gamma: float = 0.5,                      # ガンマパラメータ
        src_type: str = 'close',                 # ソースタイプ
        # シグナル閾値（リバーサル用）
        buy_band: float = 0.2,                   # 買い閾値（売られすぎ水準）
        sell_band: float = 0.8,                  # 売り閾値（買われすぎ水準）
        # ルーフィングフィルターパラメータ（オプション）
        use_roofing_filter: bool = False,        # ルーフィングフィルターを使用するか
        roofing_hp_cutoff: float = 48.0,         # ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: float = 10.0,      # ルーフィングフィルターのSuperSmootherバンドエッジ
        # シグナル設定
        position_mode: bool = True,              # ポジション維持モード(True)またはクロスオーバーモード(False)
        mean_reversion_mode: bool = False        # 平均回帰モード（傾向変化を考慮）
    ):
        """
        初期化
        
        Args:
            gamma: ガンマパラメータ（デフォルト: 0.5、パインスクリプト仕様）
            src_type: ソースタイプ（デフォルト: 'close'、パインスクリプト仕様）
            buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
            sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: False）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
            mean_reversion_mode: 平均回帰モード（傾向変化を考慮、デフォルト: False）
        """
        if mean_reversion_mode:
            signal_type = "TrendReversal_MeanReversion"
        elif position_mode:
            signal_type = "TrendReversal_Position"
        else:
            signal_type = "TrendReversal_Crossover"
            
        roofing_str = f"_roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})" if use_roofing_filter else ""
        
        super().__init__(
            f"LaguerreRSI{signal_type}EntrySignal(gamma={gamma}, {src_type}, buy={buy_band}, sell={sell_band}{roofing_str})"
        )
        
        # パラメータ検証
        if not 0.0 <= gamma <= 0.9:
            raise ValueError("gammaは0.0から0.9の範囲である必要があります")
        if not 0.0 <= buy_band <= 1.0 or not 0.0 <= sell_band <= 1.0:
            raise ValueError("buy_bandとsell_bandは0.0から1.0の範囲である必要があります")
        if buy_band >= sell_band:
            raise ValueError("リバーサル用ではbuy_bandはsell_bandより小さい必要があります")
        
        # パラメータの保存
        self._params = {
            'gamma': gamma,
            'src_type': src_type,
            'buy_band': buy_band,
            'sell_band': sell_band,
            'use_roofing_filter': use_roofing_filter,
            'roofing_hp_cutoff': roofing_hp_cutoff,
            'roofing_ss_band_edge': roofing_ss_band_edge,
            'position_mode': position_mode,
            'mean_reversion_mode': mean_reversion_mode
        }
        
        self.buy_band = buy_band
        self.sell_band = sell_band
        self.position_mode = position_mode
        self.mean_reversion_mode = mean_reversion_mode
        
        # ラゲールRSIインジケーターの初期化
        self.laguerre_rsi = LaguerreRSI(
            gamma=gamma,
            src_type=src_type,
            use_roofing_filter=use_roofing_filter,
            roofing_hp_cutoff=roofing_hp_cutoff,
            roofing_ss_band_edge=roofing_ss_band_edge
        )
        
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
                
            # ラゲールRSIの計算
            lrsi_result = self.laguerre_rsi.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if lrsi_result is None or len(lrsi_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # ラゲールRSI値の取得
            lrsi_values = lrsi_result.values
            
            # シグナルの計算（モード別）
            if self.mean_reversion_mode:
                # 平均回帰モード
                signals = calculate_mean_reversion_signals(
                    lrsi_values,
                    self.buy_band,
                    self.sell_band
                )
            elif self.position_mode:
                # ポジション維持モード（パインスクリプトreverse仕様）
                signals = calculate_trend_reversal_signals(
                    lrsi_values,
                    self.buy_band,
                    self.sell_band
                )
            else:
                # クロスオーバーモード
                signals = calculate_crossover_trend_reversal_signals(
                    lrsi_values,
                    self.buy_band,
                    self.sell_band
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"LaguerreRSITrendReversalEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_lrsi_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ラゲールRSI値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ラゲールRSI値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_values()
    
    def get_l0_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        L0値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: L0値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_l0_values()
    
    def get_l1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        L1値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: L1値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_l1_values()
    
    def get_l2_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        L2値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: L2値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_l2_values()
    
    def get_l3_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        L3値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: L3値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_l3_values()
    
    def get_cu_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CU値（上昇累積）を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CU値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_cu_values()
    
    def get_cd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CD値（下降累積）を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CD値
        """
        if data is not None:
            self.generate(data)
            
        return self.laguerre_rsi.get_cd_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.laguerre_rsi.reset() if hasattr(self.laguerre_rsi, 'reset') else None
        self._signals_cache = {}


# 便利関数
def create_laguerre_rsi_trend_reversal_signal(
    gamma: float = 0.5,
    src_type: str = 'close',
    buy_band: float = 0.2,
    sell_band: float = 0.8,
    use_roofing_filter: bool = False,
    roofing_hp_cutoff: float = 48.0,
    roofing_ss_band_edge: float = 10.0,
    position_mode: bool = True,
    mean_reversion_mode: bool = False
) -> LaguerreRSITrendReversalEntrySignal:
    """
    ラゲールRSIトレンドリバーサルエントリーシグナルを作成する便利関数
    
    Args:
        gamma: ガンマパラメータ（デフォルト: 0.5）
        src_type: ソースタイプ（デフォルト: 'close'）
        buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
        sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
        use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: False）
        roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
        roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
        position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
        mean_reversion_mode: 平均回帰モード（傾向変化を考慮、デフォルト: False）
        
    Returns:
        LaguerreRSITrendReversalEntrySignal: 設定済みのシグナルインスタンス
    """
    return LaguerreRSITrendReversalEntrySignal(
        gamma=gamma,
        src_type=src_type,
        buy_band=buy_band,
        sell_band=sell_band,
        use_roofing_filter=use_roofing_filter,
        roofing_hp_cutoff=roofing_hp_cutoff,
        roofing_ss_band_edge=roofing_ss_band_edge,
        position_mode=position_mode,
        mean_reversion_mode=mean_reversion_mode
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ラゲールRSIトレンドリバーサルエントリーシグナルのテスト ===")
    
    # テストデータ生成（ボラティリティの高いレンジ相場）
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # 振動の大きいレンジ相場データ
    prices = [base_price]
    for i in range(1, length):
        # サイン波ベースの変動 + ランダムノイズ
        cycle_pos = (i / 20.0) * 2 * np.pi
        cycle_component = np.sin(cycle_pos) * 0.02  # 2%の周期変動
        random_component = np.random.normal(0, 0.015)  # 1.5%のランダム変動
        
        change = cycle_component + random_component
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
    print(f"価格標準偏差: {df['close'].std():.2f}")
    
    # ポジション維持モードのテスト（リバーサル）
    print("\n=== ポジション維持モードのテスト（リバーサル） ===")
    signal_position = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.2,  # 売られすぎでロング
        sell_band=0.8,  # 買われすぎでショート
        position_mode=True
    )
    
    signals_pos = signal_position.generate(df)
    lrsi_values = signal_position.get_lrsi_values()
    
    long_signals = np.sum(signals_pos == 1)
    short_signals = np.sum(signals_pos == -1)
    no_signals = np.sum(signals_pos == 0)
    
    print(f"ロングシグナル: {long_signals}")
    print(f"ショートシグナル: {short_signals}")
    print(f"シグナルなし: {no_signals}")
    print(f"平均ラゲールRSI: {np.nanmean(lrsi_values):.4f}")
    print(f"RSI範囲: {np.nanmin(lrsi_values):.4f} - {np.nanmax(lrsi_values):.4f}")
    
    # クロスオーバーモードのテスト（リバーサル）
    print("\n=== クロスオーバーモードのテスト（リバーサル） ===")
    signal_crossover = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.2,
        sell_band=0.8,
        position_mode=False
    )
    
    signals_cross = signal_crossover.generate(df)
    
    long_cross = np.sum(signals_cross == 1)
    short_cross = np.sum(signals_cross == -1)
    no_cross = np.sum(signals_cross == 0)
    
    print(f"ロングシグナル: {long_cross}")
    print(f"ショートシグナル: {short_cross}")
    print(f"シグナルなし: {no_cross}")
    
    # 平均回帰モードのテスト
    print("\n=== 平均回帰モードのテスト ===")
    signal_meanrev = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.3,  # より緩い閾値
        sell_band=0.7,
        position_mode=True,
        mean_reversion_mode=True
    )
    
    signals_meanrev = signal_meanrev.generate(df)
    
    long_meanrev = np.sum(signals_meanrev == 1)
    short_meanrev = np.sum(signals_meanrev == -1)
    no_meanrev = np.sum(signals_meanrev == 0)
    
    print(f"ロングシグナル: {long_meanrev}")
    print(f"ショートシグナル: {short_meanrev}")
    print(f"シグナルなし: {no_meanrev}")
    
    # 閾値別の統計
    print("\n=== 閾値別統計 ===")
    overbought_count = np.sum(lrsi_values > 0.8)
    oversold_count = np.sum(lrsi_values < 0.2)
    neutral_count = np.sum((lrsi_values >= 0.2) & (lrsi_values <= 0.8))
    
    print(f"買われすぎ (>0.8): {overbought_count} ({overbought_count/len(lrsi_values)*100:.1f}%)")
    print(f"売られすぎ (<0.2): {oversold_count} ({oversold_count/len(lrsi_values)*100:.1f}%)")
    print(f"中立 (0.2-0.8): {neutral_count} ({neutral_count/len(lrsi_values)*100:.1f}%)")
    
    # シグナル品質評価
    print("\n=== シグナル品質評価 ===")
    
    # ロングシグナル時のRSI分析
    long_positions = signals_pos == 1
    if np.sum(long_positions) > 0:
        long_rsi_avg = np.nanmean(lrsi_values[long_positions])
        print(f"ロングシグナル時の平均RSI: {long_rsi_avg:.4f}")
    
    # ショートシグナル時のRSI分析  
    short_positions = signals_pos == -1
    if np.sum(short_positions) > 0:
        short_rsi_avg = np.nanmean(lrsi_values[short_positions])
        print(f"ショートシグナル時の平均RSI: {short_rsi_avg:.4f}")
    
    print("\n=== テスト完了 ===")