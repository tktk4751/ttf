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
def calculate_trend_follow_signals(
    lrsi_values: np.ndarray, 
    buy_band: float = 0.8,
    sell_band: float = 0.2
) -> np.ndarray:
    """
    ラゲールRSIトレンドフォローシグナルを計算する（高速化版）
    
    ロジック（パインスクリプト仕様）:
    - RSI > buy_band (0.8): ロングシグナル (1)
    - RSI < sell_band (0.2): ショートシグナル (-1)  
    - それ以外: 前回のポジションを維持
    
    Args:
        lrsi_values: ラゲールRSI値の配列
        buy_band: 買い閾値（デフォルト: 0.8）
        sell_band: 売り閾値（デフォルト: 0.2）
    
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
        
        # パインスクリプトのロジック: 
        # pos = iff(nRes > BuyBand, 1, iff(nRes < SellBand, -1, nz(pos[1], 0)))
        if current_rsi > buy_band:
            # 買われすぎ水準を超えた場合: ロングエントリー（トレンドフォロー）
            signals[i] = 1
            prev_position = 1
        elif current_rsi < sell_band:
            # 売られすぎ水準を下回った場合: ショートエントリー（トレンドフォロー）
            signals[i] = -1
            prev_position = -1
        else:
            # 閾値内の場合は前回ポジションを維持
            signals[i] = prev_position
    
    return signals


@njit(fastmath=True)
def calculate_crossover_trend_follow_signals(
    lrsi_values: np.ndarray, 
    buy_band: float = 0.8,
    sell_band: float = 0.2
) -> np.ndarray:
    """
    ラゲールRSIクロスオーバーベースのトレンドフォローシグナルを計算する（高速化版）
    
    ロジック:
    - RSIが閾値を上抜けした瞬間にのみシグナル発生
    - RSI > buy_band への上抜け: ロングシグナル (1)
    - RSI < sell_band への下抜け: ショートシグナル (-1)
    
    Args:
        lrsi_values: ラゲールRSI値の配列
        buy_band: 買い閾値（デフォルト: 0.8）
        sell_band: 売り閾値（デフォルト: 0.2）
    
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
        
        # 買い閾値の上抜け: 前回 <= buy_band、今回 > buy_band
        if prev_rsi <= buy_band and curr_rsi > buy_band:
            signals[i] = 1
        # 売り閾値の下抜け: 前回 >= sell_band、今回 < sell_band
        elif prev_rsi >= sell_band and curr_rsi < sell_band:
            signals[i] = -1
    
    return signals


class LaguerreRSITrendFollowEntrySignal(BaseSignal, IEntrySignal):
    """
    ラゲールRSI（Laguerre-based RSI）トレンドフォローエントリーシグナル
    
    特徴:
    - John Ehlers's Laguerre transform filterベースのRSI指標を使用
    - 従来のRSIよりも価格変動に敏感で、短期間のデータでも効果的
    - トレンドフォロー戦略: RSIが閾値を超えた方向にエントリー
    - オプションでルーフィングフィルターによる前処理
    
    シグナル条件（パインスクリプト仕様準拠）:
    - position_mode=True: RSI > 0.8: ロング, RSI < 0.2: ショート, それ以外: 前回ポジション維持
    - position_mode=False: 閾値クロスオーバーでのみシグナル発生
    
    パラメータ:
    - gamma: ラゲール変換のガンマ値（0.0-0.9、感度調整）
    - buy_band/sell_band: エントリー閾値（デフォルト: 0.8/0.2）
    - ルーフィングフィルター設定（オプション）
    """
    
    def __init__(
        self,
        # ラゲールRSIパラメータ
        gamma: float = 0.5,                      # ガンマパラメータ
        src_type: str = 'close',                 # ソースタイプ
        # シグナル閾値
        buy_band: float = 0.8,                   # 買い閾値
        sell_band: float = 0.2,                  # 売り閾値
        # ルーフィングフィルターパラメータ（オプション）
        use_roofing_filter: bool = False,        # ルーフィングフィルターを使用するか
        roofing_hp_cutoff: float = 48.0,         # ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: float = 10.0,      # ルーフィングフィルターのSuperSmootherバンドエッジ
        # シグナル設定
        position_mode: bool = True               # ポジション維持モード(True)またはクロスオーバーモード(False)
    ):
        """
        初期化
        
        Args:
            gamma: ガンマパラメータ（デフォルト: 0.5、パインスクリプト仕様）
            src_type: ソースタイプ（デフォルト: 'close'、パインスクリプト仕様）
            buy_band: 買い閾値（デフォルト: 0.8、パインスクリプト仕様）
            sell_band: 売り閾値（デフォルト: 0.2、パインスクリプト仕様）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: False）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
        """
        signal_type = "TrendFollow_Position" if position_mode else "TrendFollow_Crossover"
        roofing_str = f"_roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})" if use_roofing_filter else ""
        
        super().__init__(
            f"LaguerreRSI{signal_type}EntrySignal(gamma={gamma}, {src_type}, buy={buy_band}, sell={sell_band}{roofing_str})"
        )
        
        # パラメータ検証
        if not 0.0 <= gamma <= 0.999:
            raise ValueError("gammaは0.0から0.999の範囲である必要があります")
        if not 0.0 <= buy_band <= 1.0 or not 0.0 <= sell_band <= 1.0:
            raise ValueError("buy_bandとsell_bandは0.0から1.0の範囲である必要があります")
        if buy_band <= sell_band:
            raise ValueError("buy_bandはsell_bandより大きい必要があります")
        
        # パラメータの保存
        self._params = {
            'gamma': gamma,
            'src_type': src_type,
            'buy_band': buy_band,
            'sell_band': sell_band,
            'use_roofing_filter': use_roofing_filter,
            'roofing_hp_cutoff': roofing_hp_cutoff,
            'roofing_ss_band_edge': roofing_ss_band_edge,
            'position_mode': position_mode
        }
        
        self.buy_band = buy_band
        self.sell_band = sell_band
        self.position_mode = position_mode
        
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
            
            # シグナルの計算（ポジション維持またはクロスオーバー）
            if self.position_mode:
                # ポジション維持モード（パインスクリプト仕様）
                signals = calculate_trend_follow_signals(
                    lrsi_values,
                    self.buy_band,
                    self.sell_band
                )
            else:
                # クロスオーバーモード
                signals = calculate_crossover_trend_follow_signals(
                    lrsi_values,
                    self.buy_band,
                    self.sell_band
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"LaguerreRSITrendFollowEntrySignal計算中にエラー: {str(e)}")
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
def create_laguerre_rsi_trend_follow_signal(
    gamma: float = 0.5,
    src_type: str = 'close',
    buy_band: float = 0.8,
    sell_band: float = 0.2,
    use_roofing_filter: bool = False,
    roofing_hp_cutoff: float = 48.0,
    roofing_ss_band_edge: float = 10.0,
    position_mode: bool = True
) -> LaguerreRSITrendFollowEntrySignal:
    """
    ラゲールRSIトレンドフォローエントリーシグナルを作成する便利関数
    
    Args:
        gamma: ガンマパラメータ（デフォルト: 0.5）
        src_type: ソースタイプ（デフォルト: 'close'）
        buy_band: 買い閾値（デフォルト: 0.8）
        sell_band: 売り閾値（デフォルト: 0.2）
        use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: False）
        roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
        roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
        position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
        
    Returns:
        LaguerreRSITrendFollowEntrySignal: 設定済みのシグナルインスタンス
    """
    return LaguerreRSITrendFollowEntrySignal(
        gamma=gamma,
        src_type=src_type,
        buy_band=buy_band,
        sell_band=sell_band,
        use_roofing_filter=use_roofing_filter,
        roofing_hp_cutoff=roofing_hp_cutoff,
        roofing_ss_band_edge=roofing_ss_band_edge,
        position_mode=position_mode
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ラゲールRSIトレンドフォローエントリーシグナルのテスト ===")
    
    # テストデータ生成（トレンド性のあるデータ）
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # 上昇トレンド → レンジ → 下降トレンド
    prices = [base_price]
    for i in range(1, length):
        if i < 60:  # 上昇トレンド
            change = 0.005 + np.random.normal(0, 0.01)
        elif i < 140:  # レンジ相場
            change = np.random.normal(0, 0.008)
        else:  # 下降トレンド
            change = -0.005 + np.random.normal(0, 0.01)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
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
    
    # ポジション維持モードのテスト
    print("\n=== ポジション維持モードのテスト ===")
    signal_position = LaguerreRSITrendFollowEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.8,
        sell_band=0.2,
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
    
    # クロスオーバーモードのテスト
    print("\n=== クロスオーバーモードのテスト ===")
    signal_crossover = LaguerreRSITrendFollowEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.8,
        sell_band=0.2,
        position_mode=False
    )
    
    signals_cross = signal_crossover.generate(df)
    
    long_cross = np.sum(signals_cross == 1)
    short_cross = np.sum(signals_cross == -1)
    no_cross = np.sum(signals_cross == 0)
    
    print(f"ロングシグナル: {long_cross}")
    print(f"ショートシグナル: {short_cross}")
    print(f"シグナルなし: {no_cross}")
    
    # 閾値別の統計
    print("\n=== 閾値別統計 ===")
    overbought_count = np.sum(lrsi_values > 0.8)
    oversold_count = np.sum(lrsi_values < 0.2)
    neutral_count = np.sum((lrsi_values >= 0.2) & (lrsi_values <= 0.8))
    
    print(f"買われすぎ (>0.8): {overbought_count} ({overbought_count/len(lrsi_values)*100:.1f}%)")
    print(f"売られすぎ (<0.2): {oversold_count} ({oversold_count/len(lrsi_values)*100:.1f}%)")
    print(f"中立 (0.2-0.8): {neutral_count} ({neutral_count/len(lrsi_values)*100:.1f}%)")
    
    print("\n=== テスト完了 ===")