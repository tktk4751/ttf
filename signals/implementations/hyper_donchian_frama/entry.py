#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.trend_filter.hyper_donchian import HyperDonchian
from indicators.smoother.frama import FRAMA


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    hyper_donchian_values: np.ndarray, 
    frama_values: np.ndarray
) -> np.ndarray:
    """
    HyperドンチャンミッドラインとFRAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        hyper_donchian_values: Hyperドンチャンミッドライン値の配列
        frama_values: FRAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(hyper_donchian_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # Hyperドンチャンミッドライン値とFRAMA値が有効かチェック
        if np.isnan(hyper_donchian_values[i]) or np.isnan(frama_values[i]):
            signals[i] = 0
            continue
            
        # FRAMA > Hyperドンチャンミッドライン: ロングシグナル
        if frama_values[i] > hyper_donchian_values[i]:
            signals[i] = 1
        # FRAMA < Hyperドンチャンミッドライン: ショートシグナル
        elif frama_values[i] < hyper_donchian_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_crossover_signals(
    hyper_donchian_values: np.ndarray, 
    frama_values: np.ndarray
) -> np.ndarray:
    """
    HyperドンチャンミッドラインとFRAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        hyper_donchian_values: Hyperドンチャンミッドライン値の配列
        frama_values: FRAMA値の配列
    
    Returns:
        シグナルの配列（1: ゴールデンクロス, -1: デッドクロス, 0: シグナルなし）
    """
    length = len(hyper_donchian_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(hyper_donchian_values[i]) or np.isnan(frama_values[i]) or 
            np.isnan(hyper_donchian_values[i-1]) or np.isnan(frama_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_hyper_donchian = hyper_donchian_values[i-1]
        prev_frama = frama_values[i-1]
        
        # 現在の期間
        curr_hyper_donchian = hyper_donchian_values[i]
        curr_frama = frama_values[i]
        
        # ゴールデンクロス: 前期間でFRAMA <= Hyperドンチャン、現期間でFRAMA > Hyperドンチャン
        if prev_frama <= prev_hyper_donchian and curr_frama > curr_hyper_donchian:
            signals[i] = 1
        # デッドクロス: 前期間でFRAMA >= Hyperドンチャン、現期間でFRAMA < Hyperドンチャン
        elif prev_frama >= prev_hyper_donchian and curr_frama < curr_hyper_donchian:
            signals[i] = -1
    
    return signals


class HyperDonchianFRAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    Hyperドンチャンミッドライン/FRAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - Hyperドンチャンミッドライン: 80-20パーセンタイルベースのチャネル中央線（外れ値に堅牢）
    - FRAMA: フラクタル適応移動平均（市場効率性に応じて適応的に反応）
    - 軽量でシンプルな計算
    - 高速なNumba最適化
    - 従来のドンチャンより安定性向上
    
    シグナル条件:
    - position_mode=True: FRAMA > Hyperドンチャンミッドライン: ロングシグナル (1), FRAMA < Hyperドンチャンミッドライン: ショートシグナル (-1)
    - position_mode=False: ゴールデンクロス: ロングシグナル (1), デッドクロス: ショートシグナル (-1)
    """
    
    def __init__(
        self,
        # Hyperドンチャンミッドラインパラメータ
        hyper_donchian_period: int = 20,              # Hyperドンチャン期間
        hyper_donchian_src_type: str = 'hlc3',        # Hyperドンチャンソースタイプ
        
        # FRAMAパラメータ
        frama_period: int = 16,                 # FRAMA期間（偶数である必要がある）
        frama_src_type: str = 'hlc3',          # FRAMAソースタイプ
        frama_fc: int = 1,                     # FRAMA Fast Constant
        frama_sc: int = 198,                   # FRAMA Slow Constant
        frama_period_mode: str = 'fixed',      # FRAMA期間モード（'fixed' または 'dynamic'）
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = False,  # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min: float = 1.0,                 # FRAMA FC最小値（ER高い時）
        frama_fc_max: float = 13.0,                # FRAMA FC最大値（ER低い時）
        frama_sc_min: float = 60.0,                # FRAMA SC最小値（ER高い時）
        frama_sc_max: float = 250.0,               # FRAMA SC最大値（ER低い時）
        
        # Hyperドンチャン HyperER動的適応パラメータ
        hyper_donchian_period_min: float = 55.0,         # Hyperドンチャン最小期間（ER高い時）
        hyper_donchian_period_max: float = 250.0,        # Hyperドンチャン最大期間（ER低い時）
        
        # シグナル設定
        position_mode: bool = True             # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
    ):
        """
        初期化
        
        Args:
            hyper_donchian_period: Hyperドンチャン期間（デフォルト: 20）
            hyper_donchian_src_type: Hyperドンチャンソースタイプ（デフォルト: 'hlc3'）
            frama_period: FRAMA期間（偶数である必要がある、デフォルト: 16）
            frama_src_type: FRAMAソースタイプ（デフォルト: 'hlc3'）
            frama_fc: FRAMA Fast Constant（デフォルト: 1）
            frama_sc: FRAMA Slow Constant（デフォルト: 198）
            frama_period_mode: FRAMA期間モード（デフォルト: 'fixed'）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        """
        signal_type = "Position" if position_mode else "Crossover"
        
        # 動的適応文字列の作成
        adaptation_str = "_hyper_er" if enable_hyper_er_adaptation else ""
        
        super().__init__(
            f"HyperDonchianFRAMA{signal_type}EntrySignal(hyper_donchian={hyper_donchian_period}({hyper_donchian_src_type}), frama={frama_period}({frama_src_type}){adaptation_str})"
        )
        
        # パラメータの保存
        self._params = {
            'hyper_donchian_period': hyper_donchian_period,
            'hyper_donchian_src_type': hyper_donchian_src_type,
            'frama_period': frama_period,
            'frama_src_type': frama_src_type,
            'frama_fc': frama_fc,
            'frama_sc': frama_sc,
            'frama_period_mode': frama_period_mode,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'frama_fc_min': frama_fc_min,
            'frama_fc_max': frama_fc_max,
            'frama_sc_min': frama_sc_min,
            'frama_sc_max': frama_sc_max,
            'hyper_donchian_period_min': hyper_donchian_period_min,
            'hyper_donchian_period_max': hyper_donchian_period_max,
            'position_mode': position_mode
        }
        
        self.position_mode = position_mode
        
        # Hyperドンチャンミッドラインインジケーターの初期化
        self.hyper_donchian_midline = HyperDonchian(
            period=hyper_donchian_period,
            src_type=hyper_donchian_src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            period_min=hyper_donchian_period_min,
            period_max=hyper_donchian_period_max
        )
        
        # FRAMAインジケーターの初期化
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
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close']].values
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
                
            # Hyperドンチャンミッドラインの計算
            hyper_donchian_result = self.hyper_donchian_midline.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if hyper_donchian_result is None or len(hyper_donchian_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # FRAMAの計算
            frama_result = self.frama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if frama_result is None or len(frama_result.values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # Hyperドンチャンミッドライン値とFRAMA値の取得
            hyper_donchian_values = hyper_donchian_result.values
            frama_values = frama_result.values
            
            # データ長を合わせる
            min_length = min(len(hyper_donchian_values), len(frama_values))
            if min_length == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # データを切り詰め
            hyper_donchian_values = hyper_donchian_values[:min_length]
            frama_values = frama_values[:min_length]
            
            # シグナルの計算（位置関係またはクロスオーバー）
            if self.position_mode:
                # 位置関係シグナル
                signals = calculate_position_signals(
                    hyper_donchian_values,
                    frama_values
                )
            else:
                # クロスオーバーシグナル
                signals = calculate_crossover_signals(
                    hyper_donchian_values,
                    frama_values
                )
            
            # 元のデータ長に合わせて結果を拡張
            full_signals = np.zeros(len(data), dtype=np.int8)
            full_signals[:len(signals)] = signals
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = full_signals
            return full_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperDonchianFRAMACrossoverEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_hyper_donchian_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Hyperドンチャンミッドライン値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Hyperドンチャンミッドライン値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_donchian_midline.get_values() or np.array([])
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.frama.get_values() or np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FRAMAフラクタル次元を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元値
        """
        if data is not None:
            self.generate(data)
            
        return self.frama.get_fractal_dimension() or np.array([])
    
    def get_hyper_donchian_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[tuple]:
        """
        Hyperドンチャンバンド（上部・下部）を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (upper_band, lower_band)
        """
        if data is not None:
            self.generate(data)
            
        upper = self.hyper_donchian_midline.get_upper_band()
        lower = self.hyper_donchian_midline.get_lower_band()
        
        if upper is not None and lower is not None:
            return (upper, lower)
        return None
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.hyper_donchian_midline.reset() if hasattr(self.hyper_donchian_midline, 'reset') else None
        self.frama.reset() if hasattr(self.frama, 'reset') else None
        self._signals_cache = {}


# 便利関数
def create_hyper_donchian_frama_entry_signal(
    hyper_donchian_period: int = 20,
    frama_period: int = 16,
    position_mode: bool = True,
    **kwargs
) -> HyperDonchianFRAMACrossoverEntrySignal:
    """
    HyperドンチャンFRAMAエントリーシグナルを作成する便利関数
    
    Args:
        hyper_donchian_period: Hyperドンチャン期間
        frama_period: FRAMA期間
        position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        **kwargs: その他のパラメータ
        
    Returns:
        HyperDonchianFRAMACrossoverEntrySignal: 設定済みのシグナルインスタンス
    """
    return HyperDonchianFRAMACrossoverEntrySignal(
        hyper_donchian_period=hyper_donchian_period,
        frama_period=frama_period,
        position_mode=position_mode,
        **kwargs
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== HyperドンチャンFRAMAエントリーシグナルのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.002 + np.random.normal(0, 0.008)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.010)
        elif i < 150:  # 強い上昇トレンド
            change = 0.004 + np.random.normal(0, 0.006)
        else:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.008)
        
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
    
    # 位置関係シグナルをテスト
    print("\\n位置関係シグナルをテスト中...")
    position_signal = HyperDonchianFRAMACrossoverEntrySignal(
        hyper_donchian_period=20,
        frama_period=16,
        position_mode=True
    )
    
    position_signals = position_signal.generate(df)
    
    long_signals = np.sum(position_signals == 1)
    short_signals = np.sum(position_signals == -1)
    neutral_signals = np.sum(position_signals == 0)
    
    print(f"  ロングシグナル: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"  ショートシグナル: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"  中立: {neutral_signals} ({neutral_signals/len(df)*100:.1f}%)")
    
    # クロスオーバーシグナルをテスト
    print("\\nクロスオーバーシグナルをテスト中...")
    crossover_signal = HyperDonchianFRAMACrossoverEntrySignal(
        hyper_donchian_period=20,
        frama_period=16,
        position_mode=False
    )
    
    crossover_signals = crossover_signal.generate(df)
    
    golden_cross = np.sum(crossover_signals == 1)
    dead_cross = np.sum(crossover_signals == -1)
    no_cross = np.sum(crossover_signals == 0)
    
    print(f"  ゴールデンクロス: {golden_cross} ({golden_cross/len(df)*100:.1f}%)")
    print(f"  デッドクロス: {dead_cross} ({dead_cross/len(df)*100:.1f}%)")
    print(f"  クロスなし: {no_cross} ({no_cross/len(df)*100:.1f}%)")
    
    # インジケーター値のテスト
    print("\\nインジケーター値を取得中...")
    hyper_donchian_values = position_signal.get_hyper_donchian_values(df)
    frama_values = position_signal.get_frama_values(df)
    fractal_dim = position_signal.get_fractal_dimension(df)
    
    if len(hyper_donchian_values) > 0:
        print(f"  Hyperドンチャンミッドライン平均: {np.nanmean(hyper_donchian_values):.2f}")
    if len(frama_values) > 0:
        print(f"  FRAMA平均: {np.nanmean(frama_values):.2f}")
    if len(fractal_dim) > 0:
        print(f"  フラクタル次元平均: {np.nanmean(fractal_dim):.3f}")
    
    # 従来のドンチャンFRAMAとの比較
    print("\\n従来のドンチャンFRAMAとの比較...")
    try:
        from ..donchian_frama.entry import DonchianFRAMACrossoverEntrySignal
        
        traditional_signal = DonchianFRAMACrossoverEntrySignal(
            donchian_period=20,
            frama_period=16,
            position_mode=True
        )
        
        traditional_signals = traditional_signal.generate(df)
        trad_long = np.sum(traditional_signals == 1)
        trad_short = np.sum(traditional_signals == -1)
        trad_neutral = np.sum(traditional_signals == 0)
        
        print(f"  従来版ロングシグナル: {trad_long} ({trad_long/len(df)*100:.1f}%)")
        print(f"  従来版ショートシグナル: {trad_short} ({trad_short/len(df)*100:.1f}%)")
        print(f"  従来版中立: {trad_neutral} ({trad_neutral/len(df)*100:.1f}%)")
        
        # シグナル安定性比較
        hyper_changes = np.sum(np.diff(position_signals) != 0)
        trad_changes = np.sum(np.diff(traditional_signals) != 0)
        
        print(f"\\nシグナル安定性比較:")
        print(f"  Hyperドンチャン版シグナル変化: {hyper_changes}")
        print(f"  従来版シグナル変化: {trad_changes}")
        print(f"  安定性改善: {((trad_changes - hyper_changes) / trad_changes * 100):.1f}%")
        
    except ImportError:
        print("  従来のドンチャンFRAMAシグナルが見つかりませんでした")
    
    print("\\n=== テスト完了 ===")