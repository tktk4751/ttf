#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.trend_filter.hyper_donchian import HyperDonchian
from indicators.smoother.super_smoother import SuperSmoother


class HyperDonchianBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    Hyperドンチャンチャネルのブレイクアウトによるエントリーシグナル
    
    Min/Max範囲内80-20%位置ベースのHyperドンチャンを使用したブレイクアウトシグナル
    
    - 現在の終値がn期間Hyperドンチャンのアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値がn期間Hyperドンチャンのロワーバンドを下回った場合: ショートエントリー (-1)
    
    従来のドンチャンチャネルよりも安定したブレイクアウト検出が可能
    """
    
    def __init__(
        self,
        period: int = 20,
        src_type: str = 'close',
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = False,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        period_min: float = 15.0,
        period_max: float = 60.0,
        # スーパースムーザーフィルタリングオプション
        enable_super_smoother_filter: bool = True,
        super_smoother_period: int = 100,
        super_smoother_src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            period: Hyperドンチャンチャネルの期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'close'）
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか
            hyper_er_period: HyperER計算期間
            hyper_er_midline_period: HyperERミッドライン期間
            period_min: 最小期間（ER高い時）
            period_max: 最大期間（ER低い時）
            enable_super_smoother_filter: スーパースムーザーフィルタリングを有効にするか
            super_smoother_period: スーパースムーザーの期間（デフォルト: 10）
            super_smoother_src_type: スーパースムーザーのソースタイプ（デフォルト: 'close'）
        """
        params = {
            'period': period,
            'src_type': src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'period_min': period_min,
            'period_max': period_max,
            'enable_super_smoother_filter': enable_super_smoother_filter,
            'super_smoother_period': super_smoother_period,
            'super_smoother_src_type': super_smoother_src_type
        }
        
        adaptation_str = ""
        if enable_hyper_er_adaptation:
            adaptation_str = f"_HyperER({hyper_er_period},{hyper_er_midline_period})"
        
        filter_str = ""
        if enable_super_smoother_filter:
            filter_str = f"_SS({super_smoother_period})"
        
        super().__init__(
            f"HyperDonchianBreakout({period}, src={src_type}{adaptation_str}{filter_str})",
            params
        )
        
        # Hyperドンチャンインジケーターの初期化
        self._hyper_donchian = HyperDonchian(
            period=period,
            src_type=src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            period_min=period_min,
            period_max=period_max
        )
        
        # スーパースムーザーフィルタリング用インジケーターの初期化
        self._super_smoother = None
        if enable_super_smoother_filter:
            self._super_smoother = SuperSmoother(
                length=super_smoother_period,
                num_poles=2,
                src_type=super_smoother_src_type
            )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Hyperドンチャンブレイクアウトシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # Hyperドンチャンチャネルの計算
            hyper_donchian_result = self._hyper_donchian.calculate(data)
            
            # 上部バンドと下部バンドを取得
            upper_band = hyper_donchian_result.upper_band
            lower_band = hyper_donchian_result.lower_band
            
            # 価格データを取得
            if isinstance(data, pd.DataFrame):
                if self._params['src_type'] == 'close':
                    prices = data['close'].values
                elif self._params['src_type'] == 'high':
                    prices = data['high'].values
                elif self._params['src_type'] == 'low':
                    prices = data['low'].values
                elif self._params['src_type'] == 'open':
                    prices = data['open'].values
                elif self._params['src_type'] == 'hl2':
                    prices = ((data['high'] + data['low']) / 2).values
                elif self._params['src_type'] == 'hlc3':
                    prices = ((data['high'] + data['low'] + data['close']) / 3).values
                elif self._params['src_type'] == 'ohlc4':
                    prices = ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
                else:
                    prices = data['close'].values
            else:
                # NumPy配列の場合、終値（最後の列）を使用
                if data.ndim == 1:
                    prices = data
                else:
                    prices = data[:, -1]  # 終値列を仮定
            
            # シグナルの初期化
            signals = np.zeros(len(prices))
            
            # 最初の期間はシグナルなし
            period = self._params['period']
            signals[:period] = 0
            
            # スーパースムーザーフィルタリング用データの準備
            super_smoother_values = None
            if (self._params['enable_super_smoother_filter'] and 
                self._super_smoother is not None):
                try:
                    ss_result = self._super_smoother.calculate(data)
                    super_smoother_values = ss_result.values
                    # デバッグ用: スーパースムーザー計算完了
                    pass
                except Exception as e:
                    # スーパースムーザーの計算に失敗した場合はNoneに設定
                    super_smoother_values = None
            
            # ブレイクアウトの判定
            for i in range(period, len(prices)):
                # 前期のバンド値をチェック（ルックアヘッドバイアス回避）
                prev_upper = upper_band[i-1] if i > 0 else upper_band[i]
                prev_lower = lower_band[i-1] if i > 0 else lower_band[i]
                
                # 有効なバンド値がある場合のみ判定
                if not np.isnan(prev_upper) and not np.isnan(prev_lower):
                    current_price = prices[i]
                    
                    # 基本的なブレイクアウト判定
                    breakout_signal = 0
                    
                    # ロングエントリー: 現在価格がアッパーバンドを上回る
                    if current_price > prev_upper:
                        breakout_signal = 1
                    # ショートエントリー: 現在価格がロワーバンドを下回る
                    elif current_price < prev_lower:
                        breakout_signal = -1
                    
                    # スーパースムーザーフィルタリングの適用
                    if (self._params['enable_super_smoother_filter'] and 
                        self._super_smoother is not None and 
                        super_smoother_values is not None and 
                        i < len(super_smoother_values) and
                        not np.isnan(super_smoother_values[i])):
                        
                        ss_value = super_smoother_values[i]
                        
                        # ロングシグナルの場合: 終値がスーパースムーザーより上にある場合のみ許可
                        if breakout_signal == 1:
                            if current_price > ss_value:
                                signals[i] = 1
                            else:
                                signals[i] = 0  # フィルターでブロック
                        # ショートシグナルの場合: 終値がスーパースムーザーより下にある場合のみ許可  
                        elif breakout_signal == -1:
                            if current_price < ss_value:
                                signals[i] = -1
                            else:
                                signals[i] = 0  # フィルターでブロック
                        else:
                            signals[i] = 0
                    else:
                        # フィルタリング無効または計算失敗の場合は基本シグナルをそのまま使用
                        signals[i] = breakout_signal
            
            return signals
            
        except Exception as e:
            # エラー時は全てシグナルなしを返す
            print(f"Hyperドンチャンブレイクアウトシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data))
    
    def get_hyper_donchian_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Hyperドンチャンミッドライン値を取得（デバッグ用）"""
        try:
            result = self._hyper_donchian.calculate(data)
            return result.values
        except Exception:
            return np.array([])
    
    def get_hyper_donchian_bands(self, data: Union[pd.DataFrame, np.ndarray]) -> tuple:
        """Hyperドンチャンバンド値を取得（デバッグ用）"""
        try:
            result = self._hyper_donchian.calculate(data)
            return result.upper_band, result.lower_band
        except Exception:
            return np.array([]), np.array([])
    
    def get_signal_info(self) -> Dict[str, Any]:
        """シグナル情報を取得"""
        return {
            'name': self.name,
            'type': 'entry',
            'description': 'Hyperドンチャンチャネル(Min/Max範囲内80-20%位置版)ブレイクアウトエントリーシグナル',
            'parameters': self._params.copy(),
            'features': [
                'Min/Max範囲内80-20%位置ベースのHyperドンチャンチャネル使用',
                '従来ドンチャンより安定したブレイクアウト検出',
                'HyperER動的適応サポート（オプション）',
                'スーパースムーザーフィルタリング機能（オプション）',
                'ルックアヘッドバイアス回避設計'
            ]
        }


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Hyperドンチャンブレイクアウトエントリーシグナルのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとブレイクアウトが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.004 + np.random.normal(0, 0.006)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 120:  # 強いブレイクアウト
            change = 0.008 + np.random.normal(0, 0.004)
        elif i < 150:  # 安定期間
            change = np.random.normal(0, 0.005)
        else:  # 下降トレンド
            change = -0.003 + np.random.normal(0, 0.007)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
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
    
    # Hyperドンチャンブレイクアウトエントリーシグナルをテスト
    print("\\nHyperドンチャンブレイクアウトエントリーシグナルをテスト中...")
    
    # 固定期間版
    hyper_signal_fixed = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_hyper_er_adaptation=False
    )
    
    signals_fixed = hyper_signal_fixed.generate(df)
    long_signals_fixed = np.sum(signals_fixed == 1)
    short_signals_fixed = np.sum(signals_fixed == -1)
    no_signals_fixed = np.sum(signals_fixed == 0)
    
    print(f"  固定期間版 (period=20):")
    print(f"    ロングエントリー: {long_signals_fixed}")
    print(f"    ショートエントリー: {short_signals_fixed}")
    print(f"    シグナルなし: {no_signals_fixed}")
    
    # HyperER動的適応版
    print("\\nHyperER動的適応版をテスト中...")
    hyper_signal_adaptive = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_hyper_er_adaptation=True,
        hyper_er_period=14,
        hyper_er_midline_period=100,
        period_min=15.0,
        period_max=40.0
    )
    
    # スーパースムーザーフィルタリング版
    print("\\nスーパースムーザーフィルタリング版をテスト中...")
    hyper_signal_ss_filter = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_hyper_er_adaptation=False,
        enable_super_smoother_filter=True,
        super_smoother_period=200,
        super_smoother_src_type='close'
    )
    
    signals_adaptive = hyper_signal_adaptive.generate(df)
    long_signals_adaptive = np.sum(signals_adaptive == 1)
    short_signals_adaptive = np.sum(signals_adaptive == -1)
    no_signals_adaptive = np.sum(signals_adaptive == 0)
    
    print(f"  HyperER動的適応版:")
    print(f"    ロングエントリー: {long_signals_adaptive}")
    print(f"    ショートエントリー: {short_signals_adaptive}")
    print(f"    シグナルなし: {no_signals_adaptive}")
    
    signals_ss_filter = hyper_signal_ss_filter.generate(df)
    long_signals_ss_filter = np.sum(signals_ss_filter == 1)
    short_signals_ss_filter = np.sum(signals_ss_filter == -1)
    no_signals_ss_filter = np.sum(signals_ss_filter == 0)
    
    print(f"  スーパースムーザーフィルタリング版:")
    print(f"    ロングエントリー: {long_signals_ss_filter}")
    print(f"    ショートエントリー: {short_signals_ss_filter}")
    print(f"    シグナルなし: {no_signals_ss_filter}")
    
    # 従来のドンチャンと比較
    print("\\n従来のドンチャンブレイクアウトとの比較...")
    try:
        from ..donchian.entry import DonchianBreakoutEntrySignal
        
        traditional_signal = DonchianBreakoutEntrySignal(period=20)
        traditional_signals = traditional_signal.generate(df)
        
        trad_long = np.sum(traditional_signals == 1)
        trad_short = np.sum(traditional_signals == -1)
        trad_no = np.sum(traditional_signals == 0)
        
        print(f"  従来版:")
        print(f"    ロングエントリー: {trad_long}")
        print(f"    ショートエントリー: {trad_short}")
        print(f"    シグナルなし: {trad_no}")
        
        # シグナル頻度比較
        print(f"\\nシグナル頻度比較:")
        hyper_total = long_signals_fixed + short_signals_fixed
        trad_total = trad_long + trad_short
        
        print(f"  Hyperドンチャン総シグナル数: {hyper_total}")
        print(f"  従来ドンチャン総シグナル数: {trad_total}")
        
        if trad_total > 0:
            signal_change = ((hyper_total - trad_total) / trad_total) * 100
            print(f"  シグナル数変化: {signal_change:+.1f}%")
        
    except ImportError:
        print("  従来のドンチャンブレイクアウトシグナルが見つかりませんでした")
    
    # Hyperドンチャンバンド情報
    print("\\nHyperドンチャンバンド情報:")
    upper_band, lower_band = hyper_signal_fixed.get_hyper_donchian_bands(df)
    if len(upper_band) > 0 and len(lower_band) > 0:
        valid_upper = upper_band[~np.isnan(upper_band)]
        valid_lower = lower_band[~np.isnan(lower_band)]
        
        if len(valid_upper) > 0 and len(valid_lower) > 0:
            print(f"  アッパーバンド範囲: {valid_upper.min():.2f} - {valid_upper.max():.2f}")
            print(f"  ロワーバンド範囲: {valid_lower.min():.2f} - {valid_lower.max():.2f}")
            print(f"  平均バンド幅: {np.mean(valid_upper - valid_lower):.2f}")
    
    print("\\n=== テスト完了 ===")