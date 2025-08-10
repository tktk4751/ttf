#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))
from signals.implementations.mama.entry import MAMACrossoverEntrySignal


class TrendFollowTargetCalculation:
    """
    トレンドフォローモデル用目的変数計算
    
    仕様書通りの実装:
    - ATRベースのリスク調整リターン
    - 5期間ATR、3ATR損切り、7ATR利益確定
    - 300ローソク足以内という時間制約
    - 3クラス分類: 買い成功(+1)、売り成功(-1)、失敗・中立(0)
    """
    
    def __init__(self, 
                 atr_period: int = 5,
                 stop_atr_mult: float = 3.0,
                 target_atr_mult: float = 7.0,
                 max_bars: int = 300):
        """
        初期化
        
        Args:
            atr_period: ATR計算期間（デフォルト: 5）
            stop_atr_mult: 損切りライン（デフォルト: 3ATR）
            target_atr_mult: 利益目標（デフォルト: 7ATR）
            max_bars: 最大保有期間（ローソク足本数、デフォルト: 300）
        """
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.max_bars = max_bars
        
        # MAMA位置関係シグナル生成器
        self.mama_signal_generator = MAMACrossoverEntrySignal(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3'
        )
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Average True Range (ATR) を計算
        
        Args:
            data: OHLCVデータフレーム
        
        Returns:
            ATR値のシリーズ
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def generate_mama_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        MAMA位置関係シグナルを生成
        
        Args:
            data: OHLCVデータフレーム
        
        Returns:
            シグナル配列 (+1: 買い, -1: 売り, 0: 中立)
        """
        try:
            signals = self.mama_signal_generator.generate(data)
            return signals
        except Exception as e:
            print(f"MAMA信号生成エラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def calculate_atr_normalized_target(self, 
                                     data: pd.DataFrame, 
                                     mama_signals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ATRベースのリスク調整リターンを目的変数として計算
        
        Args:
            data: 価格データ (OHLCV)
            mama_signals: MAMA位置関係シグナル（None の場合は自動生成）
        
        Returns:
            ANRS: ATR正規化リターンスコア (+1: 買い成功, -1: 売り成功, 0: 失敗・中立)
        """
        print("ATR正規化目的変数計算開始...")
        
        # データの妥当性チェック
        if not self._validate_input_data(data):
            raise ValueError("入力データが無効です")
        
        # MAMA信号の生成
        if mama_signals is None:
            print("  MAMA位置関係シグナル生成中...")
            mama_signals = self.generate_mama_signals(data)
        
        # ATR計算
        print("  ATR計算中...")
        atr = self.calculate_atr(data)
        
        # 目的変数計算
        print("  目的変数計算中...")
        target_values = self._calculate_targets(data, mama_signals, atr)
        
        # 結果の分析
        self._analyze_target_distribution(target_values, mama_signals)
        
        print("ATR正規化目的変数計算完了")
        return target_values
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """入力データの妥当性をチェック"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            print("エラー: 必要なカラムが不足しています")
            return False
        
        if len(data) < self.max_bars + 100:
            print(f"警告: データが少なすぎます。最低{self.max_bars + 100}行必要です")
            return False
        
        return True
    
    def _calculate_targets(self, 
                         data: pd.DataFrame, 
                         mama_signals: np.ndarray, 
                         atr: pd.Series) -> np.ndarray:
        """
        実際の目的変数計算処理
        
        Args:
            data: 価格データ
            mama_signals: MAMA位置関係シグナル
            atr: ATR値
        
        Returns:
            目的変数配列
        """
        target_values = np.zeros(len(data), dtype=np.int8)
        signal_count = 0
        success_count = 0
        
        for i in range(len(data) - 1):
            signal = mama_signals[i]
            if signal == 0 or np.isnan(atr.iloc[i]):
                continue
            
            signal_count += 1
            current_atr = atr.iloc[i]
            
            if signal == 1:  # 買いシグナル
                success = self._evaluate_buy_signal(data, i, current_atr)
                if success:
                    target_values[i] = 1  # 買い成功
                    success_count += 1
                else:
                    target_values[i] = 0  # 失敗
                    
            elif signal == -1:  # 売りシグナル
                success = self._evaluate_sell_signal(data, i, current_atr)
                if success:
                    target_values[i] = -1  # 売り成功
                    success_count += 1
                else:
                    target_values[i] = 0  # 失敗
        
        print(f"  シグナル評価完了: {signal_count}個のシグナル、{success_count}個の成功")
        return target_values
    
    def _evaluate_buy_signal(self, 
                           data: pd.DataFrame, 
                           entry_idx: int, 
                           current_atr: float) -> bool:
        """
        買いシグナルの評価
        
        Args:
            data: 価格データ
            entry_idx: エントリー位置
            current_atr: 現在のATR値
        
        Returns:
            成功したかどうか
        """
        entry_price = data['close'].iloc[entry_idx]
        stop_loss = data['low'].iloc[entry_idx] - (self.stop_atr_mult * current_atr)
        profit_target = entry_price + (self.target_atr_mult * current_atr)
        
        # 将来の価格動向をチェック（最大300本以内）
        max_check_bars = min(entry_idx + self.max_bars + 1, len(data))
        
        for j in range(entry_idx + 1, max_check_bars):
            low_price = data['low'].iloc[j]
            high_price = data['high'].iloc[j]
            
            # 損切りラインチェック
            if low_price <= stop_loss:
                return False  # 損切り発生
            
            # 利益目標達成チェック
            if high_price >= profit_target:
                return True  # 利益目標達成
        
        return False  # 300本以内に条件を満たさなかった
    
    def _evaluate_sell_signal(self, 
                            data: pd.DataFrame, 
                            entry_idx: int, 
                            current_atr: float) -> bool:
        """
        売りシグナルの評価
        
        Args:
            data: 価格データ
            entry_idx: エントリー位置
            current_atr: 現在のATR値
        
        Returns:
            成功したかどうか
        """
        entry_price = data['close'].iloc[entry_idx]
        stop_loss = data['high'].iloc[entry_idx] + (self.stop_atr_mult * current_atr)
        profit_target = entry_price - (self.target_atr_mult * current_atr)
        
        # 将来の価格動向をチェック（最大300本以内）
        max_check_bars = min(entry_idx + self.max_bars + 1, len(data))
        
        for j in range(entry_idx + 1, max_check_bars):
            low_price = data['low'].iloc[j]
            high_price = data['high'].iloc[j]
            
            # 損切りラインチェック
            if high_price >= stop_loss:
                return False  # 損切り発生
            
            # 利益目標達成チェック
            if low_price <= profit_target:
                return True  # 利益目標達成
        
        return False  # 300本以内に条件を満たさなかった
    
    def _analyze_target_distribution(self, 
                                   target_values: np.ndarray, 
                                   mama_signals: np.ndarray) -> Dict[str, Any]:
        """
        目的変数の分布を分析
        
        Args:
            target_values: 目的変数
            mama_signals: 元のMAMAシグナル
        
        Returns:
            分析結果
        """
        unique, counts = np.unique(target_values, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        # シグナル統計
        total_signals = np.sum(mama_signals != 0)
        buy_signals = np.sum(mama_signals == 1)
        sell_signals = np.sum(mama_signals == -1)
        
        # 成功統計
        buy_success = np.sum(target_values == 1)
        sell_success = np.sum(target_values == -1)
        total_success = buy_success + sell_success
        
        # 成功率計算
        buy_success_rate = buy_success / buy_signals if buy_signals > 0 else 0
        sell_success_rate = sell_success / sell_signals if sell_signals > 0 else 0
        overall_success_rate = total_success / total_signals if total_signals > 0 else 0
        
        print(f"  目的変数分布: {distribution}")
        print(f"  総シグナル数: {total_signals} (買い: {buy_signals}, 売り: {sell_signals})")
        print(f"  成功数: {total_success} (買い: {buy_success}, 売り: {sell_success})")
        print(f"  成功率: 全体 {overall_success_rate:.2%}, 買い {buy_success_rate:.2%}, 売り {sell_success_rate:.2%}")
        
        return {
            'distribution': distribution,
            'signal_counts': {'total': total_signals, 'buy': buy_signals, 'sell': sell_signals},
            'success_counts': {'total': total_success, 'buy': buy_success, 'sell': sell_success},
            'success_rates': {
                'overall': overall_success_rate,
                'buy': buy_success_rate,
                'sell': sell_success_rate
            }
        }
    
    def get_target_info(self) -> Dict[str, Any]:
        """目的変数の設定情報を取得"""
        return {
            'atr_period': self.atr_period,
            'stop_atr_multiplier': self.stop_atr_mult,
            'target_atr_multiplier': self.target_atr_mult,
            'max_holding_bars': self.max_bars,
            'signal_source': 'MAMA_crossover',
            'target_classes': {
                1: 'buy_success',
                -1: 'sell_success',
                0: 'failure_or_neutral'
            }
        }
    
    def create_target_labels(self, target_values: np.ndarray) -> np.ndarray:
        """
        3クラス分類用のラベルを作成
        
        Args:
            target_values: 目的変数 (+1, -1, 0)
        
        Returns:
            クラスラベル (0: 失敗・中立, 1: 買い成功, 2: 売り成功)
        """
        # -1 -> 2, 0 -> 0, 1 -> 1 に変換
        labels = np.zeros_like(target_values, dtype=np.int8)
        labels[target_values == 1] = 1   # 買い成功
        labels[target_values == -1] = 2  # 売り成功
        labels[target_values == 0] = 0   # 失敗・中立
        
        return labels


def main():
    """メイン実行関数"""
    # 目的変数計算器の初期化
    target_calc = TrendFollowTargetCalculation()
    
    # 設定情報の表示
    info = target_calc.get_target_info()
    print("=== ATR正規化目的変数設定 ===")
    print(f"ATR期間: {info['atr_period']}")
    print(f"損切りライン: {info['stop_atr_multiplier']}ATR")
    print(f"利益目標: {info['target_atr_multiplier']}ATR")
    print(f"最大保有期間: {info['max_holding_bars']}ローソク足")
    print(f"シグナル源: {info['signal_source']}")
    print(f"目的変数クラス: {info['target_classes']}")


if __name__ == "__main__":
    main()