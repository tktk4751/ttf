#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))


class TrendFollowBacktesting:
    """
    トレンドフォローモデル用バックテスト
    
    仕様書通りの実装:
    - 高信頼度シグナル生成
    - 3ATR損切り・7ATR利益確定
    - 300ローソク足最大保有期間
    - リスクパリティベースポジションサイズ
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission_rate: float = 0.001,
                 confidence_threshold: float = 0.6,
                 atr_period: int = 5,
                 stop_atr_mult: float = 3.0,
                 target_atr_mult: float = 7.0,
                 max_holding_bars: int = 300):
        """
        初期化
        
        Args:
            initial_capital: 初期資金
            commission_rate: 手数料率
            confidence_threshold: シグナル生成の信頼度閾値
            atr_period: ATR計算期間
            stop_atr_mult: 損切りライン
            target_atr_mult: 利益目標
            max_holding_bars: 最大保有期間
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.confidence_threshold = confidence_threshold
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.max_holding_bars = max_holding_bars
        
        # バックテスト結果
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """ATRを計算"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def generate_trading_signals(self, 
                               predictions: np.ndarray) -> np.ndarray:
        """
        モデル予測値からトレーディングシグナルを生成
        
        Args:
            predictions: LightGBMモデルの予測確率 (3クラス分類)
        
        Returns:
            signals: +1(買い), -1(売り), 0(中立)
        """
        # 各クラスの予測確率を取得
        prob_neutral = predictions[:, 0]  # 失敗・中立クラス(0)の確率
        prob_buy = predictions[:, 1]      # 買い成功クラス(1)の確率
        prob_sell = predictions[:, 2]     # 売り成功クラス(2)の確率
        
        signals = np.zeros(len(predictions))
        
        # 高い信頼度で成功が予測される場合のみシグナル生成
        buy_mask = (
            (prob_buy >= self.confidence_threshold) & 
            (prob_buy > prob_sell) & 
            (prob_buy > prob_neutral)
        )
        sell_mask = (
            (prob_sell >= self.confidence_threshold) & 
            (prob_sell > prob_buy) & 
            (prob_sell > prob_neutral)
        )
        
        signals[buy_mask] = 1   # 買いシグナル
        signals[sell_mask] = -1 # 売りシグナル
        
        return signals
    
    def calculate_position_size(self, 
                              capital: float, 
                              price: float, 
                              atr: float,
                              risk_per_trade: float = 0.02) -> float:
        """
        リスクパリティベースのポジションサイズ計算
        
        Args:
            capital: 現在の資金
            price: エントリー価格
            atr: 現在のATR
            risk_per_trade: 1トレードあたりのリスク（資金の%）
        
        Returns:
            ポジションサイズ（株数）
        """
        # リスク金額
        risk_amount = capital * risk_per_trade
        
        # 1株あたりのリスク（3ATR）
        risk_per_share = atr * self.stop_atr_mult
        
        # ポジションサイズ
        if risk_per_share > 0:
            position_size = risk_amount / risk_per_share
            # 最大でも資金の50%まで
            max_size = capital * 0.5 / price
            position_size = min(position_size, max_size)
        else:
            position_size = 0
        
        return position_size
    
    def run_backtest(self, 
                     data: pd.DataFrame, 
                     predictions: np.ndarray) -> Dict[str, Any]:
        """
        バックテスト実行
        
        Args:
            data: OHLCVデータ
            predictions: モデル予測確率
        
        Returns:
            バックテスト結果
        """
        print("=== バックテスト実行開始 ===")
        
        # 初期化
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
        capital = self.initial_capital
        current_position = None
        
        # ATR計算
        atr = self.calculate_atr(data)
        
        # シグナル生成
        signals = self.generate_trading_signals(predictions)
        
        print(f"シグナル統計: 買い {(signals == 1).sum()}, 売り {(signals == -1).sum()}, 中立 {(signals == 0).sum()}")
        
        # データとシグナルの長さを確認・調整
        min_length = min(len(data), len(signals))
        if len(data) != len(signals):
            print(f"  長さ調整: データ{len(data)}行, シグナル{len(signals)}行 -> {min_length}行")
        
        # バックテストループ
        for i in range(min_length):
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            current_atr = atr.iloc[i] if i < len(atr) and not np.isnan(atr.iloc[i]) else 0
            current_signal = signals[i]
            
            # 既存ポジションの管理
            if current_position is not None:
                position_result = self._manage_position(
                    current_position, i, current_high, current_low, current_price, current_date
                )
                
                if position_result is not None:
                    # ポジション決済
                    capital += position_result['pnl']
                    self.trades.append(position_result)
                    current_position = None
            
            # 新規エントリー
            if current_position is None and current_signal != 0 and current_atr > 0:
                position_size = self.calculate_position_size(capital, current_price, current_atr)
                
                if position_size > 0:
                    current_position = self._create_position(
                        current_signal, i, current_price, current_low, current_high, 
                        current_atr, position_size, current_date
                    )
                    
                    # 手数料を差し引く
                    commission = position_size * current_price * self.commission_rate
                    capital -= commission
            
            # エクイティカーブの記録
            portfolio_value = capital
            if current_position is not None:
                unrealized_pnl = self._calculate_unrealized_pnl(current_position, current_price)
                portfolio_value += unrealized_pnl
            
            self.equity_curve.append({
                'date': current_date,
                'capital': capital,
                'portfolio_value': portfolio_value,
                'position': 1 if current_position is not None else 0
            })
        
        # 最後のポジションがあれば強制決済
        if current_position is not None:
            final_result = self._force_close_position(current_position, len(data)-1, 
                                                    data['close'].iloc[-1], data.index[-1])
            capital += final_result['pnl']
            self.trades.append(final_result)
        
        # 結果分析
        result = self._analyze_backtest_results(capital)
        
        print(f"バックテスト完了: {len(self.trades)}トレード")
        return result
    
    def _create_position(self, 
                        signal: int, 
                        entry_idx: int, 
                        entry_price: float,
                        current_low: float, 
                        current_high: float, 
                        atr: float, 
                        size: float, 
                        date) -> Dict[str, Any]:
        """ポジション作成"""
        if signal == 1:  # 買いポジション
            stop_loss = current_low - (self.stop_atr_mult * atr)
            take_profit = entry_price + (self.target_atr_mult * atr)
        else:  # 売りポジション
            stop_loss = current_high + (self.stop_atr_mult * atr)
            take_profit = entry_price - (self.target_atr_mult * atr)
        
        return {
            'type': 'buy' if signal == 1 else 'sell',
            'entry_idx': entry_idx,
            'entry_date': date,
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_holding_bars': self.max_holding_bars
        }
    
    def _manage_position(self, 
                        position: Dict[str, Any], 
                        current_idx: int, 
                        high: float, 
                        low: float, 
                        close: float, 
                        date) -> Optional[Dict[str, Any]]:
        """ポジション管理"""
        holding_bars = current_idx - position['entry_idx']
        
        # 最大保有期間チェック
        if holding_bars >= position['max_holding_bars']:
            return self._close_position(position, close, date, 'time_limit')
        
        if position['type'] == 'buy':
            # 利益確定チェック
            if high >= position['take_profit']:
                return self._close_position(position, position['take_profit'], date, 'take_profit')
            # 損切りチェック
            if low <= position['stop_loss']:
                return self._close_position(position, position['stop_loss'], date, 'stop_loss')
        else:  # sell
            # 利益確定チェック
            if low <= position['take_profit']:
                return self._close_position(position, position['take_profit'], date, 'take_profit')
            # 損切りチェック
            if high >= position['stop_loss']:
                return self._close_position(position, position['stop_loss'], date, 'stop_loss')
        
        return None
    
    def _close_position(self, 
                       position: Dict[str, Any], 
                       exit_price: float, 
                       exit_date, 
                       exit_reason: str) -> Dict[str, Any]:
        """ポジション決済"""
        entry_value = position['size'] * position['entry_price']
        exit_value = position['size'] * exit_price
        
        if position['type'] == 'buy':
            pnl = exit_value - entry_value
        else:  # sell
            pnl = entry_value - exit_value
        
        # 手数料を差し引く
        commission = entry_value * self.commission_rate + exit_value * self.commission_rate
        pnl -= commission
        
        return {
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'type': position['type'],
            'size': position['size'],
            'pnl': pnl,
            'return_pct': pnl / entry_value,
            'holding_bars': 0,  # 実際の保有期間は別途計算
            'exit_reason': exit_reason
        }
    
    def _force_close_position(self, 
                            position: Dict[str, Any], 
                            exit_idx: int, 
                            exit_price: float, 
                            exit_date) -> Dict[str, Any]:
        """強制決済"""
        return self._close_position(position, exit_price, exit_date, 'forced_close')
    
    def _calculate_unrealized_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """含み損益計算"""
        entry_value = position['size'] * position['entry_price']
        current_value = position['size'] * current_price
        
        if position['type'] == 'buy':
            unrealized_pnl = current_value - entry_value
        else:  # sell
            unrealized_pnl = entry_value - current_value
        
        return unrealized_pnl
    
    def _analyze_backtest_results(self, final_capital: float) -> Dict[str, Any]:
        """バックテスト結果分析"""
        if not self.trades:
            return {'error': 'トレードが発生しませんでした'}
        
        # 基本統計
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        
        # 損益分析
        pnls = [trade['pnl'] for trade in self.trades]
        returns = [trade['return_pct'] for trade in self.trades]
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        # リスク指標
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        
        sharpe_ratio = self._calculate_sharpe_ratio(equity_df['returns'])
        max_drawdown = self._calculate_max_drawdown(equity_df['portfolio_value'])
        
        # トレード分析
        exit_reasons = {}
        for trade in self.trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'trade_details': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """シャープレシオ計算"""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # 年率換算
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """最大ドローダウン計算"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """エクイティカーブの描画"""
        if not self.equity_curve:
            print("エクイティカーブデータがありません")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['date'], equity_df['portfolio_value'], label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"エクイティカーブを保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_trade_report(self) -> pd.DataFrame:
        """トレードレポート生成"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)


def main():
    """メイン実行関数"""
    # バックテスト初期化
    backtester = TrendFollowBacktesting()
    
    print("=== トレンドフォローバックテスト設定 ===")
    print(f"初期資金: ${backtester.initial_capital:,.2f}")
    print(f"手数料率: {backtester.commission_rate:.3%}")
    print(f"信頼度閾値: {backtester.confidence_threshold}")
    print(f"損切りライン: {backtester.stop_atr_mult}ATR")
    print(f"利益目標: {backtester.target_atr_mult}ATR")
    print(f"最大保有期間: {backtester.max_holding_bars}ローソク足")


if __name__ == "__main__":
    main()