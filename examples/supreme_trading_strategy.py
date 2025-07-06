#!/usr/bin/env python3
"""
🚀 Supreme Breakout Channel - 最高利益トレード戦略 🚀

このファイルは、Supreme Breakout Channelインジケーターを使用した
最も利益的なトレード戦略の実装例です。

特徴:
- 高精度エントリーシグナル（複数条件フィルタリング）
- 動的エグジット管理
- リスク管理統合
- バックテスト対応
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Supreme Breakout Channelをインポート
from indicators.supreme_breakout_channel import SupremeBreakoutChannel, SupremeBreakoutChannelResult

@dataclass
class TradeSignal:
    """トレードシグナル"""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'
    price: float
    confidence: float
    strength: float
    reason: str

@dataclass
class Position:
    """ポジション情報"""
    entry_time: datetime
    entry_price: float
    position_type: int  # 1=ロング, -1=ショート
    size: float
    stop_loss: float
    take_profit: float
    confidence: float

class SupremeTradingStrategy:
    """
    🏆 Supreme Breakout Channel 最高利益トレード戦略
    
    この戦略は以下の組み合わせで最大利益を狙います：
    1. 高信頼度ブレイクアウトシグナル (confidence >= 0.7)
    2. 強いトレンド確認 (trend_strength >= 0.6) 
    3. トレンド方向一致チェック
    4. 偽シグナルフィルター適用
    5. Supreme知能スコア活用
    6. 動的ストップロス・利確管理
    """
    
    def __init__(self,
                 # エントリー条件
                 min_confidence: float = 0.7,           # 最小信頼度
                 min_trend_strength: float = 0.6,       # 最小トレンド強度  
                 min_breakout_strength: float = 0.5,    # 最小ブレイクアウト強度
                 min_supreme_score: float = 0.6,        # 最小Supreme知能スコア
                 
                 # リスク管理
                 max_risk_per_trade: float = 0.02,      # 1トレードの最大リスク (2%)
                 profit_target_ratio: float = 2.0,      # 利益目標倍率 (リスクの2倍)
                 trailing_stop_ratio: float = 0.5,      # トレーリングストップ比率
                 
                 # ポジション管理
                 max_positions: int = 3,                # 最大同時ポジション数
                 position_sizing_method: str = 'fixed'   # 'fixed', 'confidence', 'volatility'
                 ):
        
        self.min_confidence = min_confidence
        self.min_trend_strength = min_trend_strength
        self.min_breakout_strength = min_breakout_strength
        self.min_supreme_score = min_supreme_score
        
        self.max_risk_per_trade = max_risk_per_trade
        self.profit_target_ratio = profit_target_ratio
        self.trailing_stop_ratio = trailing_stop_ratio
        
        self.max_positions = max_positions
        self.position_sizing_method = position_sizing_method
        
        # トレード履歴
        self.signals: List[TradeSignal] = []
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        
        print(f"🚀 Supreme Trading Strategy initialized")
        print(f"   📊 エントリー条件: confidence≥{min_confidence}, trend≥{min_trend_strength}")
        print(f"   🛡️  リスク管理: {max_risk_per_trade:.1%}/trade, profit target {profit_target_ratio}x")
    
    def analyze_entry_opportunity(self, 
                                result: SupremeBreakoutChannelResult, 
                                index: int,
                                price_data: pd.DataFrame) -> Optional[TradeSignal]:
        """
        🎯 最高利益エントリー機会分析
        
        Returns:
            TradeSignal or None
        """
        
        # 基本データ取得
        if index >= len(result.breakout_signals):
            return None
            
        breakout_signal = result.breakout_signals[index]
        confidence = result.signal_confidence[index]
        trend_strength = result.trend_strength[index]
        breakout_strength = result.breakout_strength[index]
        hilbert_trend = result.hilbert_trend[index]
        false_signal_filter = result.false_signal_filter[index]
        
        # ブレイクアウトシグナルがない場合は終了
        if breakout_signal == 0:
            return None
        
        # 🔥 **最重要フィルター群** 🔥
        
        # 1. 高信頼度チェック
        if confidence < self.min_confidence:
            return None
            
        # 2. 強いトレンドチェック  
        if trend_strength < self.min_trend_strength:
            return None
            
        # 3. 強いブレイクアウトチェック
        if breakout_strength < self.min_breakout_strength:
            return None
            
        # 4. 偽シグナルフィルター
        if false_signal_filter != 1:
            return None
            
        # 5. Supreme知能スコアチェック
        if result.supreme_intelligence_score < self.min_supreme_score:
            return None
            
        # 6. トレンド方向一致チェック（最重要）
        if breakout_signal > 0:  # 上抜けブレイクアウト
            if hilbert_trend < 0.6:  # でも上昇トレンドでない
                return None
            signal_type = 'BUY'
        else:  # 下抜けブレイクアウト  
            if hilbert_trend > 0.4:  # でも下降トレンドでない
                return None
            signal_type = 'SELL'
        
        # 🏆 **全ての条件をクリア** 🏆
        
        # 現在価格取得
        current_price = price_data.iloc[index]['close']
        timestamp = price_data.index[index]
        
        # エントリー理由生成
        reason = (f"Supreme Entry: conf={confidence:.2f}, "
                 f"trend_str={trend_strength:.2f}, "
                 f"breakout_str={breakout_strength:.2f}, "
                 f"hilbert={hilbert_trend:.2f}")
        
        return TradeSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=current_price,
            confidence=confidence,
            strength=breakout_strength,
            reason=reason
        )
    
    def calculate_position_size(self, 
                              signal: TradeSignal,
                              account_balance: float,
                              stop_loss_price: float) -> float:
        """
        📊 ポジションサイズ計算
        """
        
        # リスク金額計算
        risk_amount = account_balance * self.max_risk_per_trade
        
        # 価格リスク計算
        price_risk = abs(signal.price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        # 基本ポジションサイズ
        base_size = risk_amount / price_risk
        
        # ポジションサイズ調整方法
        if self.position_sizing_method == 'confidence':
            # 信頼度に基づく調整
            confidence_multiplier = signal.confidence / 0.7  # 基準信頼度で正規化
            base_size *= confidence_multiplier
            
        elif self.position_sizing_method == 'volatility':
            # ボラティリティに基づく調整（簡易版）
            volatility_factor = min(2.0, max(0.5, 1.0 / signal.strength))
            base_size *= volatility_factor
        
        return base_size
    
    def calculate_stop_loss_take_profit(self,
                                      signal: TradeSignal,
                                      result: SupremeBreakoutChannelResult,
                                      index: int) -> Tuple[float, float]:
        """
        🛡️ 動的ストップロス・利確レベル計算
        """
        
        entry_price = signal.price
        
        # チャネル情報取得
        upper_channel = result.upper_channel[index]
        lower_channel = result.lower_channel[index]
        dynamic_width = result.dynamic_width[index]
        
        if signal.signal_type == 'BUY':
            # ロングポジション
            # ストップロス: 下側チャネルまたは動的幅ベース
            stop_loss = min(lower_channel, entry_price - dynamic_width * 0.5)
            
            # 利確: 上側チャネル or リスクリワード比率ベース
            channel_target = upper_channel
            ratio_target = entry_price + (entry_price - stop_loss) * self.profit_target_ratio
            take_profit = max(channel_target, ratio_target)
            
        else:  # SELL
            # ショートポジション  
            # ストップロス: 上側チャネルまたは動的幅ベース
            stop_loss = max(upper_channel, entry_price + dynamic_width * 0.5)
            
            # 利確: 下側チャネル or リスクリワード比率ベース
            channel_target = lower_channel
            ratio_target = entry_price - (stop_loss - entry_price) * self.profit_target_ratio
            take_profit = min(channel_target, ratio_target)
        
        return stop_loss, take_profit
    
    def should_close_position(self,
                            position: Position,
                            result: SupremeBreakoutChannelResult,
                            index: int,
                            current_price: float) -> Optional[str]:
        """
        🚪 ポジションクローズ判定
        """
        
        # 基本的なストップロス・利確チェック
        if position.position_type > 0:  # ロングポジション
            if current_price <= position.stop_loss:
                return "STOP_LOSS"
            if current_price >= position.take_profit:
                return "TAKE_PROFIT"
        else:  # ショートポジション
            if current_price >= position.stop_loss:
                return "STOP_LOSS"
            if current_price <= position.take_profit:
                return "TAKE_PROFIT"
        
        # トレンド転換チェック
        if index < len(result.hilbert_trend):
            hilbert_trend = result.hilbert_trend[index]
            
            # ロングポジションでトレンドが下降に転換
            if position.position_type > 0 and hilbert_trend < 0.4:
                return "TREND_REVERSAL"
                
            # ショートポジションでトレンドが上昇に転換
            if position.position_type < 0 and hilbert_trend > 0.6:
                return "TREND_REVERSAL"
        
        # Supreme知能スコアが大幅低下
        if result.supreme_intelligence_score < 0.4:
            return "LOW_INTELLIGENCE"
        
        return None
    
    def generate_signals(self,
                        price_data: pd.DataFrame,
                        sbc_result: SupremeBreakoutChannelResult,
                        account_balance: float = 10000) -> List[TradeSignal]:
        """
        🎯 トレードシグナル生成（バックテスト用）
        """
        signals = []
        current_positions = []
        
        print(f"🧮 Analyzing {len(price_data)} data points for trading signals...")
        
        for i in range(len(price_data)):
            current_price = price_data.iloc[i]['close']
            timestamp = price_data.index[i]
            
            # 既存ポジションのクローズチェック
            positions_to_close = []
            for pos in current_positions:
                close_reason = self.should_close_position(pos, sbc_result, i, current_price)
                if close_reason:
                    close_signal = TradeSignal(
                        timestamp=timestamp,
                        signal_type='CLOSE_LONG' if pos.position_type > 0 else 'CLOSE_SHORT',
                        price=current_price,
                        confidence=pos.confidence,
                        strength=0.0,
                        reason=close_reason
                    )
                    signals.append(close_signal)
                    positions_to_close.append(pos)
                    
                    # トレード記録
                    pnl = ((current_price - pos.entry_price) * pos.position_type) / pos.entry_price
                    self.closed_trades.append({
                        'entry_time': pos.entry_time,
                        'exit_time': timestamp,
                        'entry_price': pos.entry_price,
                        'exit_price': current_price,
                        'position_type': pos.position_type,
                        'pnl_pct': pnl * 100,
                        'close_reason': close_reason
                    })
            
            # クローズしたポジションを削除
            for pos in positions_to_close:
                current_positions.remove(pos)
            
            # 新規エントリーチェック
            if len(current_positions) < self.max_positions:
                entry_signal = self.analyze_entry_opportunity(sbc_result, i, price_data)
                if entry_signal:
                    # ストップロス・利確計算
                    stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                        entry_signal, sbc_result, i)
                    
                    # ポジションサイズ計算
                    position_size = self.calculate_position_size(
                        entry_signal, account_balance, stop_loss)
                    
                    if position_size > 0:
                        signals.append(entry_signal)
                        
                        # ポジション記録
                        position = Position(
                            entry_time=timestamp,
                            entry_price=entry_signal.price,
                            position_type=1 if entry_signal.signal_type == 'BUY' else -1,
                            size=position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=entry_signal.confidence
                        )
                        current_positions.append(position)
        
        print(f"✅ Generated {len(signals)} total signals")
        print(f"📊 Closed {len(self.closed_trades)} trades")
        
        return signals
    
    def get_performance_summary(self) -> Dict:
        """
        📈 パフォーマンスサマリー
        """
        if not self.closed_trades:
            return {}
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        # 基本統計
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL統計
        total_pnl = trades_df['pnl_pct'].sum()
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        # リスクリワード比率
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 最大ドローダウン
        cumulative_pnl = trades_df['pnl_pct'].cumsum()
        max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss * losing_trades != 0 else 0
        }


def main():
    """メイン実行例"""
    
    # データ読み込み例（実際のデータに置き換えてください）
    print("📊 Loading sample data...")
    
    # ダミーデータ生成（実際の実装では実データを使用）
    dates = pd.date_range('2023-01-01', periods=1000, freq='4H')
    np.random.seed(42)
    
    # トレンドのあるランダムウォーク
    returns = np.random.randn(1000) * 0.02 + 0.0001
    prices = 50000 * np.exp(np.cumsum(returns))
    
    # OHLC作成
    price_data = pd.DataFrame(index=dates)
    price_data['close'] = prices
    price_data['high'] = prices * (1 + np.abs(np.random.randn(1000) * 0.01))
    price_data['low'] = prices * (1 - np.abs(np.random.randn(1000) * 0.01))
    price_data['open'] = price_data['close'].shift(1).fillna(price_data['close'])
    price_data['volume'] = np.random.randint(100, 1000, 1000)
    
    print(f"✅ Sample data loaded: {len(price_data)} candles")
    
    # Supreme Breakout Channel計算
    print("🚀 Calculating Supreme Breakout Channel...")
    sbc = SupremeBreakoutChannel(
        atr_period=14,
        base_multiplier=2.0,
        min_confidence_threshold=0.3  # より多くのシグナルを生成
    )
    
    sbc_result = sbc.calculate(price_data)
    print("✅ SBC calculation completed")
    
    # トレード戦略実行
    print("\n🎯 Executing Supreme Trading Strategy...")
    strategy = SupremeTradingStrategy(
        min_confidence=0.7,
        min_trend_strength=0.6,
        min_breakout_strength=0.5,
        min_supreme_score=0.5,  # 少し下げてよりトレードを生成
        max_risk_per_trade=0.02,
        profit_target_ratio=2.0
    )
    
    # シグナル生成
    signals = strategy.generate_signals(price_data, sbc_result, account_balance=10000)
    
    # パフォーマンス分析
    print("\n📈 Performance Analysis:")
    performance = strategy.get_performance_summary()
    
    if performance:
        print(f"   📊 Total Trades: {performance['total_trades']}")
        print(f"   🎯 Win Rate: {performance['win_rate']:.1%}")
        print(f"   💰 Total PnL: {performance['total_pnl_pct']:.2f}%")
        print(f"   📈 Avg Win: {performance['avg_win_pct']:.2f}%")
        print(f"   📉 Avg Loss: {performance['avg_loss_pct']:.2f}%")
        print(f"   ⚖️  Risk/Reward: {performance['risk_reward_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        print(f"   🏆 Profit Factor: {performance['profit_factor']:.2f}")
    else:
        print("   ⚠️ No trades generated with current parameters")
    
    # シグナル詳細表示
    print(f"\n🎯 Generated Signals Summary:")
    entry_signals = [s for s in signals if s.signal_type in ['BUY', 'SELL']]
    exit_signals = [s for s in signals if s.signal_type.startswith('CLOSE')]
    
    print(f"   🚀 Entry Signals: {len(entry_signals)}")
    print(f"   🚪 Exit Signals: {len(exit_signals)}")
    
    if entry_signals:
        print(f"\n🔍 First 5 Entry Signals:")
        for i, signal in enumerate(entry_signals[:5]):
            print(f"   {i+1}. {signal.timestamp}: {signal.signal_type} @ {signal.price:.2f} "
                  f"(conf={signal.confidence:.2f}, strength={signal.strength:.2f})")


if __name__ == "__main__":
    main() 