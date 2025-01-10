from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from position_sizing.position_sizing import PositionSizing
from strategies.strategy import Strategy
from backtesting.trade import Trade

class Backtester:
    """バックテストの実行クラス"""
    
    def __init__(
        self,
        strategy: Strategy,
        position_sizing: PositionSizing,
        initial_capital: float = 10000,
        max_positions: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            strategy: バックテストする戦略
            position_sizing: ポジションサイズ計算ロジック
            initial_capital: 初期資金
            max_positions: 同時に保有できる最大ポジション数
        """
        self.strategy = strategy
        self.position_sizing = position_sizing
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.trades: List[Trade] = []
        self.current_capital = initial_capital
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        バックテストを実行する
        
        Args:
            data: バックテストに使用するデータ
        
        Returns:
            Dict[str, Any]: バックテスト結果
        """
        # データの準備
        dates = data.index
        opens = data['open'].values
        closes = data['close'].values
        current_position: Optional[Trade] = None
        pending_entry: Optional[tuple] = None  # (position_type, position_size)
        pending_exit: bool = False
        
        # エントリーシグナルの生成
        entry_signals = self.strategy.generate_entry(data)
        
        # バックテストのメインループ
        for i in range(1, len(data)):
            date = dates[i]
            open_price = opens[i]
            close = closes[i]
            
            # 保留中のエグジットの処理
            if pending_exit and current_position is not None:
                current_position.close(date, open_price)
                self.trades.append(current_position)
                self.current_capital += current_position.profit_loss
                current_position = None
                pending_exit = False
            
            # 保留中のエントリーの処理
            if pending_entry is not None and current_position is None:
                position_type, position_size = pending_entry
                current_position = Trade(
                    position_type=position_type,
                    position_size=position_size
                )
                current_position.entry(date, open_price)
                pending_entry = None
            
            # 現在のポジションがある場合、エグジットシグナルをチェック
            if current_position is not None and not pending_exit:
                position_type = 1 if current_position.position_type == 'LONG' else -1
                if self.strategy.generate_exit(data, position_type, i):
                    pending_exit = True
            
            # 現在のポジションがない場合、エントリーシグナルをチェック
            if current_position is None and not pending_entry:
                position_size = self.position_sizing.calculate(
                    capital=self.current_capital,
                    price=close
                )
                
                # LONGエントリー
                if entry_signals[i] == 1:
                    pending_entry = ('LONG', position_size)
                
                # SHORTエントリー
                elif entry_signals[i] == -1:
                    pending_entry = ('SHORT', position_size)
        
        # 最後のポジションがまだオープンの場合、最終価格でクローズ
        if current_position is not None:
            current_position.close(dates[-1], closes[-1])
            self.trades.append(current_position)
            self.current_capital += current_position.profit_loss
        
        # 結果の分析
        results = self._analyze_results()
        
        return results
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        バックテスト結果を分析する
        
        Returns:
            Dict[str, Any]: 分析結果
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'final_capital': self.current_capital
            }
        
        # 勝率の計算
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        # 総利益と総損失の計算
        total_profit = sum(t.profit_loss for t in self.trades if t.profit_loss > 0)
        total_loss = sum(t.profit_loss for t in self.trades if t.profit_loss < 0)
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit + total_loss,
            'final_capital': self.current_capital
        }
