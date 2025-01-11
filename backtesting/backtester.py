from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from strategies.strategy import Strategy
from position_sizing.position_sizing import PositionSizing
from backtesting.trade import Trade


class Backtester:
    """バックテストの実行クラス"""
    
    def __init__(
        self,
        strategy: Strategy,
        position_sizing: PositionSizing,
        initial_balance: float,
        commission: float,
        max_positions: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            strategy: バックテストする戦略
            position_sizing: ポジションサイズ計算ロジック
            initial_balance: 初期資金
            commission: 手数料率
            max_positions: 同時に保有できる最大ポジション数
        """
        self.strategy = strategy
        self.position_sizing = position_sizing
        self.initial_balance = initial_balance
        self.current_capital = initial_balance
        self.commission = commission
        self.max_positions = max_positions
        self.trades: List[Trade] = []
    
    def run(self, data: Dict[str, pd.DataFrame]) -> List[Trade]:
        """
        バックテストを実行する
        
        Args:
            data: バックテストに使用するデータ
                  キー: シンボル名
                  値: 価格データのDataFrame
        
        Returns:
            List[Trade]: 全トレード結果のリスト
        """
        all_trades = []
        
        # 各シンボルに対してバックテストを実行
        for symbol, df in data.items():
            trades = self._run_single_symbol(df)
            all_trades.extend(trades)
        
        # 日付でソート
        all_trades.sort(key=lambda x: x.entry_date)
        return all_trades
    
    def _run_single_symbol(self, data: pd.DataFrame) -> List[Trade]:
        """
        単一シンボルのバックテストを実行する
        
        Args:
            data: 価格データのDataFrame
        
        Returns:
            List[Trade]: トレード結果のリスト
        """
        # データの準備
        dates = data.index
        opens = data['open'].values
        closes = data['close'].values
        current_position: Optional[Trade] = None
        pending_entry: Optional[tuple] = None  # (position_type, position_size)
        pending_exit: bool = False
        trades: List[Trade] = []
        
        # エントリーシグナルの生成
        entry_signals = self.strategy.generate_entry(data)
        
        # バックテストのメインループ
        for i in range(1, len(data)):
            date = dates[i]
            open_price = opens[i]
            close = closes[i]
            
            # 保留中のエグジットの処理
            if pending_exit and current_position is not None:
                current_position.close(date, open_price, self.current_capital)
                trades.append(current_position)
                self.current_capital = current_position.balance
                current_position = None
                pending_exit = False
            
            # 保留中のエントリーの処理
            if pending_entry is not None and current_position is None:
                position_type, position_size = pending_entry
                current_position = Trade(
                    position_type=position_type,
                    position_size=position_size,
                    commission_rate=self.commission,
                    slippage_rate=0.001  # 0.1%のスリッページ
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
                # ポジションサイズの計算
                position_size = self.position_sizing.calculate(
                    capital=self.current_capital,
                    price=open_price
                )
                
                # LONGエントリー
                if entry_signals[i] == 1:
                    pending_entry = ('LONG', position_size)
                
                # SHORTエントリー
                elif entry_signals[i] == -1:
                    pending_entry = ('SHORT', position_size)
        
        # 最後のポジションがまだオープンの場合、最終価格でクローズ
        if current_position is not None:
            current_position.close(dates[-1], closes[-1], self.current_capital)
            trades.append(current_position)
            self.current_capital = current_position.balance
        
        return trades
