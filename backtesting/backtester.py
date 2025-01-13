from typing import Dict, List, Any, Optional, Tuple, Protocol
import pandas as pd
import numpy as np
from datetime import datetime
from backtesting.trade import Trade
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

class IPositionManager(Protocol):
    """ポジション管理のインターフェース"""
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        ...
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """ポジションサイズを計算"""
        ...

class IStrategy(Protocol):
    """戦略のインターフェース"""
    def generate_entry(self, data: pd.DataFrame) -> np.ndarray:
        """エントリーシグナルを生成"""
        ...
    
    def generate_exit(self, data: pd.DataFrame, position_type: int, current_index: int) -> bool:
        """エグジットシグナルを生成"""
        ...

class Backtester:
    """バックテストの実行クラス"""
    def __init__(
        self,
        strategy: IStrategy,
        position_manager: IPositionManager,
        initial_balance: float,
        commission: float,
        max_positions: int = 1,
        verbose: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            strategy: バックテストする戦略
            position_manager: ポジション管理
            initial_balance: 初期資金
            commission: 手数料率
            max_positions: 同時に保有できる最大ポジション数
            verbose: 詳細なログを出力するかどうか
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.initial_balance = initial_balance
        self.current_capital = initial_balance
        self.commission = commission
        self.max_positions = max_positions
        self.verbose = verbose
        
        # ロガーの設定
        self._setup_logger()
    
    def _setup_logger(self):
        """ロガーの設定"""
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def _log_trade(self, trade: Trade, action: str):
        """トレード情報をログ出力
        
        Args:
            trade: トレード情報
            action: アクション（'ENTRY' or 'EXIT'）
        """
        if not self.verbose:
            return
        
        if action == 'ENTRY':
            logger.info(
                f"[{trade.symbol}] {action} at {trade.entry_date.strftime('%Y-%m-%d %H:%M')} - "
                f"{trade.position_type} @ {trade.entry_price:.2f} "
                f"Size: {trade.position_size:.2f} USD "
                f"Capital: {self.current_capital:.2f} USD"
            )
        else:  # EXIT
            logger.info(
                f"[{trade.symbol}] {action} at {trade.exit_date.strftime('%Y-%m-%d %H:%M')} - "
                f"{trade.position_type} @ {trade.exit_price:.2f} "
                f"PnL: {trade.profit_loss:.2f} USD ({trade.profit_loss_pct:.2f}%) "
                f"Capital: {trade.balance:.2f} USD"
            )

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
            trades = self._run_single_symbol(df, symbol)
            all_trades.extend(trades)
        
        # 日付でソート
        all_trades.sort(key=lambda x: x.entry_date)
        return all_trades
    
    def _run_single_symbol(self, data: pd.DataFrame, symbol: str) -> List[Trade]:
        """
        単一シンボルのバックテストを実行する
        
        Args:
            data: 価格データのDataFrame
            symbol: シンボル名
        
        Returns:
            List[Trade]: トレード結果のリスト
        """
        # データの準備
        dates = data.index
        closes = data['close'].values
        current_position: Optional[Trade] = None
        pending_entry: Optional[tuple] = None  # (position_type, position_size, entry_index)
        pending_exit: bool = False
        trades: List[Trade] = []
        
        # エントリーシグナルの生成
        entry_signals = self.strategy.generate_entry(data)
        
        if self.verbose:
            logger.info(f"\nStarting backtest for {symbol}")
            logger.info(f"Initial capital: {self.current_capital:.2f} USD")
        
        # バックテストのメインループ
        for i in range(1, len(data)):
            date = dates[i]
            close = closes[i]
            
            # 保留中のエグジットの処理
            if pending_exit and current_position is not None:
                current_position.close(date, close, self.current_capital)
                self._log_trade(current_position, 'EXIT')
                trades.append(current_position)
                self.current_capital = current_position.balance
                current_position = None
                pending_exit = False
            
            # 保留中のエントリーの処理
            if pending_entry is not None and current_position is None and self.position_manager.can_enter():
                position_type, position_size, _ = pending_entry
                current_position = Trade(
                    position_type=position_type,
                    position_size=position_size,
                    commission_rate=self.commission,
                    slippage_rate=0.001  # 0.1%のスリッページ
                )
                current_position.symbol = symbol
                current_position.entry(date, close)
                self._log_trade(current_position, 'ENTRY')
                pending_entry = None
            
            # 現在のポジションがある場合、エグジットシグナルをチェック
            if current_position is not None and not pending_exit:
                # ショートの場合は-1、ロングの場合は1を渡す
                position_type = -1 if current_position.position_type == 'SHORT' else 1
                if self.strategy.generate_exit(data, position_type, i):
                    pending_exit = True
            
            # 現在のポジションがない場合、エントリーシグナルをチェック
            if current_position is None and not pending_entry:
                # ポジションサイズの計算
                position_size = self.position_manager.calculate_position_size(
                    price=close,  # 現在の終値でポジションサイズを計算
                    capital=self.current_capital
                )
                
                # LONGエントリー
                if entry_signals[i] == 1:
                    pending_entry = ('LONG', position_size, i)
                
                # SHORTエントリー
                elif entry_signals[i] == -1:
                    pending_entry = ('SHORT', position_size, i)
        
        # 最後のポジションがまだオープンの場合、最終価格でクローズ
        if current_position is not None:
            current_position.close(dates[-1], closes[-1], self.current_capital)
            self._log_trade(current_position, 'EXIT')
            trades.append(current_position)
            self.current_capital = current_position.balance
        
        if self.verbose:
            logger.info(f"\nBacktest completed for {symbol}")
            logger.info(f"Final capital: {self.current_capital:.2f} USD")
            logger.info(f"Total trades: {len(trades)}")
        
        return trades