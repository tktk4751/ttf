from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    """個々のトレード情報を保持するデータクラス"""
    
    position_type: str  # 'LONG' or 'SHORT'
    position_size: float  # 取引数量（USD）
    commission_rate: float  # 手数料率
    slippage_rate: float = 0.001  # 0.1%のスリッページ
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    balance: float = 0.0  # 取引後の残高
    
    def entry(self, date: datetime, price: float) -> None:
        """トレードをエントリーする
        
        Args:
            date: エントリー日時
            price: エントリー価格
        """
        self.entry_date = date
        
        # 取引コストを考慮したエントリー価格を計算
        if self.position_type == 'LONG':
            self.entry_price = price * (1 + self.slippage_rate + self.commission_rate)
        else:  # SHORT
            self.entry_price = price * (1 - self.slippage_rate - self.commission_rate)
    
    def close(self, date: datetime, price: float, current_balance: float) -> None:
        """トレードをクローズする
        
        Args:
            date: エグジット日時
            price: エグジット価格
            current_balance: 現在の残高
        """
        self.exit_date = date
        
        # 取引コストを考慮したエグジット価格を計算
        if self.position_type == 'LONG':
            self.exit_price = price * (1 - self.slippage_rate - self.commission_rate)
        else:  # SHORT
            self.exit_price = price * (1 + self.slippage_rate + self.commission_rate)
        
        # 損益計算
        if self.position_type == 'LONG':
            self.profit_loss = self.position_size * ((self.exit_price / self.entry_price) - 1)
        else:  # SHORT
            self.profit_loss = self.position_size * ((self.entry_price / self.exit_price) - 1)
        
        # 損益率の計算
        self.profit_loss_pct = (self.profit_loss / self.position_size) * 100
        
        # 取引後の残高を更新
        self.balance = current_balance + self.profit_loss