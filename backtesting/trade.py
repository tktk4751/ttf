from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    """個々のトレード情報を保持するデータクラス"""
    
    position_type: str  # 'LONG' or 'SHORT'
    position_size: float
    commission_rate: float = 0.001  # 0.1%の手数料
    slippage_rate: float = 0.001   # 0.1%のスリッページ
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    
    def entry(self, entry_date: datetime, entry_price: float) -> None:
        """トレードをエントリーする
        
        Args:
            entry_date: エントリー日時
            entry_price: エントリー価格
        """
        self.entry_date = entry_date
        
        # エントリー価格にスリッページと手数料を適用
        if self.position_type == 'LONG':
            # 買い注文は高くなる
            self.entry_price = entry_price * (1 + self.slippage_rate + self.commission_rate)
        else:  # SHORT
            # 売り注文は安くなる
            self.entry_price = entry_price * (1 - self.slippage_rate - self.commission_rate)
    
    def close(self, exit_date: datetime, exit_price: float) -> None:
        """トレードをクローズする
        
        Args:
            exit_date: エグジット日時
            exit_price: エグジット価格
        """
        self.exit_date = exit_date
        
        # エグジット価格にスリッページと手数料を適用
        if self.position_type == 'LONG':
            # 売り注文は安くなる
            self.exit_price = exit_price * (1 - self.slippage_rate - self.commission_rate)
        else:  # SHORT
            # 買い注文は高くなる
            self.exit_price = exit_price * (1 + self.slippage_rate + self.commission_rate)
        
        # 損益計算
        if self.position_type == 'LONG':
            self.profit_loss = (self.exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            self.profit_loss = (self.entry_price - self.exit_price) * self.position_size
        
        # 損益率計算
        self.profit_loss_pct = (self.profit_loss / (self.entry_price * self.position_size)) * 100
