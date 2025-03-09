from typing import Dict, Any, Protocol
from position_sizing.position_sizing import PositionSizingParams

class IPositionManager(Protocol):
    """ポジション管理のインターフェース"""
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        ...

    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """詳細なポジションサイズ計算"""
        ... 