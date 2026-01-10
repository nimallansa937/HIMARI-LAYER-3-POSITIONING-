# monitoring/recovery_time_objective.py
"""
#16: Recovery Time Objective
Track max acceptable time from drawdown to new equity high
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrawdownEvent:
    """Record of a drawdown period"""
    start_date: datetime
    end_date: Optional[datetime] = None
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    days_in_drawdown: int = 0
    recovery_complete: bool = False
    
    @property
    def recovery_days(self) -> Optional[int]:
        """Days from trough to recovery (if complete)"""
        if not self.recovery_complete or not self.end_date:
            return None
        return (self.end_date - self.start_date).days


class RecoveryTimeObjective:
    """
    Track Recovery Time Objective (RTO):
    Max acceptable time from drawdown to new equity high.
    """
    
    def __init__(self, max_rto_days: int = 90):
        """
        Args:
            max_rto_days: Maximum acceptable recovery time
        """
        self.max_rto_days = max_rto_days
        self.peak_equity = 0.0
        self.current_drawdown: Optional[DrawdownEvent] = None
        self.historical_drawdowns: List[DrawdownEvent] = []
        self.equity_history: List[Dict] = []
        
    def update(self, current_equity: float, timestamp: Optional[datetime] = None):
        """Update RTO tracker with new equity value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Track equity history
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'peak': self.peak_equity
        })
        
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
            # If we were in drawdown, mark it as recovered
            if self.current_drawdown and not self.current_drawdown.recovery_complete:
                self.current_drawdown.end_date = timestamp
                self.current_drawdown.recovery_complete = True
                self.historical_drawdowns.append(self.current_drawdown)
                logger.info(f"[RTO] Recovered from drawdown in {self.current_drawdown.days_in_drawdown} days")
                self.current_drawdown = None
        
        # Check if in drawdown
        elif current_equity < self.peak_equity:
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
            
            # Start new drawdown
            if self.current_drawdown is None:
                self.current_drawdown = DrawdownEvent(
                    start_date=timestamp,
                    peak_equity=self.peak_equity,
                    trough_equity=current_equity,
                    current_equity=current_equity,
                    max_drawdown_pct=drawdown_pct,
                    days_in_drawdown=0
                )
            
            # Update existing drawdown
            else:
                self.current_drawdown.current_equity = current_equity
                self.current_drawdown.trough_equity = min(
                    self.current_drawdown.trough_equity, 
                    current_equity
                )
                self.current_drawdown.max_drawdown_pct = max(
                    self.current_drawdown.max_drawdown_pct,
                    drawdown_pct
                )
                self.current_drawdown.days_in_drawdown = (
                    timestamp - self.current_drawdown.start_date
                ).days
    
    def is_rto_violated(self) -> bool:
        """Check if current drawdown exceeds RTO"""
        if self.current_drawdown is None:
            return False
        return self.current_drawdown.days_in_drawdown > self.max_rto_days
    
    def get_average_recovery_time(self) -> Optional[float]:
        """Calculate average recovery time from historical drawdowns"""
        completed = [d for d in self.historical_drawdowns if d.recovery_complete]
        
        if not completed:
            return None
        
        recovery_days = [d.recovery_days for d in completed if d.recovery_days is not None]
        return np.mean(recovery_days) if recovery_days else None
    
    def get_max_historical_recovery_time(self) -> Optional[int]:
        """Get longest recovery time from history"""
        completed = [d for d in self.historical_drawdowns if d.recovery_complete]
        
        if not completed:
            return None
        
        recovery_days = [d.recovery_days for d in completed if d.recovery_days is not None]
        return max(recovery_days) if recovery_days else None
    
    def estimate_recovery_time(self, assumed_sharpe: float = 1.2,
                               assumed_volatility: float = 0.6) -> Optional[int]:
        """
        Estimate days until recovery based on current drawdown.
        
        Args:
            assumed_sharpe: Expected Sharpe ratio
            assumed_volatility: Expected annual volatility
            
        Returns:
            Estimated days until recovery
        """
        if self.current_drawdown is None:
            return 0
        
        # Calculate required return to recover
        drawdown_pct = self.current_drawdown.max_drawdown_pct
        required_return = drawdown_pct / (1 - drawdown_pct)
        
        # Expected daily return = Sharpe * Vol / sqrt(252)
        daily_return = (assumed_sharpe * assumed_volatility) / np.sqrt(252)
        
        if daily_return <= 0:
            return None
        
        estimated_days = int(required_return / daily_return)
        
        return estimated_days
    
    def get_current_status(self) -> Dict:
        """Get current RTO status"""
        if self.current_drawdown is None:
            return {
                'in_drawdown': False,
                'rto_violated': False,
                'peak_equity': self.peak_equity,
            }
        
        dd = self.current_drawdown
        return {
            'in_drawdown': True,
            'rto_violated': self.is_rto_violated(),
            'days_in_drawdown': dd.days_in_drawdown,
            'max_rto_days': self.max_rto_days,
            'days_remaining': self.max_rto_days - dd.days_in_drawdown,
            'current_drawdown_pct': dd.max_drawdown_pct,
            'peak_equity': dd.peak_equity,
            'trough_equity': dd.trough_equity,
            'current_equity': dd.current_equity,
            'estimated_recovery_days': self.estimate_recovery_time(),
        }
    
    def generate_rto_report(self) -> str:
        """Generate RTO status report"""
        report = "# Recovery Time Objective Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += f"**Max RTO:** {self.max_rto_days} days\n\n"
        
        # Current status
        if self.current_drawdown:
            dd = self.current_drawdown
            estimated_recovery = self.estimate_recovery_time()
            
            if self.is_rto_violated():
                status = "❌ RTO VIOLATED"
            else:
                status = "🟡 IN DRAWDOWN"
            
            report += f"## Current Drawdown {status}\n\n"
            report += f"- Started: {dd.start_date.date()}\n"
            report += f"- Days in Drawdown: {dd.days_in_drawdown}\n"
            report += f"- Days Remaining: {self.max_rto_days - dd.days_in_drawdown}\n"
            report += f"- Max Drawdown: {dd.max_drawdown_pct:.2%}\n"
            report += f"- Current Equity: ${dd.current_equity:,.2f}\n"
            report += f"- Peak Equity: ${dd.peak_equity:,.2f}\n"
            
            if estimated_recovery:
                report += f"- Estimated Recovery Time: {estimated_recovery} days\n\n"
            
            if self.is_rto_violated():
                report += "⚠️ **ACTION REQUIRED:** Drawdown exceeds maximum RTO\n"
                report += "Consider: Reducing position sizes, reviewing strategy, or pausing trading\n\n"
        
        else:
            report += "## Current Status: ✅ NO DRAWDOWN\n\n"
            report += f"- Peak Equity: ${self.peak_equity:,.2f}\n\n"
        
        # Historical statistics
        if self.historical_drawdowns:
            avg_recovery = self.get_average_recovery_time()
            max_recovery = self.get_max_historical_recovery_time()
            
            report += "## Historical Drawdowns\n\n"
            report += f"- Total Drawdowns: {len(self.historical_drawdowns)}\n"
            if avg_recovery:
                report += f"- Average Recovery Time: {avg_recovery:.0f} days\n"
            if max_recovery:
                report += f"- Max Recovery Time: {max_recovery} days\n"
            report += "\n"
        
        return report
