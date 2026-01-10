# monitoring/execution_quality_attribution.py
"""
#13: Execution Quality Attribution
Separate alpha decay from fill quality, diagnose losses
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeExecution:
    """Record of a single trade execution"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    signal_value: float  # Pre-trade signal
    target_price: float  # Expected mid-price
    fill_price: float    # Actual fill price
    size: float          # Position size in USD
    slippage_bps: float = 0.0
    total_cost_bps: float = 0.0  # Slippage + fees
    
    def __post_init__(self):
        """Calculate slippage from prices"""
        if self.target_price > 0:
            if self.side == 'buy':
                self.slippage_bps = ((self.fill_price - self.target_price) / self.target_price) * 10000
            else:
                self.slippage_bps = ((self.target_price - self.fill_price) / self.target_price) * 10000


@dataclass
class TradeOutcome:
    """Outcome of a completed round-trip trade"""
    entry: TradeExecution
    exit: TradeExecution
    signal_pnl: float = 0.0   # P&L if filled at mid-price (signal quality)
    execution_pnl: float = 0.0  # Execution drag
    
    def __post_init__(self):
        """Calculate P&L components"""
        # Signal P&L: what would have happened with perfect fills
        if self.entry.side == 'buy':
            self.signal_pnl = (self.exit.target_price - self.entry.target_price) / self.entry.target_price
        else:
            self.signal_pnl = (self.entry.target_price - self.exit.target_price) / self.entry.target_price
        self.signal_pnl *= self.entry.size
        
        # Execution P&L: cost of imperfect fills
        entry_slippage = self.entry.slippage_bps / 10000 * self.entry.size
        exit_slippage = self.exit.slippage_bps / 10000 * self.exit.size
        self.execution_pnl = -(entry_slippage + exit_slippage)  # Always negative
    
    @property
    def total_pnl(self) -> float:
        return self.signal_pnl + self.execution_pnl
    
    @property
    def execution_drag(self) -> float:
        """How much execution hurt performance"""
        return self.execution_pnl


class ExecutionQualityAnalyzer:
    """
    Separate alpha decay from fill quality.
    Diagnose whether losses are from bad signals or bad fills.
    """
    
    def __init__(self):
        self.trade_history: List[TradeOutcome] = []
        
    def record_trade(self, outcome: TradeOutcome):
        """Add completed trade to history"""
        self.trade_history.append(outcome)
        
    def record_roundtrip(self, entry: TradeExecution, exit: TradeExecution):
        """Record a complete round-trip trade"""
        outcome = TradeOutcome(entry=entry, exit=exit)
        self.trade_history.append(outcome)
        
    def calculate_signal_performance(self) -> Dict:
        """
        Measure strategy performance assuming perfect execution (mid-price fills).
        This isolates signal quality.
        """
        if not self.trade_history:
            return {'total_trades': 0}
        
        signal_pnls = [t.signal_pnl for t in self.trade_history]
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': np.mean([pnl > 0 for pnl in signal_pnls]),
            'avg_pnl': np.mean(signal_pnls),
            'std_pnl': np.std(signal_pnls) if len(signal_pnls) > 1 else 0,
            'sharpe': np.mean(signal_pnls) / np.std(signal_pnls) if np.std(signal_pnls) > 0 else 0,
            'total_pnl': np.sum(signal_pnls),
        }
    
    def calculate_execution_performance(self) -> Dict:
        """
        Measure execution quality (how much we lose to slippage/fees).
        This isolates fill quality.
        """
        if not self.trade_history:
            return {'total_trades': 0}
        
        execution_pnls = [t.execution_pnl for t in self.trade_history]
        
        avg_entry_slippage = np.mean([t.entry.slippage_bps for t in self.trade_history])
        avg_exit_slippage = np.mean([t.exit.slippage_bps for t in self.trade_history])
        
        return {
            'total_execution_cost': np.sum(execution_pnls),
            'avg_execution_drag_per_trade': np.mean(execution_pnls),
            'avg_entry_slippage_bps': avg_entry_slippage,
            'avg_exit_slippage_bps': avg_exit_slippage,
            'total_roundtrip_cost_bps': avg_entry_slippage + avg_exit_slippage,
        }
    
    def attribution_analysis(self) -> Dict:
        """
        Decompose total P&L into signal vs execution components.
        """
        signal_perf = self.calculate_signal_performance()
        exec_perf = self.calculate_execution_performance()
        
        if not self.trade_history:
            return {}
        
        total_pnl = signal_perf['total_pnl'] + exec_perf['total_execution_cost']
        
        return {
            'total_pnl': total_pnl,
            'signal_contribution': signal_perf['total_pnl'],
            'execution_contribution': exec_perf['total_execution_cost'],
            'signal_pct': signal_perf['total_pnl'] / total_pnl if total_pnl != 0 else 0,
            'execution_pct': exec_perf['total_execution_cost'] / total_pnl if total_pnl != 0 else 0,
            'signal_sharpe': signal_perf['sharpe'],
            'avg_roundtrip_cost_bps': exec_perf['total_roundtrip_cost_bps'],
        }
    
    def diagnose_problem(self) -> str:
        """
        Determine if performance issues are from signals or execution.
        
        Returns:
            Diagnostic message with recommendations
        """
        if not self.trade_history:
            return "# Execution Quality Attribution\n\nNo trades to analyze."
        
        attribution = self.attribution_analysis()
        
        diagnosis = "# Execution Quality Attribution\n\n"
        
        # Overall performance
        diagnosis += f"**Total P&L:** ${attribution['total_pnl']:,.2f}\n\n"
        
        # Attribution breakdown
        diagnosis += f"**Signal Contribution:** ${attribution['signal_contribution']:,.2f}"
        diagnosis += f" ({attribution['signal_pct']*100:.0f}%)\n"
        diagnosis += f"**Execution Drag:** ${attribution['execution_contribution']:,.2f}"
        diagnosis += f" ({attribution['execution_pct']*100:.0f}%)\n\n"
        
        # Signal quality
        diagnosis += f"**Signal Sharpe:** {attribution['signal_sharpe']:.2f}\n"
        diagnosis += f"**Avg Roundtrip Cost:** {attribution['avg_roundtrip_cost_bps']:.1f} bps\n\n"
        
        # Diagnosis
        diagnosis += "## Diagnosis\n\n"
        
        problems = []
        
        if attribution['signal_sharpe'] < 0.5:
            problems.append("❌ **PROBLEM: SIGNAL QUALITY**\n")
            problems.append("- Signal Sharpe < 0.5 indicates weak predictive power\n")
            problems.append("- Recommendation: Review signal validation, check for edge decay\n\n")
        
        if attribution['avg_roundtrip_cost_bps'] > 20:
            problems.append("❌ **PROBLEM: EXECUTION QUALITY**\n")
            problems.append(f"- Roundtrip cost ({attribution['avg_roundtrip_cost_bps']:.1f} bps) is excessive\n")
            problems.append("- Recommendation: Use limit orders, check for low liquidity periods\n\n")
        
        if not problems:
            diagnosis += "✅ **HEALTHY PERFORMANCE**\n"
            diagnosis += "- Signals are working (Sharpe ≥ 0.5)\n"
            diagnosis += "- Execution is efficient (cost < 20 bps)\n\n"
        else:
            diagnosis += "".join(problems)
        
        return diagnosis
    
    def get_trade_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.trade_history:
            return {}
        
        return {
            'total_trades': len(self.trade_history),
            'avg_signal_pnl': np.mean([t.signal_pnl for t in self.trade_history]),
            'avg_execution_drag': np.mean([t.execution_pnl for t in self.trade_history]),
            'avg_total_pnl': np.mean([t.total_pnl for t in self.trade_history]),
        }
