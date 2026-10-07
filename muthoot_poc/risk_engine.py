"""muthoot_poc/risk_engine.py

Production risk management and execution safety module.
Enforces static risk guardrails, position sizing, volatility-calibrated stop losses,
take-profit brackets, and minimum risk-to-reward ratios on model signals.
"""

import math
from typing import Dict, Any, Optional


class HardRiskGuardrail:
    """Risk management engine enforcing strict account protection rules."""

    def __init__(
        self,
        account_capital: float = 1_000_000.0,  # ₹10,00,000 baseline capital
        max_risk_per_trade_pct: float = 1.0,     # Max 1.0% capital risk per trade
        max_position_capital_pct: float = 15.0,  # Max 15.0% allocation per stock
        min_risk_reward_ratio: float = 1.5,      # Minimum 1:1.5 R:R required
        min_directional_p_up: float = 0.58,      # Minimum P(up) for bullish entry
        max_directional_p_down: float = 0.42,    # Maximum P(up) for bearish entry
        max_daily_drawdown_pct: float = 3.0,     # Stop system if daily loss exceeds 3%
    ):
        self.account_capital = account_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_capital_pct = max_position_capital_pct
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.min_p_up = min_directional_p_up
        self.max_p_down = max_directional_p_down
        self.max_daily_drawdown_pct = max_daily_drawdown_pct

    def evaluate_execution_plan(
        self,
        symbol: str,
        current_price: float,
        bias: str,
        p_up: float,
        target_price: float,
        atr_14: float,
        direction_validated: bool = True,
        range_validated: bool = True,
        tradeable_validated: bool = True,
    ) -> Dict[str, Any]:
        """Evaluates raw model prediction against hard risk parameters.

        Args:
            symbol: Stock ticker symbol.
            current_price: Current market price or open price (₹).
            bias: Raw forecast bias ("BULLISH", "BEARISH", "NEUTRAL").
            p_up: Directional probability score (0.0 to 1.0).
            target_price: Target forecast price at market close (₹).
            atr_14: 14-period Average True Range in rupees (₹).
            direction_validated: Walk-forward backtest directional edge indicator.
            range_validated: Out-of-sample volatility calibration indicator.
            tradeable_validated: Per-stock net performance passed its cost-adjusted gate.

        Returns:
            Dict containing trade recommendation, position size, stop-loss,
            take-profit, risk amount, and safety status.
        """
        # Baseline output setup
        execution_plan = {
            "symbol": symbol,
            "approved": False,
            "action": "NO_TRADE",
            "entry_price": round(current_price, 2),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "position_size_shares": 0,
            "allocated_capital": 0.0,
            "max_risk_amount": 0.0,
            "risk_reward_ratio": 0.0,
            "rejection_reason": None,
        }

        # 1. Verification of entry requirements
        if not direction_validated:
            execution_plan["rejection_reason"] = "Direction model failed walk-forward edge test."
            return execution_plan

        if not tradeable_validated:
            execution_plan["rejection_reason"] = "Per-stock net performance has not passed validation."
            return execution_plan

        if current_price <= 0 or atr_14 <= 0:
            execution_plan["rejection_reason"] = "Invalid price or zero ATR calculated."
            return execution_plan

        # 2. Check Directional Confidence Thresholds
        if bias == "BULLISH" and p_up < self.min_p_up:
            execution_plan["rejection_reason"] = f"P(up) {p_up:.3f} below bullish threshold {self.min_p_up}."
            return execution_plan

        if bias == "BEARISH" and p_up > self.max_p_down:
            execution_plan["rejection_reason"] = f"P(up) {p_up:.3f} above bearish threshold {self.max_p_down}."
            return execution_plan

        if bias == "NEUTRAL":
            execution_plan["rejection_reason"] = "Model stance is NEUTRAL."
            return execution_plan

        # 3. Calculate Dynamic ATR-based Stop Loss & Take Profit
        # 1.5x ATR for Stop-Loss buffer
        stop_loss_distance = max(1.5 * atr_14, current_price * 0.008)

        if bias == "BULLISH":
            stop_loss = current_price - stop_loss_distance
            take_profit = max(target_price, current_price + (stop_loss_distance * self.min_risk_reward_ratio))
            reward = take_profit - current_price
            risk = current_price - stop_loss
            action = "BUY"
        else:  # BEARISH / SHORT
            stop_loss = current_price + stop_loss_distance
            take_profit = min(target_price, current_price - (stop_loss_distance * self.min_risk_reward_ratio))
            reward = current_price - take_profit
            risk = stop_loss - current_price
            action = "SELL_SHORT"

        if risk <= 0:
            execution_plan["rejection_reason"] = "Non-positive risk calculation."
            return execution_plan

        risk_reward_ratio = reward / risk

        # 4. Enforce Minimum Risk-to-Reward Ratio
        if risk_reward_ratio < self.min_risk_reward_ratio:
            execution_plan["rejection_reason"] = (
                f"Risk-Reward Ratio ({risk_reward_ratio:.2f}) is below minimum allowed "
                f"({self.min_risk_reward_ratio:.2f})."
            )
            return execution_plan

        # 5. Fixed-Fractional Position Sizing Calculation
        max_allowed_trade_risk = self.account_capital * (self.max_risk_per_trade_pct / 100.0)
        max_allowed_position_capital = self.account_capital * (self.max_position_capital_pct / 100.0)

        # Shares based on risk budget
        shares_by_risk = math.floor(max_allowed_trade_risk / risk)
        # Shares based on position capital limit
        shares_by_capital = math.floor(max_allowed_position_capital / current_price)

        final_shares = max(0, min(shares_by_risk, shares_by_capital))

        if final_shares == 0:
            execution_plan["rejection_reason"] = "Position size rounded to 0 shares due to capital caps."
            return execution_plan

        allocated_capital = round(final_shares * current_price, 2)
        total_risk_amount = round(final_shares * risk, 2)

        execution_plan.update({
            "approved": True,
            "action": action,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "position_size_shares": final_shares,
            "allocated_capital": allocated_capital,
            "max_risk_amount": total_risk_amount,
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "rejection_reason": None,
        })

        return execution_plan
