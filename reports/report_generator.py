"""Professional report generator for backtesting results.

Produces HTML, CSV, and JSON reports with embedded Plotly charts.
PDF export is attempted via weasyprint but falls back gracefully.
"""

from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from reports.charts import generate_all_charts


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #0F172A;
            --paper: #1E293B;
            --border: #334155;
            --text: #E2E8F0;
            --text-muted: #94A3B8;
            --primary: #6366F1;
            --positive: #10B981;
            --negative: #EF4444;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            background: linear-gradient(135deg, var(--primary), #22D3EE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        h2 {{
            font-size: 1.3rem;
            color: var(--text);
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.2rem;
        }}
        .card .label {{ color: var(--text-muted); font-size: 0.85rem; }}
        .card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }}
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .chart-container {{
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.6rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-muted); font-weight: 600; font-size: 0.85rem; }}
        td {{ font-family: 'Fira Code', monospace; }}
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    <p class="subtitle">Generated: {timestamp} | Symbol: {symbol}</p>

    <h2>Performance Summary</h2>
    <div class="grid">
        {performance_cards}
    </div>

    <h2>Risk Metrics</h2>
    <div class="grid">
        {risk_cards}
    </div>

    <h2>Trade Statistics</h2>
    <div class="grid">
        {trade_cards}
    </div>

    {charts_html}

    {optimization_html}

    <div class="footer">
        Market Regime Bot — Phase 7 Backtest Report
    </div>
</div>
</body>
</html>
"""


def _metric_card(label: str, value: Any, is_pct: bool = False, color_coded: bool = False) -> str:
    """Generate HTML for a single metric card."""
    if isinstance(value, float):
        display = f"{value:.2f}{'%' if is_pct else ''}"
    else:
        display = str(value)

    css_class = ""
    if color_coded and isinstance(value, (int, float)):
        css_class = "positive" if value > 0 else "negative" if value < 0 else ""

    return f"""<div class="card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{display}</div>
    </div>"""


class ReportGenerator:
    """Generate professional backtesting reports.

    Args:
        results: Complete backtest results dictionary.
        symbol: Instrument ticker.
        output_dir: Directory for report output files.
    """

    def __init__(
        self,
        results: dict[str, Any],
        symbol: str = "UNKNOWN",
        output_dir: str | Path = "./reports",
    ) -> None:
        self._results = results
        self._symbol = symbol
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(self, filename: str = "backtest_report.html") -> Path:
        """Generate an HTML report.

        Args:
            filename: Output filename.

        Returns:
            Path to the generated HTML file.
        """
        html = self._build_html()
        path = self._output_dir / filename
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to {}", path)
        return path

    def generate_pdf(self, filename: str = "backtest_report.pdf") -> Path | None:
        """Generate a PDF report (requires weasyprint).

        Args:
            filename: Output filename.

        Returns:
            Path to the PDF file, or None if weasyprint is unavailable.
        """
        html_path = self.generate_html("_temp_report.html")
        pdf_path = self._output_dir / filename

        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            logger.info("PDF report saved to {}", pdf_path)
            html_path.unlink(missing_ok=True)
            return pdf_path
        except ImportError:
            logger.warning("weasyprint not installed — PDF export unavailable. HTML report retained.")
            final_html = self._output_dir / filename.replace(".pdf", ".html")
            html_path.rename(final_html)
            return final_html
        except Exception as exc:
            logger.error("PDF generation failed: {}", exc)
            return html_path

    def generate_csv(self, filename: str = "trades.csv") -> Path:
        """Export trade data to CSV.

        Args:
            filename: Output filename.

        Returns:
            Path to the CSV file.
        """
        trades = self._results.get("trades", [])
        path = self._output_dir / filename

        if not trades:
            path.write_text("No trades\n", encoding="utf-8")
            return path

        fieldnames = list(trades[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

        logger.info("CSV report saved to {} ({} trades)", path, len(trades))
        return path

    def generate_json(self, filename: str = "backtest_results.json") -> Path:
        """Export full results to JSON.

        Args:
            filename: Output filename.

        Returns:
            Path to the JSON file.
        """
        path = self._output_dir / filename

        export = {k: v for k, v in self._results.items() if k != "equity_series"}
        if "equity_series" in self._results:
            es = self._results["equity_series"]
            if isinstance(es, pd.Series):
                export["equity_summary"] = {
                    "start": float(es.iloc[0]) if len(es) > 0 else 0,
                    "end": float(es.iloc[-1]) if len(es) > 0 else 0,
                    "length": len(es),
                }

        def _serialise(obj: Any) -> Any:
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        import numpy as np

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=_serialise)

        logger.info("JSON report saved to {}", path)
        return path

    def generate_all(self) -> dict[str, Path]:
        """Generate all report formats.

        Returns:
            Dictionary mapping format names to file paths.
        """
        paths: dict[str, Path] = {
            "html": self.generate_html(),
            "csv": self.generate_csv(),
            "json": self.generate_json(),
        }

        pdf = self.generate_pdf()
        if pdf:
            paths["pdf"] = pdf

        return paths

    # ------------------------------------------------------------------
    # HTML building
    # ------------------------------------------------------------------

    def _build_html(self) -> str:
        """Assemble the full HTML report."""
        perf = self._results.get("analytics", {}).get("performance", {})
        risk = self._results.get("analytics", {}).get("risk", {})
        trade_stats = self._results.get("analytics", {}).get("trades", {})

        perf_cards = "".join([
            _metric_card("Total Return", self._results.get("total_return", 0), True, True),
            _metric_card("Annual Return", self._results.get("annual_return", 0), True, True),
            _metric_card("CAGR", perf.get("cagr", 0), True, True),
            _metric_card("Sharpe Ratio", self._results.get("sharpe_ratio", 0), color_coded=True),
            _metric_card("Sortino Ratio", self._results.get("sortino_ratio", 0), color_coded=True),
            _metric_card("Max Drawdown", self._results.get("max_drawdown", 0), True),
            _metric_card("Profit Factor", self._results.get("profit_factor", 0)),
            _metric_card("Win Rate", self._results.get("win_rate", 0), True),
        ])

        risk_cards = "".join([
            _metric_card("Volatility", risk.get("volatility", 0), True),
            _metric_card("VaR (95%)", risk.get("var_95", 0), True),
            _metric_card("CVaR (95%)", risk.get("cvar_95", 0), True),
            _metric_card("Ulcer Index", risk.get("ulcer_index", 0)),
            _metric_card("Beta", risk.get("beta", "N/A")),
            _metric_card("Alpha", risk.get("alpha", "N/A"), True),
        ])

        winning = trade_stats.get("winning", {})
        losing = trade_stats.get("losing", {})
        trade_cards = "".join([
            _metric_card("Total Trades", self._results.get("total_trades", 0)),
            _metric_card("Winning Trades", winning.get("count", 0)),
            _metric_card("Losing Trades", losing.get("count", 0)),
            _metric_card("Best Trade", self._results.get("best_trade", 0), True, True),
            _metric_card("Worst Trade", self._results.get("worst_trade", 0), True, True),
            _metric_card("Avg Trade", self._results.get("average_trade", 0), True, True),
            _metric_card("Win Streak", trade_stats.get("longest_win_streak", 0)),
            _metric_card("Loss Streak", trade_stats.get("longest_loss_streak", 0)),
        ])

        charts_html = self._generate_charts_html()

        opt = self._results.get("optimization", {})
        optimization_html = ""
        if opt:
            optimization_html = f"""
            <h2>Optimization Results</h2>
            <div class="card">
                <p><strong>Method:</strong> {opt.get('method', 'N/A')}</p>
                <p><strong>Best Params:</strong> {opt.get('best_params', {})}</p>
                <p><strong>Evaluated:</strong> {opt.get('total_evaluated', 0)} combinations</p>
            </div>
            """

        return _HTML_TEMPLATE.format(
            title="Backtest Report",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            symbol=self._symbol,
            performance_cards=perf_cards,
            risk_cards=risk_cards,
            trade_cards=trade_cards,
            charts_html=charts_html,
            optimization_html=optimization_html,
        )

    def _generate_charts_html(self) -> str:
        """Generate HTML for embedded Plotly charts."""
        equity = self._results.get("equity_series")
        trades = self._results.get("trades", [])

        if equity is None or (isinstance(equity, pd.Series) and len(equity) < 2):
            return ""

        try:
            charts = generate_all_charts(
                equity_series=equity,
                trades=trades,
                regimes=self._results.get("regime_log"),
            )

            html_parts = []
            for name, fig in charts.items():
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
                html_parts.append(
                    f'<div class="chart-container">{chart_html}</div>'
                )

            return "\n".join(html_parts)

        except Exception as exc:
            logger.warning("Chart generation failed: {}", exc)
            return '<p class="subtitle">Charts could not be generated.</p>'
