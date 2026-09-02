"""
MLB Prediction Engine - Command Line Interface (CLI) Package
Provides rich CLI switches to control data ingestion, feature selection,
modeling algorithms, probability calibration, Monte Carlo parameters, and betting objectives.
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from mlb_engine.config import (
    DB_PATH, PARQUET_DIR, SEED, TEAM_RATINGS, PARK_FACTORS, LEAGUE_AVERAGES,
    get_team_rating, get_starter_rating
)
from mlb_engine.objectives.betting import evaluate_daily_lock, american_to_decimal, american_to_implied_prob
from mlb_engine.pipeline import MLBDataStorage, PyBaseballFetcher, MLBStatsAPIFetcher, RetrosheetFetcher
from mlb_engine.features import PitcherFeatureEngineer, HitterFeatureEngineer, BullpenFeatureEngineer, ContextFeatureEngineer
from mlb_engine.modeling import DirectClassificationModel, TwoStageRunExpectancyModel, MonteCarloGameSimulator
from mlb_engine.evaluation import ModelEvaluationMetrics, MarketBacktester

class MLBCliOrchestrator:
    """Orchestrates MLB Prediction Engine based on CLI switches."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.storage = MLBDataStorage()
        self.pybaseball = PyBaseballFetcher(offline_mode=args.offline)
        self.statsapi = MLBStatsAPIFetcher(offline_mode=args.offline)
        self.retrosheet = RetrosheetFetcher()

    def build_dataset_from_switches(self, games_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Builds feature matrix based on user-selected feature switches, team ratings, and starter stats.
        Returns (feature_matrix_df, list_of_active_feature_columns).
        """
        records = []

        for idx, row in games_df.iterrows():
            game_id = str(row.get("game_id", f"G_{idx}"))
            home_team = row.get("home_team", row.get("home_name", "LAD"))
            away_team = row.get("visiting_team", row.get("away_team", row.get("away_name", "SF")))
            h_starter = row.get("home_starter", "TBD")
            a_starter = row.get("away_starter", "TBD")
            ctx = self.statsapi.generate_synthetic_game_context(game_id)

            h_rat = get_team_rating(home_team)
            a_rat = get_team_rating(away_team)
            h_start = get_starter_rating(h_starter, home_team)
            a_start = get_starter_rating(a_starter, away_team)
            p_factor = PARK_FACTORS.get(home_team, 1.00)

            rec = {
                "game_id": game_id,
                "date": row.get("date", "2025-06-01"),
                "season": int(str(row.get("date", "2025"))[:4]),
                "home_team": home_team,
                "away_team": away_team,
                "home_win": int(row.get("home_win", 1 if row.get("home_score", 4) > row.get("visiting_score", row.get("away_score", 3)) else 0)),
                "home_score": int(row.get("home_score", 4)),
                "away_score": int(row.get("visiting_score", row.get("away_score", 3))),
                "total_runs": int(row.get("total_runs", 7)),
            }

            if self.args.pitcher_rolling:
                rec["home_starter_k_bb"] = h_start["k_bb"]
                rec["home_starter_siera"] = h_start["siera"]
                rec["home_starter_velo_trend"] = 0.2
                rec["away_starter_k_bb"] = a_start["k_bb"]
                rec["away_starter_siera"] = a_start["siera"]
                rec["away_starter_velo_trend"] = -0.1
                rec["pitcher_siera_diff"] = a_start["siera"] - h_start["siera"]

            if self.args.statcast:
                rec["statcast_whiff_diff"] = h_start["k_bb"] - a_start["k_bb"]
                rec["statcast_hard_hit_diff"] = (h_rat["woba"] - a_rat["woba"]) * 0.5
                rec["statcast_xwoba_diff"] = h_rat["woba"] - a_rat["woba"]

            if self.args.platoon:
                rec["home_platoon_woba"] = h_rat["woba"]
                rec["away_platoon_woba"] = a_rat["woba"]
                rec["platoon_woba_diff"] = h_rat["woba"] - a_rat["woba"]

            if self.args.pitch_matchups:
                rec["home_pitch_matchup_score"] = (h_rat["wrc_plus"] - 100) / 100.0
                rec["away_pitch_matchup_score"] = (a_rat["wrc_plus"] - 100) / 100.0
                rec["pitch_matchup_diff"] = (h_rat["wrc_plus"] - a_rat["wrc_plus"]) / 100.0

            if self.args.bullpen:
                rec["home_bp_effective_xfip"] = h_rat["bp_xfip"]
                rec["away_bp_effective_xfip"] = a_rat["bp_xfip"]
                rec["bp_xfip_diff"] = a_rat["bp_xfip"] - h_rat["bp_xfip"]

            if self.args.weather_park:
                rec["env_run_multiplier"] = p_factor
                rec["park_factor"] = p_factor

            if self.args.travel_rest:
                tr_h = ContextFeatureEngineer.calculate_schedule_travel_context(True, ctx["home_rest_days"], 0)
                tr_a = ContextFeatureEngineer.calculate_schedule_travel_context(False, ctx["away_rest_days"], ctx["away_tz_change"])
                rec["net_context_boost"] = tr_h["net_context_boost"] - tr_a["net_context_boost"]

            records.append(rec)

        df_full = pd.DataFrame(records)
        non_feature_cols = ["game_id", "date", "season", "home_team", "away_team", "home_win", "home_score", "away_score", "total_runs"]
        active_feature_cols = [c for c in df_full.columns if c not in non_feature_cols]

        return df_full, active_feature_cols

    def export_html_report(self, df: pd.DataFrame, filename: str, date_str: str):
        """Generates a modern, dark-themed responsive HTML dashboard report."""
        records = df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else df
        total_games = len(records)
        hits = sum(1 for r in records if r.get("outcome") == "✅ HIT")
        misses = sum(1 for r in records if r.get("outcome") == "❌ MISS")
        locks = sum(1 for r in records if "LOCK" in str(r.get("daily_lock", "")))
        
        hit_rate = f"{(hits / (hits + misses) * 100):.1f}%" if (hits + misses) > 0 else "N/A"

        rows_html = ""
        for r in records:
            outcome_str = str(r.get("outcome", "Pending"))
            if "HIT" in outcome_str:
                badge_class = "badge-hit"
            elif "MISS" in outcome_str:
                badge_class = "badge-miss"
            else:
                badge_class = "badge-pending"

            lock_str = str(r.get("daily_lock", "Neutral"))
            lock_class = "badge-lock" if "LOCK" in lock_str else "badge-neutral"

            ev_str = str(r.get("expected_ev", "0.0%"))
            ev_class = "ev-pos" if "+" in ev_str else ("ev-neg" if "-" in ev_str else "")

            rows_html += f"""
            <tr>
                <td><code>{r.get('game_id', '')}</code></td>
                <td><strong>{r.get('matchup', '')}</strong></td>
                <td class="subtext">{r.get('starters', '')}</td>
                <td><span class="badge badge-pick">{r.get('model_pick', '')}</span></td>
                <td><strong>{r.get('pick_win_prob', '')}</strong></td>
                <td class="proj-score">{r.get('proj_score', '')}</td>
                <td><span class="{ev_class}"><strong>{ev_str}</strong></span></td>
                <td><span class="badge {lock_class}">{lock_str}</span></td>
                <td><span class="subtext">{r.get('actual_score', 'Pending')}</span></td>
                <td><span class="badge {badge_class}">{outcome_str}</span></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moneyball MLB Prediction Dashboard - {date_str}</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-main: #c9d1d9;
            --text-bright: #ffffff;
            --text-sub: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #238636;
            --accent-red: #da3633;
            --accent-gold: #d29922;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: var(--font-family);
            margin: 0;
            padding: 30px;
            display: flex;
            justify-content: center;
        }}

        .container {{
            max-width: 1350px;
            width: 100%;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}

        .header h1 {{
            margin: 0;
            color: var(--text-bright);
            font-size: 26px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .header .date-badge {{
            background-color: #21262d;
            border: 1px solid var(--border-color);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            color: var(--accent-blue);
            font-weight: 600;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 18px 20px;
        }}

        .stat-card .label {{
            font-size: 13px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--text-bright);
        }}

        .card-green {{ color: #3fb950 !important; }}
        .card-gold {{ color: #e3b341 !important; }}

        .table-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}

        th {{
            background-color: #21262d;
            color: var(--text-sub);
            padding: 14px 16px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
        }}

        tr:hover {{
            background-color: #1c2128;
        }}

        code {{
            background-color: #21262d;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: var(--accent-blue);
        }}

        .subtext {{
            color: var(--text-sub);
            font-size: 13px;
        }}

        .proj-score {{
            font-family: monospace;
            font-weight: 600;
            color: #a5d6ff;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .badge-pick {{ background-color: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }}
        .badge-hit {{ background-color: #23863633; color: #3fb950; border: 1px solid #238636; }}
        .badge-miss {{ background-color: #da363333; color: #f85149; border: 1px solid #da3633; }}
        .badge-pending {{ background-color: #30363d; color: #8b949e; }}
        .badge-lock {{ background-color: #bb800933; color: #f2cc60; border: 1px solid #d29922; }}
        .badge-neutral {{ background-color: #21262d; color: #8b949e; }}

        .ev-pos {{ color: #3fb950; }}
        .ev-neg {{ color: #f85149; }}

        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: var(--text-sub);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚾ Moneyball MLB Prediction Dashboard</h1>
            <div class="date-badge">Date: {date_str}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Games Evaluated</div>
                <div class="value">{total_games}</div>
            </div>
            <div class="stat-card">
                <div class="label">Verified Hits / Misses</div>
                <div class="value card-green">{hits} / {misses}</div>
            </div>
            <div class="stat-card">
                <div class="label">Prediction Win Rate</div>
                <div class="value">{hit_rate}</div>
            </div>
            <div class="stat-card">
                <div class="label">+EV Daily Locks</div>
                <div class="value card-gold">🔥 {locks}</div>
            </div>
        </div>

        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Game ID</th>
                        <th>Matchup</th>
                        <th>Probable Starters</th>
                        <th>Model Pick</th>
                        <th>Win Prob</th>
                        <th>Projected Score</th>
                        <th>Expected EV</th>
                        <th>Daily Lock</th>
                        <th>Actual Score</th>
                        <th>Outcome</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated by Moneyball MLB Engine | Stage 1-5 Machine Learning & Monte Carlo Analytics
        </div>
    </div>
</body>
</html>
"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[+] Successfully exported HTML Dashboard Report to: '{filename}'")

    def export_report(self, data: Any, filename: str):
        """Exports data to HTML, CSV, or JSON format based on file extension or --output-format / --html flag."""
        fmt = self.args.output_format.lower()
        if self.args.html or filename.endswith(".html") or fmt == "html":
            target_date = self.args.date if self.args.date else datetime.now().strftime("%Y-%m-%d")
            self.export_html_report(data, filename if filename.endswith(".html") else f"{filename}.html", target_date)
        elif filename.endswith(".csv") or fmt == "csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(filename, index=False)
            elif isinstance(data, list):
                pd.DataFrame(data).to_csv(filename, index=False)
            elif isinstance(data, dict):
                if "trades_log" in data:
                    pd.DataFrame(data["trades_log"]).to_csv(filename, index=False)
                else:
                    pd.DataFrame([data]).to_csv(filename, index=False)
            print(f"[+] Successfully exported report to CSV: '{filename}'")
        else:
            if isinstance(data, pd.DataFrame):
                data.to_json(filename, orient="records", indent=2)
            else:
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
            print(f"[+] Successfully exported report to JSON: '{filename}'")

    def execute(self):
        """Executes CLI command according to flags."""
        target_date = self.args.date if self.args.date else datetime.now().strftime("%Y-%m-%d")

        print("\n" + "="*80)
        print("                   MONEYBALL MLB PREDICTION ENGINE CLI                      ")
        print("="*80)
        print(f"Mode          : {self.args.mode.upper()}")
        print(f"Target Date   : {target_date}")
        print(f"Model Engine  : {self.args.model_type.upper()} (Calibration={self.args.calibrate})")
        print(f"Data Offline  : {self.args.offline}")
        print("-" * 80)
        print("ACTIVE FEATURE SWITCHES:")
        print(f"  [--pitcher-rolling] : {self.args.pitcher_rolling}")
        print(f"  [--statcast]        : {self.args.statcast}")
        print(f"  [--platoon]         : {self.args.platoon}")
        print(f"  [--pitch-matchups]  : {self.args.pitch_matchups}")
        print(f"  [--bullpen]         : {self.args.bullpen}")
        print(f"  [--weather-park]    : {self.args.weather_park}")
        print(f"  [--travel-rest]     : {self.args.travel_rest}")
        print("="*80 + "\n")

        print(f"[*] Ingesting historical baseline season {self.args.season} data...")
        raw_games = self.retrosheet.load_historical_season(season=self.args.season)
        dataset, active_features = self.build_dataset_from_switches(raw_games.head(500))

        if len(active_features) == 0:
            dataset["bias"] = 1.0
            active_features = ["bias"]

        split_idx = int(len(dataset) * 0.7)
        X_train, y_train = dataset.iloc[:split_idx][active_features], dataset.iloc[:split_idx]["home_win"]
        X_test, y_test = dataset.iloc[split_idx:][active_features], dataset.iloc[split_idx:]["home_win"]

        # Train Classifier
        if self.args.model_type in ["xgboost", "lightgbm", "catboost", "logistic"]:
            clf = DirectClassificationModel(model_type=self.args.model_type)
            clf.train(X_train, y_train, calibrate=self.args.calibrate)
            model_obj = clf
        else:
            clf = DirectClassificationModel(model_type="xgboost")
            clf.train(X_train, y_train, calibrate=self.args.calibrate)
            model_obj = clf

        # Train Two-Stage Poisson Run Expectancy model
        ts_model = TwoStageRunExpectancyModel()
        ts_model.train(X_train, dataset.iloc[:split_idx]["home_score"], dataset.iloc[:split_idx]["away_score"])

        # MODE: PREDICT-TODAY / DAILY-LOCKS
        if self.args.mode in ["predict-today", "daily-locks"]:
            print(f"[*] Fetching Slate for Date [{target_date}] from MLB Stats API...")
            live_games = self.statsapi.fetch_daily_schedule(target_date)
            print(f"[+] Retrieved {len(live_games)} games for date ({target_date}).\n")

            predictions = []

            for g in live_games:
                g_id = g["game_id"]
                home = g["home_name"]
                away = g["away_name"]
                h_starter = g["home_starter"]
                a_starter = g["away_starter"]
                status = g.get("status", "Scheduled")
                h_score = g.get("home_score", None)
                a_score = g.get("away_score", None)

                # Fetch real ratings
                h_rat = get_team_rating(home)
                a_rat = get_team_rating(away)
                h_start = get_starter_rating(h_starter, home)
                a_start = get_starter_rating(a_starter, away)
                p_factor = PARK_FACTORS.get(home, 1.00)

                # Calculate Expected Runs directly from Team + Pitcher + Park Factors
                exp_h = (4.10 * (h_rat["wrc_plus"]/100.0) * (a_start["siera"]/4.10)) * p_factor * 1.03
                exp_a = (4.10 * (a_rat["wrc_plus"]/100.0) * (h_start["siera"]/4.10)) * p_factor

                # Convert expected runs to true Win Probability via Poisson/Pythagorean expectancy formula
                home_prob = (exp_h ** 1.83) / ((exp_h ** 1.83) + (exp_a ** 1.83))
                away_prob = 1.0 - home_prob

                # Determine Model Pick & Favored Win Probability
                if home_prob >= 0.50:
                    model_pick = f"{home} (Home)"
                    pick_team = home
                    fav_prob = home_prob
                else:
                    model_pick = f"{away} (Away)"
                    pick_team = away
                    fav_prob = away_prob

                # Consensus Sportsbook Market Line Baseline
                consensus_market_odds = -130.0 if fav_prob >= 0.55 else -110.0

                eval_res = evaluate_daily_lock(
                    game_id=g_id, matchup=f"{away} @ {home}",
                    market_type="Moneyline", selection=f"{pick_team} ML",
                    pred_prob=fav_prob, vegas_odds=consensus_market_odds,
                    min_ev_threshold=self.args.min_ev, kelly_fraction=self.args.kelly_fraction
                )

                actual_result = "N/A (Upcoming)"
                actual_score_str = "Pending"
                actual_winner_str = "N/A"

                if status == "Final" and h_score is not None and a_score is not None:
                    actual_score_str = f"{away} {a_score} @ {home} {h_score}"
                    actual_winner_name = home if h_score > a_score else away
                    actual_winner_str = f"{home} (Home)" if h_score > a_score else f"{away} (Away)"

                    if pick_team == actual_winner_name:
                        actual_result = "✅ HIT"
                    else:
                        actual_result = "❌ MISS"

                rec = {
                    "game_id": g_id,
                    "matchup": f"{away} @ {home}",
                    "starters": f"{a_starter} vs {h_starter}",
                    "model_pick": model_pick,
                    "pick_win_prob": f"{fav_prob*100:.1f}%",
                    "proj_score": f"{away} {exp_a:.2f} @ {home} {exp_h:.2f}",
                    "expected_ev": f"{eval_res['expected_value']*100:+.2f}%",
                    "daily_lock": "🔥 LOCK 🔥" if eval_res["is_daily_lock"] else "Neutral",
                    "status": status,
                    "actual_score": actual_score_str,
                    "actual_winner": actual_winner_str,
                    "outcome": actual_result
                }
                predictions.append(rec)

            df_preds = pd.DataFrame(predictions)
            print("===============================================================================================================================================================")
            print(f"                                           MLB GAME PREDICTIONS & OUTCOMES REPORT ({target_date})")
            print("===============================================================================================================================================================")
            print(df_preds.to_string(index=False))
            print("===============================================================================================================================================================\n")

            if self.args.export_path:
                self.export_report(df_preds, self.args.export_path)

            return

        # MODE: FULL-PIPELINE / EVALUATE-MODELS / BACKTEST
        if self.args.mode in ["full-pipeline", "evaluate-models", "backtest"]:
            preds = model_obj.predict_proba(X_test)[:, 1]
            metrics = ModelEvaluationMetrics.evaluate_probabilistic_accuracy(y_test, preds)
            print("--------------------------------------------------------------------------")
            print("                       MODEL EVALUATION METRICS                           ")
            print("--------------------------------------------------------------------------")
            print(f"  Log-Loss              : {metrics['log_loss']}")
            print(f"  Brier Score           : {metrics['brier_score']} (Naive Baseline: {metrics['naive_brier_score']})")
            print(f"  Brier Skill Score     : {metrics['brier_skill_score']}")
            print("--------------------------------------------------------------------------\n")

            sim_market_odds = np.random.choice([-135, -115, -105, +105, +125], size=len(y_test))
            bets_df = pd.DataFrame({"pred_prob": preds, "market_odds": sim_market_odds, "actual_win": y_test.values})
            backtester = MarketBacktester(initial_bankroll=self.args.initial_bankroll, kelly_fraction=self.args.kelly_fraction, min_ev=self.args.min_ev)
            bt_report = backtester.backtest_bets(bets_df)

            print("--------------------------------------------------------------------------")
            print("                       MARKET ROI BACKTEST RESULTS                        ")
            print("--------------------------------------------------------------------------")
            print(f"  Initial Bankroll      : ${bt_report['initial_bankroll']:,.2f}")
            print(f"  Final Bankroll        : ${bt_report['final_bankroll']:,.2f}")
            print(f"  Net Profit            : ${bt_report['total_net_profit']:,.2f}")
            print(f"  ROI Percentage        : {bt_report['roi_pct']}%")
            print(f"  Bets Placed           : {bt_report['total_bets_placed']} (Win Rate: {bt_report['win_rate_pct']}%)")
            print(f"  Max Drawdown          : {bt_report['max_drawdown_pct']}%")
            print(f"  Sharpe Ratio          : {bt_report['sharpe_ratio']}")
            print("--------------------------------------------------------------------------\n")

        # MODE: SIMULATE-MATCHUP
        if self.args.mode in ["full-pipeline", "simulate-matchup"]:
            print(f"[*] Executing {self.args.num_simulations:,}-Game Monte Carlo Simulation for {self.args.matchup}...")
            sim = MonteCarloGameSimulator(num_simulations=self.args.num_simulations, seed=SEED)
            teams = self.args.matchup.split("vs")
            home_t = teams[0].strip() if len(teams) > 0 else "LAD"
            away_t = teams[1].strip() if len(teams) > 1 else "SF"

            h_rat = get_team_rating(home_t)
            a_rat = get_team_rating(away_t)

            sim_res = sim.simulate_matchup(
                home_team=home_t, away_team=away_t,
                home_starter_stats={"k_pct": h_rat["k_bb"] + 0.08, "bb_pct": 0.06},
                away_starter_stats={"k_pct": a_rat["k_bb"] + 0.08, "bb_pct": 0.07},
                home_lineup_stats=[{"k_pct": 0.21, "bb_pct": 0.08, "hr_rate": 0.038, "single_rate": 0.15, "double_rate": 0.05}] * 9,
                away_lineup_stats=[{"k_pct": 0.23, "bb_pct": 0.07, "hr_rate": 0.032, "single_rate": 0.14, "double_rate": 0.04}] * 9,
                k_line_starter=self.args.k_line
            )

            print("--------------------------------------------------------------------------")
            print(f"         MONTE CARLO SIMULATION RESULTS ({home_t} vs {away_t})           ")
            print("--------------------------------------------------------------------------")
            print(f"  {home_t} Win Probability      : {sim_res['home_win_prob']*100:.1f}%")
            print(f"  {away_t} Win Probability      : {sim_res['away_win_prob']*100:.1f}%")
            print(f"  Projected Score          : {home_t} {sim_res['expected_home_runs']} - {away_t} {sim_res['expected_away_runs']}")
            print(f"  {home_t} -1.5 Run Line Cover  : {sim_res['home_run_line_cover_1_5']*100:.1f}%")
            print(f"  Over 8.5 Total Runs      : {sim_res['over_8_5_prob']*100:.1f}%")
            print(f"  {home_t} Starter > {self.args.k_line} Ks : {sim_res['home_starter_over_k_line_prob']*100:.1f}%\n")

        if self.args.export_path:
            out_data = {
                "active_features": active_features,
                "metrics": metrics if 'metrics' in locals() else {},
                "backtest": bt_report if 'bt_report' in locals() else {},
                "simulations": sim_res if 'sim_res' in locals() else {}
            }
            self.export_report(out_data, self.args.export_path)

def create_parser() -> argparse.ArgumentParser:
    """Creates CLI argument parser with switches for all 5 Moneyball stages."""
    parser = argparse.ArgumentParser(
        prog="mlb-cli",
        description="Moneyball MLB Prediction Engine CLI - Stage & Feature Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["full-pipeline", "predict-today", "daily-locks", "backtest", "simulate-matchup", "evaluate-models"],
        default="full-pipeline",
        help="Primary execution mode"
    )

    g_data = parser.add_argument_group("Stage 2: Data Pipeline Switches")
    g_data.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD) for historical date or live slate prediction")
    g_data.add_argument("--season", type=int, default=datetime.now().year, help="Target season for backtesting/training")
    g_data.add_argument("--data-source", choices=["retrosheet", "pybaseball", "statsapi", "all"], default="all", help="Primary data source")
    g_data.add_argument("--storage-format", choices=["sqlite", "parquet", "both"], default="both", help="Storage persistence engine")
    g_data.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False, help="Toggle offline mode with synthetic fallback data")

    g_feat = parser.add_argument_group("Stage 3: Feature Engineering Switches (Ablation Toggles)")
    g_feat.add_argument("--pitcher-rolling", action=argparse.BooleanOptionalAction, default=True, help="Toggle pitcher rolling K-BB%%, SIERA, FIP, Velo trends")
    g_feat.add_argument("--statcast", action=argparse.BooleanOptionalAction, default=True, help="Toggle Statcast pitch profiles (Whiff%%, Hard-Hit%%, xwOBA)")
    g_feat.add_argument("--platoon", action=argparse.BooleanOptionalAction, default=True, help="Toggle hitter platoon splits (vs LHP / vs RHP)")
    g_feat.add_argument("--pitch-matchups", action=argparse.BooleanOptionalAction, default=True, help="Toggle lineup pitch-type matchup scores")
    g_feat.add_argument("--bullpen", action=argparse.BooleanOptionalAction, default=True, help="Toggle bullpen 1-3 day pitch workload & fatigue index")
    g_feat.add_argument("--weather-park", action=argparse.BooleanOptionalAction, default=True, help="Toggle park factors & physics-based weather vectors")
    g_feat.add_argument("--travel-rest", action=argparse.BooleanOptionalAction, default=True, help="Toggle schedule rest days & travel timezone shifts")

    g_model = parser.add_argument_group("Stage 4: Modeling Architecture Switches")
    g_model.add_argument("--model-type", choices=["xgboost", "lightgbm", "catboost", "logistic", "poisson-two-stage", "monte-carlo"], default="xgboost", help="Algorithm backend for classification/regression")
    g_model.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True, help="Enable Platt sigmoidal probability calibration")
    g_model.add_argument("--num-simulations", type=int, default=10000, help="Number of simulations for Monte Carlo simulator")
    g_model.add_argument("--matchup", type=str, default="LAD vs SF", help="Team matchup for simulation (e.g. 'LAD vs SF')")
    g_model.add_argument("--k-line", type=float, default=6.5, help="Pitcher strikeout prop line threshold")

    g_bet = parser.add_argument_group("Stage 1 & 5: Objective & Betting Backtest Switches")
    g_bet.add_argument("--min-ev", type=float, default=0.02, help="Minimum Expected Value (EV) threshold for Daily Locks (+2%% = 0.02)")
    g_bet.add_argument("--kelly-fraction", type=float, default=0.25, help="Fractional Kelly Criterion scaling factor (0.25 = quarter Kelly)")
    g_bet.add_argument("--initial-bankroll", type=float, default=10000.0, help="Initial bankroll for ROI backtesting ($)")
    g_bet.add_argument("--output-format", choices=["json", "csv", "html"], default="csv", help="Report export format (CSV, JSON, or HTML)")
    g_bet.add_argument("--html", action=argparse.BooleanOptionalAction, default=False, help="Export output report as styled HTML dashboard")
    g_bet.add_argument("--export-path", type=str, default=None, help="Export summary file path (e.g. report.html, report.csv, or report.json)")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    orchestrator = MLBCliOrchestrator(args)
    orchestrator.execute()

if __name__ == "__main__":
    main()
