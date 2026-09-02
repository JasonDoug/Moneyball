# Moneyball: Effective MLB Prediction & Betting System

A production-grade, modular Major League Baseball (MLB) machine learning engine structured around five core stages:

1. **Objective Definition & Betting Mathematics**
2. **Data Pipeline & Storage Architecture**
3. **Feature Engineering & Feature Ablation Toggles**
4. **Multi-Model Machine Learning Architecture**
5. **Evaluation, Calibration & Market Backtesting Framework**

---

## 🏗 System Architecture & Five Core Stages

```
                          ┌───────────────────────────┐
                          │   Stage 1: Objectives     │
                          │ - Moneyline & Run Line    │
                          │ - Totals (Over / Under)   │
                          │ - Pitcher Ks & Batter Props│
                          │ - EV & Kelly Criterion    │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   Stage 2: Data Pipeline  │
                          │ - PyBaseball & Statcast   │
                          │ - MLB Stats API (Live)    │
                          │ - Retrosheet Game Logs    │
                          │ - SQLite & DuckDB Parquet │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │ Stage 3: Feature Engine   │
                          │ - Pitcher K-BB%, SIERA    │
                          │ - Statcast Whiff/xwOBA    │
                          │ - Platoon Splits & Pitch  │
                          │ - Bullpen Workload & Park │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │ Stage 4: Multi-Model Arch │
                          │ - XGBoost / LightGBM      │
                          │ - Poisson 2-Stage Run Exp │
                          │ - 10,000 Sim Monte Carlo  │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │ Stage 5: Evaluation & ROI │
                          │ - Time-Series Cross-Val   │
                          │ - Brier Score & ECE       │
                          │ - Market Odds ROI & Kelly │
                          └───────────────────────────┘
```

---

## 🚀 Quick Start & Installation

### 1. Environment Initialization
```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Launch Interactive Web Dashboard UI
```bash
./bin/mlb-web 8000
```
Open **`http://localhost:8000`** in your browser to access the full interactive dark-mode dashboard (featuring live slate predictions, Monte Carlo simulator, and market ROI backtester).

### 3. Predict Today's Live Real-World MLB Slate
```bash
./bin/mlb-cli --mode predict-today --no-offline
```

### 4. Export Styled HTML Report
```bash
./bin/mlb-cli --mode predict-today --html --export-path report.html
```

### 5. Run Unit Test Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

---

## 💻 Command Line Interface (`mlb-cli`) & All `--switches`

The CLI tool `./bin/mlb-cli` provides granular `--switches` to control every stage of data ingestion, feature selection, model training, simulation parameters, and betting objectives.

### Complete List of Command Line Switches

```text
usage: mlb-cli [-h]
               [--mode {full-pipeline,predict-today,daily-locks,backtest,simulate-matchup,evaluate-models}]
               [--season SEASON]
               [--data-source {retrosheet,pybaseball,statsapi,all}]
               [--storage-format {sqlite,parquet,both}]
               [--offline | --no-offline]
               [--pitcher-rolling | --no-pitcher-rolling]
               [--statcast | --no-statcast] [--platoon | --no-platoon]
               [--pitch-matchups | --no-pitch-matchups]
               [--bullpen | --no-bullpen] [--weather-park | --no-weather-park]
               [--travel-rest | --no-travel-rest]
               [--model-type {xgboost,lightgbm,catboost,logistic,poisson-two-stage,monte-carlo}]
               [--calibrate | --no-calibrate]
               [--num-simulations NUM_SIMULATIONS] [--matchup MATCHUP]
               [--k-line K_LINE] [--min-ev MIN_EV]
               [--kelly-fraction KELLY_FRACTION]
               [--initial-bankroll INITIAL_BANKROLL]
               [--export-path EXPORT_PATH]
```

### Switch Descriptions by Category

#### 🎯 1. Execution Modes (`--mode`)
- `--mode predict-today`: Ingests **today's real live MLB slate** from MLB Stats API, runs predictions, and prints win probabilities, projected run scores, and Daily Locks.
- `--mode daily-locks`: Scans slate for high positive Expected Value (+EV) bets exceeding the `--min-ev` threshold.
- `--mode simulate-matchup`: Runs a 10,000+ game Monte Carlo simulation for a specific matchup (e.g. `--matchup "LAD vs SF"`).
- `--mode backtest`: Executes chronological market closing line ROI backtesting with Kelly Criterion stake sizing.
- `--mode evaluate-models`: Calculates probabilistic error metrics (Log-Loss, Brier Score, ECE calibration).
- `--mode full-pipeline`: Runs all 5 stages end-to-end.

#### 🌐 2. Data Pipeline & Mode Switches (Stage 2)
- `--offline / --no-offline`:
  - `--no-offline`: Enables **Live Mode**. Fetches real live data over the internet from MLB Stats API (`statsapi`), Baseball Savant (`pybaseball`), and FanGraphs.
  - `--offline`: Enables **Offline Mode**. Uses fast synthetic fallback generators for offline testing and CI/CD benchmarks.
- `--season INT`: Target season year for backtesting/training (default: `2024`).
- `--data-source {all, retrosheet, pybaseball, statsapi}`: Data provider filter (default: `all`).
- `--storage-format {sqlite, parquet, both}`: Persistence layer selection (default: `both`).

#### ⚡ 3. Feature Engineering & Feature Ablation Switches (Stage 3)
Allows toggling specific feature domains on/off for model ablation studies:
- `--pitcher-rolling / --no-pitcher-rolling`: Toggle starting pitcher rolling metrics ($K-BB\%$, SIERA, FIP/xFIP, fastball velocity trends).
- `--statcast / --no-statcast`: Toggle Statcast pitch-level profiles (Whiff%, Hard-Hit% against $\ge 95\text{ mph}$, $xwOBA$ against).
- `--platoon / --no-platoon`: Toggle hitter platoon splits (rolling wOBA and wRC+ vs LHP / vs RHP).
- `--pitch-matchups / --no-pitch-matchups`: Toggle lineup pitch-type matchup scores against starter's primary pitch mix (FF, SL, CH, CU, SI).
- `--bullpen / --no-bullpen`: Toggle bullpen 1, 2, and 3-day pitch workloads & fatigue penalties.
- `--weather-park / --no-weather-park`: Toggle park factors (Coors Field down to Oracle Park) and physics-based weather vectors (temperature carry, wind direction/magnitude).
- `--travel-rest / --no-travel-rest`: Toggle schedule rest days and timezone travel fatigue penalties.

#### 🤖 4. Modeling Architecture Switches (Stage 4)
- `--model-type {xgboost, lightgbm, catboost, logistic, poisson-two-stage, monte-carlo}`: Selects classification/regression algorithm engine (default: `xgboost`).
- `--calibrate / --no-calibrate`: Toggles Platt sigmoidal probability calibration via `CalibratedClassifierCV` (default: `True`).
- `--num-simulations INT`: Iteration count for event-based Monte Carlo game simulator (default: `10000`).
- `--matchup STR`: Matchup team string for simulation mode (default: `"LAD vs SF"`).
- `--k-line FLOAT`: Pitcher strikeout prop line threshold (default: `6.5`).

#### 💰 5. Objectives & Betting Evaluation Switches (Stage 1 & 5)
- `--min-ev FLOAT`: Minimum Expected Value ($EV$) threshold for Daily Locks (default: `0.02` = +2% EV).
- `--kelly-fraction FLOAT`: Fractional Kelly Criterion scaling factor (default: `0.25` = quarter Kelly).
- `--initial-bankroll FLOAT`: Initial bankroll for backtest simulation (default: `$10,000`).
- `--export-path PATH`: JSON export target file path for reports.

---

## 💡 CLI Usage Examples

### Example 1: Predict Today's Live Games
```bash
./bin/mlb-cli --mode predict-today --no-offline
```

### Example 2: Feature Ablation Study with LightGBM
Evaluate LightGBM model performance without bullpen and weather signals:
```bash
./bin/mlb-cli --model-type lightgbm --no-bullpen --no-weather-park
```

### Example 3: Deep Monte Carlo Simulation (15,000 Games)
Run an event-based Markov Chain simulation for a specific matchup and custom K-line:
```bash
./bin/mlb-cli --mode simulate-matchup --matchup "NYY vs BOS" --num-simulations 15000 --k-line 7.5
```

### Example 4: CatBoost Market Backtest & JSON Export
Backtest CatBoost predictions against market closing lines using half-Kelly stake sizing (`0.50`) and export to JSON:
```bash
./bin/mlb-cli --model-type catboost --kelly-fraction 0.50 --export-path live_report.json
```

---

## 📊 Core Modules Architecture

- **[`mlb_engine/objectives/betting.py`](file:///home/jason/sports/mlb_engine/objectives/betting.py)**: De-vigging, expected value ($EV$), and Kelly Criterion calculations.
- **[`mlb_engine/pipeline/`](file:///home/jason/sports/mlb_engine/pipeline/)**: SQLite relational storage, DuckDB Parquet columnar layer, `pybaseball`, `statsapi`, and Retrosheet ingestion.
- **[`mlb_engine/features/`](file:///home/jason/sports/mlb_engine/features/)**: Modular feature engineers covering starting pitchers, hitter lineups, bullpens, park factors, and physics-based weather vectors.
- **[`mlb_engine/modeling/`](file:///home/jason/sports/mlb_engine/modeling/)**: Direct Classifiers (XGBoost, LightGBM, CatBoost, Logistic), Two-Stage Poisson Run Expectancy, and 10,000-Game Monte Carlo Simulator.
- **[`mlb_engine/evaluation/`](file:///home/jason/sports/mlb_engine/evaluation/)**: Time-series cross-validation, Brier Score, ECE reliability diagrams, and closing line ROI backtester.

---

## 🧪 Running Unit Tests

To run the automated test suite across all 5 core stages:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```
