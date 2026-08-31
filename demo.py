"""
Moneyball MLB Prediction Engine - Full End-to-End Demonstration Script

Demonstrates:
1. Objective definition (Odds conversion, EV calculation, Kelly Criterion)
2. Data Pipeline ingestion (Retrosheet / PyBaseball / MLB Stats API)
3. Feature Engineering (Pitchers, Hitters, Bullpens, Park & Weather vectors)
4. Modeling Architecture (XGBoost, Poisson 2-Stage Run Exp, Monte Carlo 10,000 Sim)
5. Model Evaluation & Market ROI Backtesting
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mlb_engine.pipeline_runner import MLBPredictionPipeline

def main():
    print("\n" + "="*80)
    print("      MONEYBALL MLB PREDICTION SYSTEM - END-TO-END DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize and execute pipeline
    pipeline = MLBPredictionPipeline(offline_mode=True)
    pipeline.run_full_pipeline()

    print("\n[SUCCESS] All 5 core stages executed successfully!")
    print("The system is fully modularized and ready for production data ingestion and backtesting.\n")

if __name__ == "__main__":
    main()
