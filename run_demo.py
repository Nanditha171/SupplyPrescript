"""
Supply Prescript - Demo Launcher Script
Executes training check and starts the Streamlit interactive dashboard.
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    print("==================================================")
    print("   SUPPLY PRESCRIPT — CLOSED-LOOP DEMO LAUNCHER   ")
    print("   Author: Prasanna (ML & Optimization Lead)      ")
    print("==================================================")

    # 1. Verify Model Artifacts
    models_dir = os.path.join(ROOT_DIR, "ml", "models")
    clf_path = os.path.join(models_dir, "delay_classifier.joblib")
    reg_path = os.path.join(models_dir, "delay_regressor.joblib")

    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        print(">>> Model artifacts not found. Training initial XGBoost baseline...")
        from ml.train_model import train_models
        train_models(save_artifacts=True)
    else:
        print(">>> Verified ML model artifacts.")

    # 2. Start Streamlit Demo
    app_path = os.path.join(ROOT_DIR, "frontend", "app.py")
    print(f">>> Launching Streamlit Demo: {app_path}")
    print(">>> Open your browser to http://localhost:8501")
    print("==================================================")

    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.headless=true",
            "--server.port=8501",
        ], env=env)
    except KeyboardInterrupt:
        print("\n>>> Demo stopped by user.")


if __name__ == "__main__":
    main()
