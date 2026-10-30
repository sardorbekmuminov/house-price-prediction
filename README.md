# House Price Prediction

A beginner-friendly regression project using the California Housing dataset from scikit-learn.

## Goal
Train and compare a baseline, Linear Regression, and Random Forest model to predict median house values.

## Dataset
The California Housing dataset is downloaded automatically by scikit-learn on first run. Internet access may be required.
Dataset documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

## Project structure
- `notebooks/house_price_prediction.ipynb` — step-by-step analysis
- `src/train.py` — runnable training script
- `requirements.txt` — dependencies

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

To open the notebook:
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

## Evaluation
The project reports MAE, RMSE, and R² on a held-out test set. Run it before adding any numerical results here; do not invent performance claims.

## Learning outcomes
- Inspecting a real-world dataset
- Train/test splitting
- Training regression models
- Evaluating and comparing model performance

Educational project only; predictions are not official property valuations.
