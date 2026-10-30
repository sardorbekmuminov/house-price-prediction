import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:20s} MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

def main():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    print("Dataset shape:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=18, random_state=42, n_jobs=-1
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate(name, model, X_test, y_test)

if __name__ == "__main__":
    main()
