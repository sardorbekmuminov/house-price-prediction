import numpy as np
import matplotlib.pyplot as plt

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

    # 1. Load dataset
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    print("Dataset shape:", X.shape)
    print("Missing values:", int(X.isna().sum().sum()))

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define models
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            random_state=42,
            n_jobs=-1
        )
    }

    # 4. Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate(name, model, X_test, y_test)

    # 5. Feature Importance
    rf_model = models["Random Forest"]

    importance = rf_model.feature_importances_

    feature_importance = dict(zip(X.columns, importance))

    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1]
        )
    )

    print("\nFeature Importance:")

    for feature, score in feature_importance.items():
        print(f"{feature:15s}: {score:.4f}")

    # 6. Visualize Feature Importance
    plt.figure(figsize=(10, 6))

    plt.barh(
        list(feature_importance.keys()),
        list(feature_importance.values())
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Random Forest - Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importance.png")

    plt.show()



    # 7. Actual vs Predicted

    y_pred = rf_model.predict(X_test)

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, alpha=0.3)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("Actual House Values")
    plt.ylabel("Predicted House Values")
    plt.title("Actual vs Predicted - Random Forest")

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    plt.show()

if __name__ == "__main__":
    main()