from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, config):
    X = df.drop("purchased", axis=1)
    y = df["purchased"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    return X_train, X_test, y_train, y_test, scaler
