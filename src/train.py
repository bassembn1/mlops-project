""" import joblib
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.config_loader import load_config

def train():
    config = load_config()

    df = load_data(config["data"]["path"])

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, config)

    model = build_model(config)
    model.fit(X_train, y_train)

    joblib.dump(model, config["output"]["model_path"])
    joblib.dump(scaler, "models/scaler.joblib")

    print("Training completed and model saved.")

if __name__ == "__main__":
    train()
 """
 
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.config_loader import load_config


def train():
    config = load_config()
    #mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("heart_disease_experiment")
    # بدء تسجيل تجربة
    with mlflow.start_run():

        df = load_data(config["data"]["path"])

        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, config)

        model = build_model(config)
        model.fit(X_train, y_train)

        # التنبؤ
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # تسجيل الإعدادات
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])

        # تسجيل النتيجة
        mlflow.log_metric("accuracy", acc)

        # تسجيل الموديل
        #mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(model, name="model")
        # حفظ نسخة محلية
        joblib.dump(model, config["output"]["model_path"])
        joblib.dump(scaler, "models/scaler.joblib")

        print(f"Training completed. Accuracy: {acc}")


if __name__ == "__main__":
    train()
