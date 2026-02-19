from sklearn.ensemble import RandomForestClassifier

def build_model(config):
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"]
    )
    return model
