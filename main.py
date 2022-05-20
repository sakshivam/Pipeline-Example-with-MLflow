import mlflow


def workflow():
    with mlflow.start_run():
        print("Cleaning Step")
        mlflow.run(".", "cleaning", parameters={})

        print("Feature Engineering Step")
        mlflow.run(".", "feature_engg", parameters={})

        print("Feature Selection Step")
        mlflow.run(".", "feature_selection", parameters={})

        print("Modelling Step")
        mlflow.run(".", "modelling", parameters={})

        print("Scoring Step")
        mlflow.run(".", "scoring", parameters={})


if __name__ == '__main__':
    workflow()
