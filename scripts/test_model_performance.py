import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-108-65-246.ap-south-1.compute.amazonaws.com:5000/")


@pytest.mark.parametrize("model_name, holdout_data_path, vectorizer_path", [
    ("yt_chrome_plugin_model", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),  # Replace with your actual paths
])
def test_model_performance(model_name, holdout_data_path, vectorizer_path):
    client = MlflowClient()

    # Search all versions (no stage filtering since stages are deprecated)
    versions = client.search_model_versions(f"name = '{model_name}'")

    assert versions, f"No versions found for model '{model_name}'"

    # Pick latest version by version number
    latest_version_info = max(versions, key=lambda v: int(v.version))

    try:
        model_uri = f"models:/{model_name}/{latest_version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer locally
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Load holdout dataset
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()
        y_holdout = holdout_data.iloc[:, -1]

        # Handle NaN in text input
        X_holdout_raw = X_holdout_raw.fillna("")

        # Transform text data using vectorizer
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(X_holdout_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

        # Predict
        y_pred_new = model.predict(X_holdout_tfidf_df)

        # Metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        # Expected thresholds (adjust as appropriate)
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        assert accuracy_new >= expected_accuracy, f"Accuracy should be at least {expected_accuracy}, got {accuracy_new}"
        assert precision_new >= expected_precision, f"Precision should be at least {expected_precision}, got {precision_new}"
        assert recall_new >= expected_recall, f"Recall should be at least {expected_recall}, got {recall_new}"
        assert f1_new >= expected_f1, f"F1 score should be at least {expected_f1}, got {f1_new}"

        print(f"Performance test passed for model '{model_name}' version {latest_version_info.version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")
