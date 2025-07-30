import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-108-65-246.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, vectorizer_path", [
    ("yt_chrome_plugin_model", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, vectorizer_path):
    client = MlflowClient()

    # Get all versions for the model (stages deprecated, so no filtering by stage)
    versions = client.search_model_versions(f"name = '{model_name}'")
    
    assert versions, f"No versions found for model '{model_name}'"

    # Pick latest version by version number (convert string to int)
    latest_version_info = max(versions, key=lambda v: int(v.version))

    try:
        # Load model by model registry URI
        model_uri = f"models:/{model_name}/{latest_version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer from local file path
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Create dummy input and transform using vectorizer
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Predict
        prediction = model.predict(input_df)

        # Validation assertions
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version_info.version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
