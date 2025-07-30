import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://ec2-3-108-65-246.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name", [
    "yt_chrome_plugin_model",
])
def test_load_latest_model_version(model_name):
    client = MlflowClient()

    # Search all versions of the model by name (no stage filtering)
    filter_str = f"name = '{model_name}'"
    versions = client.search_model_versions(filter_str)

    assert versions, f"No versions found for model '{model_name}'"

    # Use 'version' attribute (string), convert to int for max comparison
    latest_version_info = max(versions, key=lambda v: int(v.version))

    try:
        model_uri = f"models:/{model_name}/{latest_version_info.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version_info.version} loaded successfully.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
