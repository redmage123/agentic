import requests

TCA_URL = "http://tca-service:5001"  # Update with TCA's service URL

def get_prediction(data):
    """
    Sends prediction data to the TCA and returns the response.
    """
    try:
        response = requests.post(f"{TCA_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
