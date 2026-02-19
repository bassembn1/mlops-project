import requests

def test_predict():
    url = "http://localhost:8000/predict"
    data = {
        "age": 30,
        "salary": 6000,
        "experience": 5
    }

    response = requests.post(url, params=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
