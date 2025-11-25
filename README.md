# Road Accident Severity Prediction

A complete machine learning project that predicts how severe road accidents will be based on conditions like weather, speed limit, and time of day. We built everything from training the model to deploying it on cloud so anyone can use it through a website.

## Live Links

**Try it yourself:** [Frontend App](https://road-accident-predictor.streamlit.app/)  
**API:** [API Endpoint](https://road-accident-api-1028014290034.us-east1.run.app)  
**API Docs:** [Interactive Documentation](https://road-accident-api-1028014290034.us-east1.run.app/docs)

---

## What This Project Does

This project takes information about road conditions and predicts if an accident will be:
- **Slight** (minor injuries)
- **Serious** (major injuries)  
- **Fatal** (deaths)

You can enter things like speed limit, weather, time of day and the model tells you how severe an accident might be.

---

## Team

**Students:** Jaideep Aher & Roshan Gill  
**Course:** AIPI 510, Fall 2025  
**University:** Duke University

---

## Dataset

We used UK Road Safety data from 2005 to 2023. The dataset has information about 1.8 million accidents but we used 200,000 for training.

**Where we got it:** [Kaggle - UK Road Safety Dataset](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles)

**What features we used:**
- Speed limit
- Number of vehicles involved
- Number of casualties
- Hour of day
- Light conditions (daylight or dark)
- Weather conditions
- Road surface (dry, wet, icy)
- Urban or rural area

**What we predict:**
The severity of the accident in 3 categories (Slight, Serious, Fatal)

---

## How The Model Works

We used **Logistic Regression** which is a simple machine learning algorithm for classification. 

**Model settings:**
- Used balanced class weights to handle imbalanced data
- 200 iterations for training
- StandardScaler for feature scaling
- Label encoding for categorical features

**Performance:**
- Accuracy: 52%
- F1 Score: 0.33

The accuracy is not super high mainly because most accidents in the data are fatal which creates an imbalance. But the focus of this project was learning deployment and MLOps practices rather than getting perfect accuracy.

---

## Tech Stack

**Cloud Services:**
- Google Cloud Storage (for storing data)
- Google Cloud Run (for hosting API)
- Streamlit Cloud (for hosting website)

**Tools & Frameworks:**
- Python 3.10
- FastAPI (for building API)
- Streamlit (for frontend)
- Docker (for containerization)
- MLFlow (for tracking experiments)
- scikit-learn (for ML model)

---

## Project Structure
```
road-accident-prediction/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── model_loader.py    # Loads trained model
│   └── requirements.txt   # API dependencies
├── frontend/              # Streamlit web app
│   ├── app.py            # Frontend code
│   └── requirements.txt  # Frontend dependencies
├── models2/              # Trained model files
│   ├── accident_severity_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
├── config/               # Configuration files
│   └── config.yaml
├── Dockerfile           # Container setup
└── README.md           # This file
```

---

## How To Run This Locally

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jaydip12357/road-accident-prediction.git
cd road-accident-prediction
```

2. Install requirements:
```bash
pip install -r api/requirements.txt
```

3. Run the API:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Open in browser:
```
http://localhost:8000/docs
```

### Using Docker

1. Build the image:
```bash
docker build -t road-accident-api .
```

2. Run the container:
```bash
docker run -p 8080:8080 road-accident-api
```

3. Test it:
```bash
curl http://localhost:8080/health
```

---

## API Usage

### Check if API is working
```bash
curl https://road-accident-api-1028014290034.us-east1.run.app/health
```

### Make a prediction
```bash
curl -X POST https://road-accident-api-1028014290034.us-east1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speed_limit": 30,
    "number_of_vehicles": 2,
    "number_of_casualties": 1,
    "hour": 17,
    "light_conditions": "Daylight",
    "weather_conditions": "Fine no high winds",
    "road_surface_conditions": "Dry",
    "urban_or_rural_area": "Urban"
  }'
```

### Response you get
```json
{
  "prediction": "Fatal",
  "probabilities": {
    "Slight": 0.21,
    "Serious": 0.34,
    "Fatal": 0.44
  },
  "confidence": 0.44
}
```

---

## Using Python
```python
import requests

url = "https://road-accident-api-1028014290034.us-east1.run.app/predict"

data = {
    "speed_limit": 70,
    "number_of_vehicles": 3,
    "number_of_casualties": 2,
    "hour": 23,
    "light_conditions": "Darkness - lights unlit",
    "weather_conditions": "Raining + high winds",
    "road_surface_conditions": "Wet or damp",
    "urban_or_rural_area": "Rural"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Deployment

### API Deployment (Google Cloud Run)

We deployed our API to Google Cloud Run which automatically scales based on traffic.

**How we did it:**
1. Connected GitHub repo to Cloud Build
2. Cloud Build creates Docker image automatically
3. Image gets deployed to Cloud Run
4. We get a public URL that anyone can access

### Frontend Deployment (Streamlit Cloud)

The website is hosted on Streamlit Cloud for free.

**Steps:**
1. Signed in to Streamlit Cloud with GitHub
2. Selected our repository
3. Pointed to frontend/app.py
4. Deployed automatically

---

## What We Learned

This project taught us:
- How to store data in cloud instead of locally
- Building REST APIs with FastAPI
- Packaging apps in Docker containers
- Deploying to cloud platforms
- Using MLFlow to track experiments
- Building user interfaces with Streamlit
- Working with GitHub branches and pull requests
- Making ML projects reproducible

---

## Problems We Faced

**Class Imbalance:** Most accidents were fatal in our sample which made the model biased. We used balanced class weights but it didnt help much.

**Version Issues:** Had problems loading model files because of numpy version differences. Fixed by retraining in same environment.

**Docker Configuration:** Took time to get all paths correct in Dockerfile. Initially models folder name was wrong.

**API Imports:** Python import errors when running API. Fixed by adjusting module paths.

---

## Future Improvements

If we continue this project we would:
- Try other models like Random Forest or XGBoost
- Use more features from the dataset
- Better handle the class imbalance problem
- Add user login and tracking
- Set up monitoring for API
- Add more charts and visualizations
- Test different models with A/B testing

---

## Important Note

This model is for educational purposes only. Do not use it for making real decisions about road safety or emergency response. The accuracy is not high enough for critical applications.

---

## Links

**GitHub:** https://github.com/jaydip12357/road-accident-prediction  
**API:** https://road-accident-api-1028014290034.us-east1.run.app  
**Frontend:** https://road-accident-predictor.streamlit.app/  
**Pull Request:** https://github.com/jaydip12357/road-accident-prediction/pull/1  
**Dataset:** https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles

---

## Contact

For questions or issues, open an issue on GitHub or contact:
- Jaideep Aher: aherjaideep1@gmail.com
- GitHub: [@jaydip12357](https://github.com/jaydip12357)

---

Made with ☕ at Duke University



AI was used in some parts for code ideas and while understanding on use of docker.
