## AI Acknowledgment


This project was developed with assistance from Claude (Anthropic). AI was used for generating some parts of the code, debugging, and architectural decisions.


# Road Accident Severity Prediction

ML pipeline for predicting road accident severity (Slight/Serious/Fatal) using UK road safety data.

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

**Where we got it:** [Kaggle - UK Road Safety Dataset](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-videos)


## Quick Start

### 1. Install Dependencies
```bash
pip install -r api/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
This trains a Logistic Regression model and saves artifacts to `models2/`.

### 3. Run the API
```bash
cd api && uvicorn main:app --host 0.0.0.0 --port 8080
```

### 4. Run the Frontend
```bash
cd frontend && streamlit run app.py
```

## Reusing This Pipeline

To adapt for your own dataset:

1. **Prepare your data**: Place CSV in `content/` with a target column for severity classification

2. **Modify `train.py`**:
   - Update `selected_features` list (line 58-62) with your feature columns
   - Update `target_mapping` (line 81) with your target classes
   - Adjust data filtering logic if needed

3. **Update `config/config.yaml`**: Change feature list and project settings

4. **Update API schemas**: Modify `AccidentInput` in `api/main.py` to match your features

5. **Update frontend**: Adjust input fields in `frontend/app.py`

## Project Structure
```
├── train.py              # Training pipeline with MLflow tracking
├── api/main.py           # FastAPI prediction endpoint
├── frontend/app.py       # Streamlit UI
├── config/config.yaml    # Configuration
├── models2/              # Saved model artifacts
└── content/road.csv      # Training data
```

## Tech
- **ML**: scikit-learn, pandas, MLflow
- **API**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Deployment**: Docker, Google Cloud Run

---

## Links

**GitHub:** https://github.com/jaydip12357/road-accident-prediction  
**API:** https://road-accident-api-1028014290034.us-east1.run.app  
**Frontend:** https://road-accident-predictor.streamlit.app/  
**Pull Request:** https://github.com/jaydip12357/road-accident-prediction/pull/1  
**Dataset:** https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles

---


Made with ☕ at Duke University

