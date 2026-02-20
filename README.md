# Wine Quality Predictor (Assignment 5)

ML project using the UCI Wine Quality dataset to predict wine quality from 11 physicochemical features.

## Components
- Model training script: `train_model.py`
- Flask app: `app/app.py`
- Docker container: `app/Dockerfile`
- Templates: `app/templates/`

## Live App (Heroku)
https://wine-quality-checkamani-2026-d976b81ec1b8.herokuapp.com/

## Run locally (Docker)
cd app
docker build -t wine-quality-app .
docker run -p 8080:8080 wine-quality-app
