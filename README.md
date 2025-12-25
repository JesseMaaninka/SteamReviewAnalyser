# SteamReviewAnalyser

## Instructions to run the App

### Running the application locally

After cloning or downloading the project folder

1. Create or activate virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Start the app with Streamlit

```
streamlit run app.py
```

4. Open in browser

    http://localhost:8501



### Running the app using Docker

1. Build the Docker image

```
docker build -t steam-review-analyser .
```

2. Run the Docker container

```
docker run -p 8501:8501 steam-review-analyser
```

3. Open the app in a browser

    http://localhost:8501




## The App

This application analyzes user reviews from the Steam platform to study how textual characteristics relate to review sentiment. By fetching reviews for a selected game, the app explores differences between positive and negative reviews using descriptive statistics, word frequency analysis, and a simple machine-learning classifier.