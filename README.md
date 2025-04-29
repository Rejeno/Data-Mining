# Emotion Detection Diary App

A simple web application that detects emotions from text input using a pre-trained SVM model.

## Overview

This application uses a Flask backend with an HTML/CSS frontend to analyze the emotions expressed in the user's text. It employs a Support Vector Machine (SVM) model for multi-label classification to detect various emotions like happiness, sadness, anger, etc.

## Features

- Text input for users to express their thoughts
- Emotion analysis using a pre-trained SVM model
- Display of detected emotions with appropriate responses
- Simple and intuitive user interface

## Prerequisites

- Python 3.6 or higher
- Flask
- NLTK
- scikit-learn
- pickle (for loading the model)

## Setup Instructions

1. Make sure you have Python installed on your system.

2. Install the required packages:
   ```
   pip install flask nltk scikit-learn
   ```

3. Ensure your model files are in the root directory:
   - `svm_model.pkl` - Your trained SVM model
   - `vectorizer.pkl` - Your trained vectorizer (TfidfVectorizer or CountVectorizer)

4. Run the application:
   ```
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Project Structure

```
emotion-detection-app/
│
├── app.py                 # Flask application
├── svm_model.pkl          # Pre-trained SVM model
├── vectorizer.pkl         # Pre-trained vectorizer
├── static/
│   └── styles.css         # CSS styling
└── templates/
    └── index.html         # HTML template
```

## How It Works

1. The user enters text in the input area and clicks "Analyze Emotions"
2. The text is sent to the Flask backend
3. The text is preprocessed (tokenization, cleaning, stopword removal, normalization)
4. The preprocessed text is vectorized using the pre-trained vectorizer
5. The SVM model predicts the emotions present in the text
6. The detected emotions and an appropriate response are returned to the frontend
7. The results are displayed to the user
