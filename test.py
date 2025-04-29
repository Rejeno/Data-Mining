"""
Test script to verify your model and vectorizer are working correctly.
Run this script to verify your model is functioning as expected.
"""

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
try:
    print("Attempting to load model and vectorizer with joblib...")
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Success! Model and vectorizer loaded.")
    print(f"Model type: {type(model)}")
    print(f"Vectorizer type: {type(vectorizer)}")
except Exception as e:
    print(f"Error loading with joblib: {e}")
    print("Trying to load with pickle as a fallback...")
    
    import pickle
    try:
        model = pickle.load(open('svm_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        print("Success! Model and vectorizer loaded with pickle.")
    except Exception as e:
        print(f"Error loading with pickle: {e}")
        print("Failed to load the model and vectorizer with both methods.")
        exit(1)

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# List of emotions
emotions = ['affection', 'anger', 'approval', 'confusion', 'disgust', 'fear', 
            'happiness', 'neutral', 'pride', 'sadness', 'surprise']

# Test examples
test_examples = [
    "I'm feeling so happy and joyful today!",
    "I'm really angry about what happened yesterday.",
    "I don't understand anything. I'm feeling so confused, lost, and uncertain. Nothing is clear.",
    "I'm feeling sad and depressed.",
    "That's disgusting!",
    "I'm so proud of what I accomplished.",
    "I love you so much!",
    "I'm really scared right now.",
    "Wow! That was such a surprise!"
]

print("\n===== TESTING MODEL PREDICTIONS =====")
for text in test_examples:
    print(f"\nInput text: \"{text}\"")
    processed = preprocess_text(text)
    print(f"Processed: \"{processed}\"")
    
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    
    print(f"Raw prediction: {prediction}")
    
    # Handle different possible return formats
    if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
        # If it's a 2D array, take the first element
        prediction = prediction[0]
    
    # Get predicted emotions
    detected = [emotions[i] for i, val in enumerate(prediction) if val == 1]
    
    if not detected:
        detected = ['neutral']
    
    print(f"Detected emotions: {detected}")

print("\nTest completed. If your model is working correctly, you should see appropriate")
print("emotions detected for each test example above.")