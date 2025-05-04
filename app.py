from flask import Flask, render_template, request, jsonify
import joblib  # Using joblib instead of pickle
import os
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize stop words, lemmatizer, and stemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Initialize the Flask app
app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

# Define emotion responses
emotion_responses = {
    'affection': "It's wonderful to see you expressing affection. Positive connections are important for wellbeing. Affection brings people closer together and reminds us that we are loved and valued. Whether you're showing kindness to someone else or receiving it, these small moments matter deeply. Never underestimate how much a gentle word or gesture can brighten someone's day—including your own. ",
    'anger': "I notice you're feeling angry. Remember to take deep breaths and process these feelings in a healthy way. Anger is often a response to feeling hurt or misunderstood. Try to pause, reflect on the root cause, and consider expressing your feelings calmly. Healthy outlets like talking, writing, or exercising can help transform anger into clarity and action.",
    'approval': "I see you're expressing approval. It's great to recognize and appreciate things that align with your values. When we express approval, it encourages others and reinforces positive behaviors. It's a subtle but powerful way of shaping the kind of world we want to live in. Keep noticing what resonates with you—your approval holds meaning.",
    'confusion': "It seems you're feeling confused. Taking time to sort through your thoughts can help bring clarity. Confusion is often the mind's way of signaling that it's working through something unfamiliar or complex. Don't rush the process—ask questions, break the topic into smaller pieces, and remember it's okay not to understand everything right away. Clarity often follows curiosity.",
    'disgust': "I detect feelings of disgust. Remember that strong negative reactions often tell us something important about our boundaries. Disgust can be a natural reaction to things that feel morally or physically wrong. It's your inner compass trying to protect you. It might help to reflect on what triggered this emotion and what values or experiences are tied to it.",
    'fear': "I notice signs of fear in your writing. Remember that fear is a natural response, but try not to let it overwhelm you. Fear can keep us safe, but it can also limit us if it takes control. Gently remind yourself of the things you can manage and the strength you carry within. Sometimes just acknowledging fear gives it less power over us. You're stronger than you think.",
    'happiness': "Your happiness shines through your words! What a wonderful feeling to celebrate. Happiness, whether from a small win or a big event, adds light to our lives. Don't be afraid to lean into this joy, share it, and let it recharge you. Moments like these help carry us through tougher times—so savor it.",
    'neutral': "Your tone seems quite neutral. Sometimes a balanced perspective helps us see things clearly. Being in a neutral state can be restful—it gives you space to observe, process, or decide without emotional pressure. It's okay to feel steady, even if it seems uneventful. Sometimes neutrality is where clarity is born.",
    'pride': "I sense pride in your words. It's important to recognize and celebrate your achievements. Pride comes from effort and perseverance, and it's a sign of growth. You've likely overcome obstacles to get here, so take a moment to truly appreciate your journey. Let that feeling fuel your next steps—you're building something meaningful.",
    'sadness': "I'm noticing signs of sadness. Remember that it's okay to feel down sometimes, and these feelings will pass. Sadness often shows up when something we care about changes or feels distant. Allow yourself to feel it without rushing to fix it. Be kind to yourself—healing takes time, and you're not alone. There's always hope ahead, even if it's not visible just yet.",
    'surprise': "Surprise comes through in your message. Unexpected events can certainly stir up strong emotions. Whether it's a joyful surprise or a shocking twist, give yourself time to process. Change can be exciting or unsettling, but either way, it offers a chance to learn and adapt. Stay open—you might discover something new about yourself or the situation."
}

# List of emotions the model predicts
emotions = ['affection', 'anger', 'approval', 'confusion', 'disgust', 'fear', 
            'happiness', 'neutral', 'pride', 'sadness', 'surprise']

# Step 1: Tokenize (cell by cell)
def tokenize_words(text):
    tokens = word_tokenize(text)
    print(f"Tokens: {tokens}")  # Debugging line
    return tokens

# Step 2: Clean text (lowercase, remove punctuation, numbers, and unwanted characters)
def clean_text(tokens):
    if not tokens:  # Check if the token list is empty
        return ''
    
    cleaned_tokens = []
    for token in tokens:
        token = token.lower()  # lowercase
        token = token.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        token = re.sub(r'\d+', '', token)  # remove numbers
        # Additional cleaning step for unwanted characters
        token = token.replace("’", "'")  # fix the apostrophe
        token = token.replace("`", "")  # remove any stray backticks
        
        if token:  # Only add non-empty tokens
            cleaned_tokens.append(token)
    
    print(f"Cleaned Tokens: {cleaned_tokens}")  # Debugging print
    return cleaned_tokens  # Return list of cleaned tokens

# Step 3: Remove stopwords
def remove_stopwords(tokens):
    filtered = [word for word in tokens if word not in stop_words]
    print(f"Removed Stopwords: {filtered}")  # Debugging print
    return filtered

# Step 4: Lemmatization and Stemming
def normalize_text(tokens):
    if not isinstance(tokens, list) or not tokens:
        return "none"
    
    normalized = []
    for token in tokens:
        try:
            lemma = lemmatizer.lemmatize(token)
            stemmed = stemmer.stem(lemma)
            normalized.append(stemmed)
        except:
            continue
    
    print(f"Normalized Text: {normalized}")  # Debugging print
    return ' '.join(normalized) if normalized else 'none'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded properly'}), 500
    
    try:
        # Step 1: Tokenize the text
        tokens = tokenize_words(text)
        
        # Step 2: Clean the tokens
        cleaned_tokens = clean_text(tokens)
        
        # Step 3: Remove stopwords
        filtered_tokens = remove_stopwords(cleaned_tokens)
        
        # Step 4: Lemmatize and Stem the text
        normalized_text = normalize_text(filtered_tokens)
        
        # Vectorize the preprocessed text
        vectorized_text = vectorizer.transform([normalized_text])
        
        # Predict the emotions
        prediction = model.predict(vectorized_text)[0]
        
        # Get emotions based on prediction
        detected_emotions = [emotions[i] for i, val in enumerate(prediction) if val == 1]
        
        if not detected_emotions:
            detected_emotions = ['neutral']
        
        # Generate responses
        responses = [emotion_responses[emotion] for emotion in detected_emotions]
        
        return jsonify({
            'emotions': detected_emotions,
            'responses': responses,
            'combined_response': ' '.join(responses)
        })
    
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)

