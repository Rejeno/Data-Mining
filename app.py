from flask import Flask, render_template, request, jsonify
import joblib  # Using joblib instead of pickle
import os
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import random

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
    'affection': [
        "Affection is a beautiful expression of human connection. It reminds us of the bonds we share with others and the warmth that comes from meaningful relationships. Whether you're showing love or receiving it, these moments create a sense of belonging and emotional safety. Affection can uplift spirits, strengthen bonds, and create memories that last a lifetime. It’s a gentle reminder that we’re not alone in this world, and that kindness and care can make a profound difference in someone’s life. \n\n Tips: \n 1. Share a heartfelt compliment with someone you care about. \n 2. Write a thoughtful note or letter to express your feelings. \n 3. Spend quality time with loved ones, free from distractions, to deepen your connection.",
        "When we express affection, we nourish not just others but ourselves. A kind gesture, a thoughtful message, or even a smile can be a healing force in someone’s life. Affection fosters emotional intimacy and builds trust, creating a foundation for relationships that can weather life’s challenges. It’s in these gentle exchanges that we find the strength to face the world together. \n\n Tips: \n 1. Give a warm hug or a reassuring pat on the back to someone who needs it. \n 2. Cook a meal or share a meaningful activity with someone special. \n 3. Listen actively and attentively to someone’s story, showing that you value their presence.",
        "Affection is a language of its own, transcending words and speaking directly to the heart. It’s a way of saying, 'I see you, I value you, and I care about you.' When you show affection, you’re not just sharing a moment; you’re creating a memory that can bring comfort and joy for years to come. These acts of kindness and care are the threads that weave the fabric of our relationships. \n\n Tips: \n 1. Create a scrapbook or photo album of shared memories to celebrate your bond. \n 2. Plan a surprise for someone you love to show them how much they mean to you. \n 3. Share your favorite song, book, or movie with someone as a way to connect on a deeper level.",
        "Affection is like sunlight—it brightens relationships and fosters growth. It’s a powerful way to show others that they matter and that they’re not alone. When people share kind words or acts of care, it brings a sense of safety and belonging that is vital to emotional wellbeing. Affection is a gift that keeps on giving, creating ripples of positivity that extend far beyond the initial act. \n\n Tips: \n 1. Send a thoughtful text or message to check in on someone you care about. \n 2. Share a favorite book, movie, or experience with a friend to create a shared connection. \n 3. Volunteer together with someone you care about to strengthen your bond while giving back to the community.",
        "In a world that can sometimes feel cold and disconnected, affection is the warmth that reminds us of our shared humanity. It’s a gentle nudge to be present, to connect, and to cherish the moments we have with one another. Acts of affection, no matter how small, can create a ripple effect of kindness and compassion that touches countless lives. \n\n Tips: \n 1. Write a heartfelt letter to someone you appreciate, expressing your gratitude and love. \n 2. Share a laugh or a joke with someone to brighten their day. \n 3. Create a gratitude jar together with a loved one, filling it with notes of appreciation and happy memories."
    ],

    'anger': [
        "Anger is a natural and powerful emotion that signals when something feels wrong or unjust. It’s your body and mind reacting to a perceived boundary being crossed or a deep pain being triggered. Instead of suppressing it, consider listening to it. What is your anger trying to tell you? What needs to be addressed or healed? By understanding the root of your anger, you can channel it into constructive actions that bring resolution and growth. \n\n Tips: \n 1. Take a moment to breathe deeply and ground yourself before reacting. \n 2. Write down what’s making you angry to gain clarity and perspective. \n 3. Talk to someone you trust about your feelings to process them in a healthy way.",
        "Anger can be a powerful motivator for change. It’s a call to action, urging you to address what feels wrong and to advocate for yourself or others. When channeled constructively, anger can fuel positive transformation and inspire courage to stand up for what matters. Use it as an opportunity to reflect on your values and take steps toward creating the change you wish to see. \n\n Tips: \n 1. Channel your anger into a creative project, such as writing, painting, or music. \n 2. Engage in physical activity, like exercise or sports, to release pent-up energy. \n 3. Participate in activism or community service to turn your anger into meaningful action.",
        "It's natural to feel angry, but how you process it determines whether it becomes destructive or transformative. Anger can be a signal that something important to you is at stake. By acknowledging and addressing it, you can use it as a tool for self-awareness and growth. Journaling, speaking openly, or engaging in physical activity can help you process your anger in a healthy way. \n\n Tips: \n 1. Practice mindfulness or meditation to observe your anger without judgment. \n 2. Go for a walk or run to clear your mind and release tension. \n 3. Find a healthy outlet for your emotions, such as talking to a therapist or trusted friend.",
        "Anger can be a double-edged sword. It can protect you from harm, but it can also lead to impulsive decisions or hurtful words. When you feel anger rising, take a moment to pause and reflect. What is the underlying issue? How can you express your feelings in a way that promotes understanding rather than conflict? By taking a step back, you can transform anger into a catalyst for positive change. \n\n Tips: \n 1. Use 'I' statements to express your feelings without blaming others. \n 2. Practice deep breathing or grounding techniques to calm your mind and body. \n 3. Seek professional help if anger feels overwhelming or unmanageable.",
        "Anger can be a powerful teacher. It often reveals what you care about most deeply and what needs to change in your life or the world around you. Instead of letting it fester, use it as an opportunity to reflect on your values and take action. By channeling your anger into constructive outlets, you can create positive change for yourself and others. \n\n Tips: \n 1. Reflect on the root cause of your anger and what it reveals about your values. \n 2. Engage in activities that promote self-care and emotional healing. \n 3. Consider seeking professional support if anger becomes overwhelming."
    ],

    'approval': [
        "Approval reflects a shared alignment of values, ethics, or aesthetics. When you express it, you're offering affirmation—and that can be a powerful motivator for positive action in others.\n\n Tips: \n 1. Acknowledge someone’s effort with a simple 'thank you.' \n 2. Share your appreciation publicly, like on social media. \n 3. Write a note of encouragement to someone.",

        "Approval is a form of connection. It’s a way to say, 'I see you, and I appreciate what you bring to the table.' This can strengthen relationships and build trust over time.\n\n Tips: \n 1. Celebrate small wins with those around you. \n 2. Offer constructive feedback when appropriate. \n 3. Create a culture of appreciation in your environment.",

        "By acknowledging and approving something, you're reinforcing what feels right to you. It's not just encouragement to others—it's an internal signal that you're standing by your values.\n\n Tips: \n 1. Reflect on what you value and why. \n 2. Share your values with those close to you. \n 3. Create a vision board of what you approve of.",

        "Approval can be validating. Whether you're receiving it or giving it, it fosters mutual respect and reinforces positive behavior patterns. It’s a cornerstone of healthy social interaction.\n\n Tips: \n 1. Be specific in your praise to make it meaningful. \n 2. Encourage others to express their approval too. \n 3. Reflect on how approval affects your relationships.",

        "We often underestimate the strength of our approval. A nod of agreement or a heartfelt compliment can shift someone’s day—or even their trajectory. Keep giving your light where it’s deserved.\n\n Tips: \n 1. Share stories of how someone's effort inspired you. \n 2. Practice random acts of praise. \n 3. Invite others to share moments of appreciation in a group setting.",

        "Expressing approval is also an act of gratitude. You're recognizing something good and saying, 'I see you, and I value this.' Those affirmations matter deeply in building community and trust.\n\n Tips: \n 1. Keep a 'praise log' to track what you appreciated daily. \n 2. Set a goal to uplift someone every day. \n 3. Reflect on how giving approval improves your own well-being."
    ],

    'confusion': [
        "Confusion isn't failure—it's a sign that you're actively engaging with new ideas. Learning isn't always linear; allow yourself the grace to feel lost while you're growing.\n\n Tips: \n 1. Take a break and revisit the topic later. \n 2. Discuss your confusion with someone knowledgeable. \n 3. Write down your thoughts to clarify them.",

        "If you're confused, it means you're stretching beyond your comfort zone. These moments are fertile ground for deep learning and self-discovery. Break the problem down and take it one step at a time.\n\n Tips: \n 1. Break down the information into smaller parts. \n 2. Use visual aids like diagrams or charts. \n 3. Ask questions to clarify your understanding.",

        "Everyone feels confused sometimes, especially when dealing with complexity. Try writing out what you know, what you don’t, and what you *think*—you'll often find clarity in the contrast.\n\n Tips: \n 1. Create a mind map to visualize your thoughts. \n 2. Use analogies to relate new concepts to familiar ones. \n 3. Practice mindfulness to reduce anxiety around confusion.",

        "Confusion is the brain’s way of asking for clarification, not giving up. It invites you to pause and seek more context. Trust that the fog will lift as you continue exploring.\n\n Tips: \n 1. Take a step back and breathe deeply. \n 2. Seek out additional resources or perspectives. \n 3. Embrace the uncertainty as part of the learning process.",

        "Take a deep breath and know that confusion is temporary. Sometimes, the best breakthroughs arise just after the most frustrating mental roadblocks. Keep asking questions—that’s how answers come.\n\n Tips: \n 1. Practice self-compassion during confusing times. \n 2. Engage in activities that help you relax and refocus. \n 3. Reflect on past confusions that led to growth.",

        "Confusion can be a sign of growth. It means you're challenging your understanding and pushing boundaries. Embrace it as part of the learning process.\n\n Tips: \n 1. Recall a time when confusion helped you grow. \n 2. Journal your current questions without needing immediate answers. \n 3. Remind yourself that confusion is a signal—not a stop sign."
    ],

    'disgust': [
        "Disgust is an intense and visceral reaction—often tied to our sense of morality or personal boundaries. Reflecting on what triggers this emotion can reveal deeply held values or unresolved memories.\n\n Tips: \n 1. Journal about what disgusts you and why. \n 2. Discuss your feelings with someone you trust. \n 3. Explore the cultural context of your disgust.",

        "It’s okay to feel disgust—it’s a protective emotion. It helps us navigate environments, people, and situations that conflict with our sense of safety or integrity.\n\n Tips: \n 1. Identify the source of your disgust. \n 2. Set boundaries to protect your emotional space. \n 3. Seek professional help if needed.",

        "Your sense of disgust might be flagging something that feels ethically or physically wrong to you. Treat it as a signal to investigate, not necessarily to immediately act on.\n\n Tips: \n 1. Reflect on the values that inform your disgust. \n 2. Talk with someone who has a different perspective. \n 3. Ask whether the disgust points to a value you want to uphold or reevaluate.",

        "Disgust can help refine our social filters. Whether it’s injustice, behavior, or even imagery—this emotion challenges us to reevaluate our environments and actions.\n\n Tips: \n 1. Read or listen to diverse viewpoints. \n 2. Consider if your reaction is emotional, cultural, or instinctive. \n 3. Use disgust as a prompt for ethical reflection.",

        "Sometimes disgust is tied to learned experiences or cultural norms. Reflecting on its source can help you decide if it aligns with your values—or if it's time to reframe your perspective.\n\n Tips: \n 1. Trace the origin of the feeling without judgment. \n 2. Research cultural differences in what is considered 'disgusting.' \n 3. Ask yourself whether the feeling is helping or hindering your well-being."
    ],

    'fear': [
        "Fear is a natural and protective emotion—it alerts you to risk and prepares your body to respond. But left unchecked, it can also hold you back from growth and discovery. Ask yourself: Is this fear keeping me safe, or is it keeping me small? Recognizing that distinction is the first step toward reclaiming your power.\n\nTips:\n1. Write down your fears to bring clarity and perspective.\n2. Speak with someone you trust about what you're experiencing.\n3. Begin with small, manageable steps to gradually face what scares you.",
        
        "Everyone experiences fear—it's a sign you're alive and aware. Rather than pushing it down, give it a name. Write about it, speak it aloud. You’ll often find that its grip loosens once it’s acknowledged. Understanding your fear doesn’t make you weaker; it makes you wiser.\n\nTips:\n1. Practice mindfulness to observe fear without judgment.\n2. Keep a fear journal—record what triggers it and how you respond.\n3. Remind yourself of times you’ve handled fear well in the past.",
        
        "Fear is often your mind’s way of trying to protect you from uncertainty. But it can be transformed into motivation if you don’t let it freeze you. True courage is moving forward despite being afraid. Make a plan, prepare, and take action one step at a time.\n\nTips:\n1. Break down large fears into small, manageable pieces.\n2. Create a realistic plan to face or reduce each fear.\n3. Reach out for support—no one has to face fear alone.",
        
        "Sometimes fear shows up as self-doubt, hesitation, or even procrastination. Instead of ignoring it, confront it directly: ask yourself, 'What am I really afraid of?' and 'What’s the worst that could happen—and could I handle it?' This gives you agency and preparation, which fear often steals away.\n\nTips:\n1. Reflect on previous fears you’ve successfully overcome.\n2. Use visualization to picture yourself confronting and handling the fear.\n3. Celebrate even small wins as you work through your fear.",
        
        "Fear often reveals what matters most—it highlights what you care deeply about. Instead of resisting it, try to understand its message. What is it pointing you toward? Sometimes the things we fear most are the very things we are meant to grow into.\n\nTips:\n1. Journal about what your fear may be trying to protect or reveal.\n2. Practice mental rehearsals of facing your fear successfully.\n3. Reward yourself for progress, even if it feels small—it’s still growth."
    ],
    
    'happiness': [
        "Happiness, in any form, is worth acknowledging. It doesn’t have to come from grand events—it can bloom in quiet moments, kind gestures, or personal wins. When you allow yourself to fully experience happiness, you’re honoring the good in your life. Absorb it, savor it, and let it remind you that joy is always worth noticing.\n\nTips:\n1. Keep a gratitude journal to record things that bring you happiness.\n2. Share your joyful moments with others—it multiplies the feeling.\n3. Reflect on how happiness shows up in your daily life.",
        
        "Let your happiness shine without apology. When you embrace joy, you not only nourish your own well-being but become a beacon for others. Celebrate your wins, no matter the size, and let that positivity radiate outward. It’s okay to be joyful—you deserve it.\n\nTips:\n1. Share a recent happy moment with someone you trust.\n2. Create a vision board of what brings you joy and fulfillment.\n3. Make time for activities that light you up inside.",
        
        "Sometimes, we question our happiness—as if it’s something we must earn. But joy doesn’t need justification. You are worthy of happiness simply by being human. Let yourself smile, laugh, and feel content. These are not signs of naivety—they're signs of vitality.\n\nTips:\n1. Reflect on what brings you authentic joy.\n2. Make a habit of doing at least one joyful thing each day.\n3. Express gratitude aloud for the happiness you experience.",
        
        "Happiness lives in the quiet details—a sunrise, a meaningful conversation, a moment of peace. These seemingly small things often bring the greatest sense of joy when we take the time to notice them. Be present for these moments. They are the heartbeat of a fulfilling life.\n\nTips:\n1. Practice mindfulness to help you stay present in joyful moments.\n2. Create a list of everyday things that make you smile.\n3. Pause during your day to notice and appreciate small delights.",
        
        "Your happiness is a reflection of your resilience. After everything you've been through, joy is a sign that you're healing, growing, and moving forward. Don’t feel guilty for feeling good. Let happiness in—it’s not a reward, it’s a right.\n\nTips:\n1. Keep a gratitude journal to recognize how far you've come.\n2. Share your happiness with others—it creates connection.\n3. Look for ways to bring happiness into the lives of those around you."
    ],

    'neutral': [
        "Neutrality is often misunderstood as passivity, but in truth, it can be a grounding and valuable emotional state. When you're neutral, you're not disengaged—you’re observing, processing, and allowing space between stimulus and response. It’s a moment of balance where insight can emerge without the intensity of emotional extremes.\n\nTips:\n1. Reflect on your current emotional state without judging it as good or bad.\n2. Take this time to check in with yourself and recharge through calm self-care practices.\n3. Try mindfulness or breathing exercises to deepen your sense of presence in the moment.",
        
        "A neutral state is like still water—quiet, yet full of potential beneath the surface. It’s a natural pause in emotional intensity that invites clarity, rest, and subtle growth. Don’t rush out of it. This calm space is where you can gather your energy, refocus, and regain emotional stability.\n\nTips:\n1. Notice the calmness and allow yourself to rest in it without pressure to feel something else.\n2. Use this moment to ground yourself through reflection, journaling, or quiet walks.\n3. Engage in mindful routines that bring you peace and connection to the present.",
        
        "Emotional neutrality gives you a chance to respond with intention rather than react impulsively. It’s a form of emotional intelligence—an indicator of self-regulation and mindfulness. In this space, you can think clearly, observe your surroundings objectively, and prepare thoughtfully for whatever comes next.\n\nTips:\n1. Pause and appreciate your ability to stay centered.\n2. Reflect on what direction you want to move in from here.\n3. Practice breathing or grounding exercises to support your sense of balance.",
        
        "Neutrality isn’t apathy—it’s a meaningful pause between emotion and action. This in-between space is often where your clearest decisions are made. Without the push or pull of strong emotion, you can weigh your thoughts with greater clarity and choose a path that aligns with your true intentions.\n\nTips:\n1. Let yourself sit with stillness rather than trying to change your emotional state.\n2. Reflect on choices you’re considering and the values that guide them.\n3. Use calming, routine activities to stay grounded and centered.",
        
        "Emotional neutrality can be a form of self-preservation in moments of overwhelm. When the world feels chaotic, being in a neutral emotional space helps you create a buffer—a quiet mental room where you can breathe, think, and reset. Trust that neutrality is not a lack of emotion, but a place of calm strength.\n\nTips:\n1. Practice mindfulness or grounding to support this sense of emotional calm.\n2. Acknowledge your neutrality and allow it to be what it is without forcing change.\n3. Use this time to rest, reflect, or simply be present without judgment."
    ],
    
    'pride': [
        "Pride in your work, progress, or growth is something to cherish. It’s a clear sign of your effort, dedication, and perseverance paying off. Instead of brushing past it, allow yourself to fully recognize how far you’ve come. Pride can be a wellspring of motivation, confidence, and self-respect.\n\nTips:\n1. Reflect on your accomplishments—big or small—and what they mean to your journey.\n2. Share your sense of pride with someone who supports and encourages you.\n3. Use this moment to set new intentions or goals fueled by your current momentum.",
        
        "When you feel proud, it means you’re acknowledging your own value and contributions—and that’s powerful. It shows that you’re recognizing your own worth, and there’s no need to downplay that. Let your pride lift you, affirm your efforts, and remind you of what you’re capable of.\n\nTips:\n1. Take a moment to internalize what you’ve achieved without minimizing it.\n2. Celebrate your efforts and invite others to celebrate with you.\n3. Consider how this moment can shape your next step forward.",
        
        "Your pride reflects a journey—one filled with perseverance, growth, and overcoming obstacles. It’s not just about the end result, but everything you pushed through to get here. Let this moment be one of self-recognition and celebration. You’ve earned it.\n\nTips:\n1. Pause to reflect on how you’ve grown and what it took to get here.\n2. Share your story—it may inspire someone else.\n3. Set new goals that align with your continued growth and purpose.",
        
        "Pride is more than personal satisfaction—it’s a celebration of your ability to rise, achieve, and stay true to your path. Sharing it with others invites connection and inspiration. When you take pride in your accomplishments, you empower yourself and those around you.\n\nTips:\n1. Reflect on the challenges you’ve overcome and how they shaped your strength.\n2. Let others witness your pride—it may encourage them too.\n3. Use this pride as a springboard for future growth or leadership.",
        
        "Pride, when grounded in truth, becomes a steady source of inner confidence. You know what you’ve been through to reach this place, and that story matters. Let yourself feel fulfilled. You don’t need to prove it to anyone—just own your story, your effort, and your success.\n\nTips:\n1. Think about what values and actions brought you to this moment.\n2. Share your pride authentically—it reflects self-respect, not arrogance.\n3. Identify the next milestone you’d like to pursue, with pride as your compass."
    ],

    'sadness': [
        "Sadness is a deeply human emotion that deserves to be met with compassion rather than avoidance. It’s often a sign that something or someone important has touched your heart. This emotion connects us to love, memory, and the meaning we find in our lives. Rather than pushing it away, honoring your sadness can create space for healing, reflection, and ultimately, renewal. Even when it's heavy, it can carry important messages about what matters most.\n\nTips:\n1. Give yourself permission to feel your emotions fully and without guilt.\n2. Reach out to someone you trust—connection can ease the weight of sorrow.\n3. Prioritize self-care, whether it’s rest, nourishment, or gentle movement.",
        
        "When sadness visits, it’s a call to slow down and listen inward. These are the moments that require tenderness and patience with yourself. There’s no rush to 'fix' the feeling—just being present with it can be healing. Let sadness flow through you like a river, knowing it doesn’t define you—it simply reflects your depth and humanity.\n\nTips:\n1. Sit quietly and acknowledge what you're feeling, without judgment.\n2. Share your thoughts with someone who can offer a caring presence.\n3. Do something nurturing for yourself, like taking a walk, listening to calming music, or resting.",
        
        "Sadness isn’t a sign of weakness—it’s a reflection of your capacity to care, to love, and to be affected by the world around you. It speaks to your emotional depth and your ability to connect with life on a meaningful level. Let it come and go as it needs to, like waves washing onto a shore. You don’t need to rush it away—healing takes time, and that’s okay.\n\nTips:\n1. Let yourself experience the sadness fully, without trying to control it.\n2. Talk to a trusted friend or write about what you're going through.\n3. Treat yourself gently—rest, nourish yourself, and breathe.",
        
        "Tears are not a sign of failure—they’re the body’s way of releasing what the heart holds too tightly. Sometimes words aren't enough, and that's when emotion takes over. Allow yourself the space to grieve what’s been lost, whether it’s a person, an opportunity, or a part of yourself. In time, healing will find its way back to you, and the weight will lighten.\n\nTips:\n1. Write in a journal to explore and release your thoughts.\n2. Do comforting activities like drinking warm tea, reading, or being in nature.\n3. If the sadness feels overwhelming, consider talking to a mental health professional.",
        
        "Sadness can bring you to a place of stillness, asking you to pause and reconnect with yourself. Just like clouds pass across the sky, emotions shift with time. In the quiet that sadness brings, you might discover insight, compassion, or a renewed appreciation for brighter days. Be gentle with yourself—this is part of the journey.\n\nTips:\n1. Accept the sadness as a natural part of emotional life.\n2. Open up to someone who can offer a kind ear or comforting words.\n3. Practice soothing routines—light a candle, take a bath, or rest your mind with a favorite book or show."
    ],
   
    'surprise': [
        "Surprise has a powerful way of jolting us out of the ordinary. It interrupts the familiar rhythm of daily life and reminds us that the world is filled with wonder, unpredictability, and new possibilities. Whether the surprise is joyful or difficult, it creates an opening—a moment that invites us to pause, reassess, and grow in unexpected ways. Often, it’s through these moments of disruption that we come to understand ourselves better, uncover hidden strengths, or see things from a new perspective.\n\nTips:\n1. Try not to resist the unexpected—lean into it and see where it might take you.\n2. Look back at a past surprise and notice how it changed your direction or thinking.\n3. Share your surprise with a friend or loved one—it can turn a solitary moment into a shared experience.",
        
        "Surprises come in many forms, and their impact can range from delightful excitement to temporary discomfort. What they have in common is the ability to challenge our expectations and open our eyes to new realities. They can push us to adapt, reevaluate, and sometimes rediscover the joy or uncertainty we’ve overlooked. Even when a surprise catches us off guard, it’s a chance to reconnect with the present moment and see the world through a refreshed lens.\n\nTips:\n1. Pause, take a deep breath, and allow yourself to process the emotions tied to the surprise.\n2. Journal your initial thoughts, feelings, and any shifts in perspective it caused.\n3. Consider what this surprise might be teaching you—and how it could become a catalyst for personal growth.",
        
        "Surprise has a unique ability to uncover our inner flexibility and resilience. When the unexpected happens, it forces us to change course, rethink our assumptions, or even celebrate something new. In those moments, you’re often more capable than you think—your ability to navigate the unknown becomes clearer. Each surprise is a test of your ability to adjust, learn, and grow. When you look back, you might be surprised at how often these moments shaped who you are today.\n\nTips:\n1. Think of a time when an unexpected event challenged you—how did you adapt and what did you learn?\n2. Rather than resisting, try to explore what opportunities this surprise might present.\n3. Talk about your experience with someone—it helps build understanding and connection.",
        
        "Life’s surprises, whether joyful or jarring, offer a moment of awakening. They break our assumptions and remind us that no plan is ever set in stone. In embracing these twists, we make room for flexibility, insight, and often, personal growth. Surprises can deepen our appreciation for the present, encourage creativity, or give us the push we didn’t know we needed to move forward with courage and clarity.\n\nTips:\n1. Embrace the surprise rather than fearing it—there may be a hidden gift.\n2. Reflect on how previous surprises altered your path in meaningful ways.\n3. Discuss the experience with someone to gain fresh perspectives or emotional support.",
        
        "The unexpected carries energy that can awaken curiosity, stir emotions, and prompt transformation. While some surprises may feel disruptive, others can breathe new life into our routines or open up opportunities we never considered. When you meet surprise with openness rather than resistance, it becomes a powerful force for growth and connection.\n\nTips:\n1. Take a moment to acknowledge and sit with your initial reaction.\n2. Write down your thoughts to uncover how the surprise is affecting you.\n3. Consider how this moment might shape your journey in unexpected but meaningful ways."
    ]

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
        responses = [random.choice(emotion_responses[emotion]).replace('\n', '<br>') for emotion in detected_emotions]

        
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

