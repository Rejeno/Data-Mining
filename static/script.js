document.getElementById('analyze-btn').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') {
        alert('Please enter some text to analyze.');
        return;
    }
    
    // Show loading spinner
    document.getElementById('loading').style.display = 'flex';
    
    // Call API to analyze emotion
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: userInput })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading spinner
        document.getElementById('loading').style.display = 'none';
        
        // Display results
        displayResults(data.emotions, data.combined_response);
    })
    .catch(error => {
        // Hide loading spinner
        document.getElementById('loading').style.display = 'none';
        
        console.error('Error:', error);
        alert('Error analyzing emotions. Please try again.');
    });
});

function displayResults(emotions, response) {
    const emotionsContainer = document.getElementById('emotions-list');
    const responseContainer = document.getElementById('emotion-response');
    
    // Clear previous results
    emotionsContainer.innerHTML = '';
    
    // Display emotions
    emotions.forEach(emotion => {
        const emotionElement = document.createElement('span');
        emotionElement.classList.add('emotion-tag');
        emotionElement.textContent = emotion;
        emotionsContainer.appendChild(emotionElement);
    });
    
    // Display response
    responseContainer.innerHTML = `<p>${response}</p>`;
}
