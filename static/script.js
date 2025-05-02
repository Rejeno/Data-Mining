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

// Scroll-triggered animation (Intersection Observer)
const animatedElements = document.querySelectorAll('.animate-on-scroll'); // Make sure elements have this class

// Set up the intersection observer options
const options = {
  root: null, // Use the viewport as the root
  rootMargin: '0px',
  threshold: 0.1 // 10% of the element must be visible
};

// Set up the intersection observer callback
const observer = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    console.log(entry.target);  // Add this line to check if the observer is firing
    if (entry.isIntersecting) {
      entry.target.classList.add('animate-fade-up'); // Add fade-up animation class when element is in view
      observer.unobserve(entry.target); // Stop observing after the animation is triggered
    }
  });
}, options);

// Observe each element
animatedElements.forEach(element => {
  observer.observe(element);
});