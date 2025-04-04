/*
 * main.js
 * Main file for page interaction.
 * Listens for user input, calls the API functions, and updates the DOM.
 */

document.addEventListener('DOMContentLoaded', function() {
  // Get DOM elements
  const searchInput = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');
  const searchResultsContainer = document.getElementById('search-results');
  const predictionResultContainer = document.getElementById('prediction-result');

  // Event listener for the search button
  searchButton.addEventListener('click', function() {
    const topic = searchInput.value.trim();
    if (!topic) {
      alert('Please enter a topic.');
      return;
    }

    // Clear previous results
    searchResultsContainer.innerHTML = '';
    predictionResultContainer.innerHTML = '';

    // Show loading hints
    searchResultsContainer.innerHTML = '<p>Loading search results...</p>';
    predictionResultContainer.innerHTML = '<p>Loading prediction...</p>';

    // Call API interfaces in parallel
    Promise.all([fetchSearchResults(topic), fetchModelPrediction(topic)])
      .then(([searchResults, modelPrediction]) => {
        searchResultsContainer.innerHTML = '';

        if (searchResults.length > 0) {
          // Sort by distance ascending (lower means more similar)
          searchResults.sort((a, b) => b.distance - a.distance);

          searchResults.forEach((result, index) => {
            const resultDiv = createResultItem(result, index);
            searchResultsContainer.appendChild(resultDiv);
          });
        } else {
          searchResultsContainer.innerHTML = '<p>No results found.</p>';
        }

        // Update the prediction result display area
        predictionResultContainer.innerHTML = `<p>Prediction for "${modelPrediction.topic}": ${modelPrediction.prediction}</p>`;
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        searchResultsContainer.innerHTML = '<p>Error loading search results.</p>';
        predictionResultContainer.innerHTML = '<p>Error loading prediction.</p>';
      });
  });

  /**
   * Create a DOM element for a search result.
   * Display format:
   *   Paper {index}: {title}
   *   Similarity: {distance}
   *   Abstract: (initially collapsed, expandable with a toggle button)
   * @param {Object} result - Article object returned by the API, containing title, abstract, distance, etc.
   * @param {number} index - Result index.
   * @returns {HTMLElement} - The created DOM element.
   */
  function createResultItem(result, index) {
    const container = document.createElement('div');
    container.classList.add('result-item');

    // Title
    const titleElem = document.createElement('h3');
    titleElem.textContent = `Paper ${index + 1}: ${result.title}`;
    container.appendChild(titleElem);

    // Similarity (distance)
    const similarityElem = document.createElement('p');
    similarityElem.textContent = `Similarity: ${result.distance}`;
    container.appendChild(similarityElem);

    // Abstract
    const abstractElem = document.createElement('p');
    // Add collapsed class to initially show only one line (controlled by CSS)
    abstractElem.classList.add('abstract', 'collapsed');
    // Prevent undefined
    abstractElem.textContent = result.abstract || '';
    container.appendChild(abstractElem);

    // Toggle button for expanding/collapsing
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = 'Show More';
    toggleBtn.addEventListener('click', function() {
      if (abstractElem.classList.contains('collapsed')) {
        abstractElem.classList.remove('collapsed');
        toggleBtn.textContent = 'Show Less';
      } else {
        abstractElem.classList.add('collapsed');
        toggleBtn.textContent = 'Show More';
      }
    });
    container.appendChild(toggleBtn);

    return container;
  }
});

/**
 * Demo function for testing.
 * Auto-fills the search input with a demo topic and triggers a search.
 */
function demoSearch() {
  document.getElementById('search-input').value = 'Demo Topic';
  document.getElementById('search-button').click();
}

window.demoSearch = demoSearch;