/*
 * api.js
 * Encapsulates API calls to the backend.
 */

/**
 * Fetch search results for a given topic from the backend.
 * @param {string} topic - The search topic.
 * @returns {Promise<Array>} - A promise that resolves to an array of paper objects.
 */
function fetchSearchResults(topic) {
  return fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query: topic,
      top_k: 5  // Request 5 results
    })
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      // Assume the returned data structure is { results: [ ... ], aggregated_metrics: { ... } }
      return data.results || [];
    })
    .catch(error => {
      console.error('Error fetching search results:', error);
      return [];
    });
}

/**
 * Simulate fetching a model prediction for a given topic.
 * @param {string} topic - The search topic.
 * @returns {Promise<Object>} - A promise that resolves to a prediction object.
 */
function fetchModelPrediction(topic) {
  const mockPrediction = {
    topic: topic,
    prediction: Math.floor(Math.random() * 100)
  };

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockPrediction);
    }, 800);
  });
}

// Optional: export functions if using modules
// export { fetchSearchResults, fetchModelPrediction };