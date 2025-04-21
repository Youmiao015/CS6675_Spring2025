/**
 * POST /search
 * @param {string} topic 
 * @param {number} topK  
 * @returns {Promise<Array>} papers array
 */
export function fetchSearchResults(topic, topK = 5) {
  return fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: topic, top_k: topK })
  })
    .then(res => res.json())
    .then(data => data.results || [])
    .catch(err => {
      console.error('Error fetching search results:', err);
      return [];
    });
}

/**
 * POST /search/aggregate_plot
 * @param {string} topic
 * @returns {Promise<string|null>} object‑URL or null
 */
export function fetchAggregatePlot(topic) {
  return fetch('http://localhost:8000/search/aggregate_plot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: topic, top_k: 10000 })
  })
    .then(res => res.blob())
    .then(blob => URL.createObjectURL(blob))
    .catch(err => {
      console.error('Error fetching aggregate plot:', err);
      return null;
    });
}

/**
 * POST /search/prediction_demo
 * @param {string} topic
 * @returns {Promise<string|null>} object‑URL or null
 */
export function fetchPredictionDemoImage(topic) {
  return fetch('http://localhost:8000/search/prediction_demo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: topic, top_k: 10000 })
  })
    .then(res => res.blob())
    .then(blob => URL.createObjectURL(blob))
    .catch(err => {
      console.error('Error fetching prediction demo image:', err);
      return null;
    });
}