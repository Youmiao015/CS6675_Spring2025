/* main.js  ─ UI logic (ES‑Module) */
import {
  fetchSearchResults,
  fetchAggregatePlot,
  fetchPredictionDemoImage
} from './api.js';

document.addEventListener('DOMContentLoaded', () => {
  /* Grab DOM elements */
  const searchInput  = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');
  const searchPanel  = document.getElementById('search-panel');
  const predictionPanel = document.getElementById('prediction-panel');

  if (!searchInput || !searchButton || !searchPanel || !predictionPanel) {
    console.error('Required elements are missing.');
    return;
  }

  /* Search click */
  searchButton.addEventListener('click', () => {
    const topic = searchInput.value.trim();
    if (!topic) { alert('Please enter a topic.'); return; }

    searchPanel.innerHTML     = '<p>Loading search results…</p>';
    predictionPanel.innerHTML = '<p>Loading prediction…</p>';

    Promise.all([
      fetchSearchResults(topic),
      fetchAggregatePlot(topic),
      fetchPredictionDemoImage(topic)
    ])
      .then(([papers, aggUrl, predUrl]) => {
        renderSearchPanel(papers, aggUrl);
        renderPredictionPanel(predUrl);
      })
      .catch(err => {
        console.error(err);
        searchPanel.innerHTML     = '<p>Error loading search results.</p>';
        predictionPanel.innerHTML = '<p>Error loading prediction.</p>';
      });
  });

  /* ---------- helpers ---------- */
  function renderSearchPanel(results, aggUrl) {
    searchPanel.innerHTML = '';

    /* aggregate plot */
    if (aggUrl) {
      const img = document.createElement('img');
      img.src = aggUrl;
      img.alt = 'Aggregate Plot';
      img.classList.add('responsive-img');
      img.addEventListener('click', () => img.classList.toggle('enlarged'));
      searchPanel.appendChild(img);
    }

    /* toggle button + hidden text box */
    const btn  = document.createElement('button');
    btn.textContent = 'Show Text Details';
    const box = document.createElement('div');
    box.style.display = 'none';
    searchPanel.appendChild(btn);
    searchPanel.appendChild(box);

    btn.addEventListener('click', () => {
      const hide = box.style.display === 'none';
      box.style.display = hide ? 'block' : 'none';
      btn.textContent  = hide ? 'Hide Text Details' : 'Show Text Details';
    });

    /* populate text box */
    if (results.length) {
      results
        .sort((a, b) =>
          (b.similarity ?? b.distance ?? 0) - (a.similarity ?? a.distance ?? 0))
        .forEach((r, i) => box.appendChild(textItem(r, i)));
    } else {
      box.innerHTML = '<p>No results found.</p>';
    }
  }

  function renderPredictionPanel(predUrl) {
    predictionPanel.innerHTML = '';
    if (predUrl) {
      const img = document.createElement('img');
      img.src = predUrl;
      img.alt = 'Prediction Results';
      img.classList.add('responsive-img');
      img.addEventListener('click', () => img.classList.toggle('enlarged'));
      predictionPanel.appendChild(img);
    } else {
      predictionPanel.innerHTML = '<p>Error loading prediction image.</p>';
    }
  }

  function textItem(result, idx) {
    const div = document.createElement('div');
    div.classList.add('text-result-item');

    const title = document.createElement('h3');
    title.textContent = `Paper ${idx + 1}: ${result.title}`;
    div.appendChild(title);

    const abstract = document.createElement('p');
    abstract.textContent = result.abstract || '';
    abstract.classList.add('abstract');
    div.appendChild(abstract);

    const sim = document.createElement('p');
    sim.textContent = `Similarity: ${result.similarity ?? result.distance ?? 'N/A'}`;
    div.appendChild(sim);

    return div;
  }
});