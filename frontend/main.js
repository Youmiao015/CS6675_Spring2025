/* main.js — UI logic (ES‑Module) */
import {
  fetchSearchResults,
  fetchAggregatePlot
  // ✅ No longer importing fetchPredictionDemoImage
} from './api.js';

document.addEventListener('DOMContentLoaded', () => {
  /* Grab DOM elements */
  const searchInput  = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');
  const resultPanel  = document.getElementById('result-panel');

  if (!searchInput || !searchButton || !resultPanel) {
    console.error('Required elements are missing.');
    return;
  }

  /* Search click */
  searchButton.addEventListener('click', () => {
    const topic = searchInput.value.trim();
    if (!topic) { alert('Please enter a topic.'); return; }

    resultPanel.innerHTML = '<p>Loading…</p>';

    Promise.all([
      fetchSearchResults(topic),
      fetchAggregatePlot(topic)
      // ✅ Removed the third API call
    ])
      .then(([papers, aggUrl]) => {
        renderResultPanel(papers, aggUrl);   // Only passing two arguments
      })
      .catch(err => {
        console.error(err);
        resultPanel.innerHTML = '<p>Error loading information.</p>';
      });
  });

  /* ---------- helpers ---------- */
  function renderResultPanel(results, aggUrl) {  // <- Two parameters
    resultPanel.innerHTML = '';

    /* 1. aggregate plot */
    if (aggUrl) {
      const img = document.createElement('img');
      img.src = aggUrl;
      img.alt = 'Aggregate Plot';
      img.classList.add('responsive-img');
      img.addEventListener('click', () => img.classList.toggle('enlarged'));
      resultPanel.appendChild(img);
    }

    /* 2. toggle button + hidden text box */
    const btn  = document.createElement('button');
    btn.textContent = 'Show Text Details';
    const box = document.createElement('div');
    box.style.display = 'none';
    resultPanel.appendChild(btn);
    resultPanel.appendChild(box);

    btn.addEventListener('click', () => {
      const hide = box.style.display === 'none';
      box.style.display   = hide ? 'block' : 'none';
      btn.textContent     = hide ? 'Hide Text Details' : 'Show Text Details';
    });

    /* 3. populate text box */
    if (results.length) {
      results
        .sort((a, b) =>
          (b.similarity ?? b.distance ?? 0) - (a.similarity ?? a.distance ?? 0))
        .forEach((r, i) => box.appendChild(textItem(r, i)));
    } else {
      box.innerHTML = '<p>No results found.</p>';
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