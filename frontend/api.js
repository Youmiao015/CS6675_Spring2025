/* ------------------------------------------------------------------
 * api.js
 * 负责调用 FastAPI 后端，返回 Promise 结果。
 * 不包含任何 DOM 操作；界面更新逻辑写在 main.js 中。
 * ------------------------------------------------------------------ */

/**
 * POST /search
 * @param {string} topic - 查询主题
 * @param {number} topK  - 返回前 K 条
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
 * 返回一张聚合柱状图 (PNG) 的 blob URL
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
 * 返回预测柱状图 (PNG) 的 blob URL
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