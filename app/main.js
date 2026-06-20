const state = {
  lastResponse: null,
};

const elements = {
  apiUrl: document.querySelector("#apiUrl"),
  maxResults: document.querySelector("#maxResults"),
  debugMode: document.querySelector("#debugMode"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  recommendations: document.querySelector("#recommendations"),
  recommendationCount: document.querySelector("#recommendationCount"),
  traceOutput: document.querySelector("#traceOutput"),
  traceStatus: document.querySelector("#traceStatus"),
  apiDot: document.querySelector("#apiDot"),
  apiStatus: document.querySelector("#apiStatus"),
  checkHealth: document.querySelector("#checkHealth"),
  clearChat: document.querySelector("#clearChat"),
};

if (!["localhost", "127.0.0.1"].includes(window.location.hostname)) {
  elements.apiUrl.value = window.location.origin;
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message) return;
  await sendMessage(message);
});

elements.checkHealth.addEventListener("click", checkHealth);
elements.clearChat.addEventListener("click", clearChat);

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt || "";
    elements.input.focus();
  });
});

elements.debugMode.addEventListener("change", () => {
  renderTrace(state.lastResponse);
});

async function sendMessage(message) {
  appendMessage("user", message);
  elements.input.value = "";
  setLoading(true);

  const pending = appendMessage("agent", "Thinking...");
  try {
    const response = await fetch(`${apiBaseUrl()}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        max_results: Number(elements.maxResults.value || 5),
        debug: elements.debugMode.checked,
      }),
    });

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const payload = await response.json();
    state.lastResponse = payload;
    updateAgentMessage(pending, payload.answer || "No answer returned.");
    renderRecommendations(payload.recommendations || []);
    renderTrace(payload);
    showToast(payload.status === "ok" ? "Response received." : "Agent returned a failure status.");
  } catch (error) {
    updateAgentMessage(pending, "Could not reach the API. Check that the backend is running.");
    showToast(error.message, true);
  } finally {
    setLoading(false);
  }
}

async function checkHealth() {
  elements.apiStatus.textContent = "Checking...";
  elements.apiDot.className = "status-dot";
  try {
    const response = await fetch(`${apiBaseUrl()}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    elements.apiStatus.textContent = payload.status === "ok" ? "API healthy" : "Unexpected health";
    elements.apiDot.className = payload.status === "ok" ? "status-dot ok" : "status-dot fail";
  } catch (error) {
    elements.apiStatus.textContent = "API unavailable";
    elements.apiDot.className = "status-dot fail";
    showToast(error.message, true);
  }
}

function renderRecommendations(items) {
  elements.recommendationCount.textContent = String(items.length);
  if (!items.length) {
    elements.recommendations.className = "empty-state";
    elements.recommendations.textContent = "No recommendations returned.";
    return;
  }

  elements.recommendations.className = "recommendation-list";
  elements.recommendations.innerHTML = items.map(renderTrackCard).join("");
}

function renderTrackCard(track) {
  const tags = [...(track.mood || []), ...(track.genres || []), ...(track.tags || [])]
    .slice(0, 6)
    .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
    .join("");
  const links = [
    track.preview_url ? `<a href="${escapeAttribute(track.preview_url)}" target="_blank" rel="noreferrer">Preview</a>` : "",
    track.spotify_url ? `<a href="${escapeAttribute(track.spotify_url)}" target="_blank" rel="noreferrer">Spotify</a>` : "",
  ]
    .filter(Boolean)
    .join("");
  const initials = (track.title || "MM").slice(0, 2).toUpperCase();

  return `
    <article class="track-card">
      <div class="cover">${escapeHtml(initials)}</div>
      <div>
        <h3>${escapeHtml(track.title || "Untitled")}</h3>
        <p>${escapeHtml(track.artist || "Unknown artist")}</p>
        ${track.reason ? `<p>${escapeHtml(track.reason)}</p>` : ""}
        ${tags ? `<div class="tag-row">${tags}</div>` : ""}
        ${links ? `<div class="track-links">${links}</div>` : ""}
      </div>
    </article>
  `;
}

function renderTrace(payload) {
  if (!payload || !elements.debugMode.checked) {
    elements.traceStatus.textContent = "hidden";
    elements.traceOutput.textContent = "Trace hidden.";
    return;
  }

  elements.traceStatus.textContent = payload.trace ? "visible" : "empty";
  elements.traceOutput.textContent = JSON.stringify(
    {
      status: payload.status,
      tool_calls: payload.tool_calls,
      trace: payload.trace,
    },
    null,
    2,
  );
}

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="avatar">${role === "user" ? "ME" : "AI"}</div>
    <div class="bubble"><p>${escapeHtml(text)}</p></div>
  `;
  elements.messages.appendChild(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function updateAgentMessage(article, text) {
  const paragraph = article.querySelector("p");
  paragraph.textContent = text;
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function clearChat() {
  elements.messages.innerHTML = "";
  appendMessage("agent", "Ready.");
  state.lastResponse = null;
  renderRecommendations([]);
  renderTrace(null);
}

function setLoading(isLoading) {
  elements.sendButton.disabled = isLoading;
  elements.sendButton.textContent = isLoading ? "Sending" : "Send";
}

function apiBaseUrl() {
  return elements.apiUrl.value.replace(/\/$/, "");
}

function showToast(message, isError = false) {
  const toast = document.createElement("div");
  toast.className = `toast${isError ? " error" : ""}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  window.setTimeout(() => toast.remove(), 2800);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
