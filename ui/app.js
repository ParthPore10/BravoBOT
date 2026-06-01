const form = document.querySelector("#chatForm");
const messages = document.querySelector("#messages");
const queryInput = document.querySelector("#queryInput");
const sendButton = document.querySelector("#sendButton");
const apiUrlInput = document.querySelector("#apiUrl");
const candidateKInput = document.querySelector("#candidateK");
const finalKInput = document.querySelector("#finalK");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const fileLabel = document.querySelector("#fileLabel");
const replaceIndex = document.querySelector("#replaceIndex");
const uploadButton = document.querySelector("#uploadButton");
const uploadButtonText = document.querySelector("#uploadButtonText");
const uploadResult = document.querySelector("#uploadResult");
const uploadProgress = document.querySelector("#uploadProgress");
const selectedFiles = document.querySelector("#selectedFiles");
let hasIndexedSources = false;
let isUploading = false;

function setStatus(text, state = "ready") {
  statusText.textContent = text;
  statusDot.className = "status-dot";

  if (state === "loading") {
    statusDot.classList.add("loading");
  }

  if (state === "error") {
    statusDot.classList.add("error");
  }
}

function setUploadProgress(active) {
  uploadProgress.classList.toggle("active", active);
  uploadProgress.setAttribute("aria-hidden", String(!active));
}

function formatUploadSummary(result) {
  const fileCount = result.files?.length || 0;
  const skippedFiles = result.skipped_files || [];
  const skippedCount = skippedFiles.length;
  const newChunks = result.new_chunks || 0;
  const totalChunks = result.total_chunks || 0;

  if (skippedCount && !newChunks) {
    return `${skippedCount} duplicate file(s) skipped. ${totalChunks} chunks remain indexed.`;
  }

  if (skippedCount) {
    return `${fileCount} file(s) processed, ${newChunks} new chunks, ${totalChunks} total chunks, ${skippedCount} duplicate skipped.`;
  }

  return `${fileCount} file(s), ${newChunks} new chunks, ${totalChunks} total chunks indexed.`;
}

function formatUploadAssistantMessage(result) {
  const skippedCount = result.skipped_files?.length || 0;
  const newChunks = result.new_chunks || 0;

  if (skippedCount && !newChunks) {
    return "That file is already indexed, so I skipped the duplicate. You can ask questions against the existing sources.";
  }

  if (skippedCount) {
    return "Your new document sources are indexed. I skipped duplicate files that were already in the source library.";
  }

  return "Your document sources are indexed. Ask a question about them.";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }

  return Number(value).toFixed(4);
}

function renderSources(sources = []) {
  if (!sources.length) {
    return "";
  }

  const sourceCards = sources
    .map((source, index) => {
      return `
        <div class="source">
          <div class="source-title">
            <span>Source ${index + 1}</span>
            <span>${escapeHtml(source.source_file || "Unknown file")}</span>
          </div>
          <div class="source-meta">
            <span>Page ${escapeHtml(source.page_number || "unknown")}</span>
          </div>
          ${
            source.preview_text
              ? `<p class="source-preview">${escapeHtml(source.preview_text)}</p>`
              : ""
          }
        </div>
      `;
    })
    .join("");

  return `<div class="sources">${sourceCards}</div>`;
}

function addMessage(role, content, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = role === "user" ? "You" : "";
  const avatarClass = role === "assistant" ? "avatar avatar-character" : "avatar";
  article.innerHTML = `
    <div class="${avatarClass}">${avatar}</div>
    <div class="bubble">
      <p>${escapeHtml(content)}</p>
      ${renderSources(sources)}
    </div>
  `;

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function addTypingIndicator() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.dataset.typing = "true";
  article.innerHTML = `
    <div class="avatar avatar-character"></div>
    <div class="bubble">
      <div class="typing-indicator" aria-label="Assistant is typing">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  `;

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function addStreamingAssistantMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="avatar avatar-character"></div>
    <div class="bubble">
      <p class="stream-answer"></p>
      <div class="stream-sources"></div>
    </div>
  `;

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function updateStreamingAssistantMessage(article, content, sources = []) {
  const answer = article.querySelector(".stream-answer");
  const sourceSlot = article.querySelector(".stream-sources");

  answer.textContent = content;
  sourceSlot.innerHTML = renderSources(sources);
  messages.scrollTop = messages.scrollHeight;
}

async function askQuestion(query) {
  const response = await fetch(apiUrlInput.value.trim(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      candidate_k: Number(candidateKInput.value),
      final_k: Number(finalKInput.value),
    }),
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = payload.detail || `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return payload;
}

function getChatStreamUrl() {
  return apiUrlInput.value.trim().replace(/\/chat\/?$/, "/chat/stream");
}

async function askQuestionStream(query, onToken, onSources) {
  const response = await fetch(getChatStreamUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      candidate_k: Number(candidateKInput.value),
      final_k: Number(finalKInput.value),
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail || `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let sources = [];

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }

      const event = JSON.parse(line);

      if (event.type === "token") {
        answer += event.text || "";
        onToken(answer);
      }

      if (event.type === "sources") {
        sources = event.sources || [];
        onSources(sources);
      }

      if (event.type === "error") {
        throw new Error(event.message || "Streaming failed.");
      }
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);

    if (event.type === "token") {
      answer += event.text || "";
      onToken(answer);
    }

    if (event.type === "sources") {
      sources = event.sources || [];
      onSources(sources);
    }

    if (event.type === "error") {
      throw new Error(event.message || "Streaming failed.");
    }
  }

  return { answer, sources };
}

function getUploadUrl() {
  return apiUrlInput.value.trim().replace(/\/chat\/?$/, "/upload");
}

function getApiBaseUrl() {
  return apiUrlInput.value.trim().replace(/\/chat\/?$/, "");
}

function getSourceStatusUrl() {
  return apiUrlInput.value.trim().replace(/\/chat\/?$/, "/sources/status");
}

async function loadSourceStatus() {
  try {
    const response = await fetch(getSourceStatusUrl());
    if (!response.ok) {
      return;
    }

    const status = await response.json();
    hasIndexedSources = Boolean(status.has_sources);

    if (hasIndexedSources) {
      fileLabel.textContent = "Sources indexed";
      selectedFiles.textContent = status.files?.length ? status.files.join(", ") : "";
      uploadResult.className = "upload-result";
      uploadResult.textContent = `${status.files.length} file(s), ${status.total_chunks} chunks indexed.`;
      setStatus("Ready");
    }
  } catch {
    // Backend may not be running yet. Keep the UI usable for file selection.
  }
}

async function uploadDocuments() {
  const files = Array.from(fileInput.files || []);

  if (!files.length) {
    fileInput.click();
    return null;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("replace", String(replaceIndex.checked));

  const response = await fetch(getUploadUrl(), {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = payload.detail || `Upload failed with status ${response.status}`;
    throw new Error(detail);
  }

  return payload;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollUploadJob(jobId) {
  while (true) {
    const response = await fetch(`${getApiBaseUrl()}/jobs/${jobId}`);
    const job = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = job.detail || `Job status failed with status ${response.status}`;
      throw new Error(detail);
    }

    if (job.status === "completed") {
      return job.result;
    }

    if (job.status === "failed") {
      throw new Error(job.error || "Indexing failed.");
    }

    uploadResult.textContent = `Indexing documents... (${job.status})`;
    await sleep(1000);
  }
}

async function resolveUploadResult(uploadResponse) {
  if (uploadResponse?.job_id) {
    uploadResult.textContent = "Upload accepted. Indexing documents...";
    return pollUploadJob(uploadResponse.job_id);
  }

  return uploadResponse;
}

async function startUpload() {
  if (isUploading) {
    return;
  }

  const selectedFiles = Array.from(fileInput.files || []);

  if (!selectedFiles.length) {
    uploadResult.className = "upload-result";
    uploadResult.textContent = "Choose a PDF or TXT file to index.";
    return;
  }

  isUploading = true;
  uploadButton.classList.add("disabled");
  uploadButton.setAttribute("aria-disabled", "true");
  uploadButtonText.textContent = "Uploading";
  uploadResult.className = "upload-result";
  uploadResult.textContent = `Uploading and indexing ${selectedFiles.length} file(s)...`;
  setUploadProgress(true);
  setStatus("Indexing", "loading");

  try {
    const uploadResponse = await uploadDocuments();
    const result = await resolveUploadResult(uploadResponse);

    if (!result) {
      return;
    }

    hasIndexedSources = true;
    fileLabel.textContent = "Sources indexed";
    selectedFiles.textContent = result.files?.length ? result.files.join(", ") : "";
    uploadResult.textContent = formatUploadSummary(result);
    fileInput.value = "";
    addMessage("assistant", formatUploadAssistantMessage(result));
    setStatus("Ready");
  } catch (error) {
    uploadResult.className = "upload-result error";
    uploadResult.textContent = error.message;
    setStatus("Upload failed", "error");
  } finally {
    setUploadProgress(false);
    isUploading = false;
    uploadButton.classList.remove("disabled");
    uploadButton.setAttribute("aria-disabled", "false");
    uploadButtonText.textContent = "Upload";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  if (!hasIndexedSources) {
    addMessage("assistant", "Upload a PDF or TXT source before asking questions.");
    fileInput.click();
    setStatus("Upload required", "error");
    return;
  }

  addMessage("user", query);
  const typingIndicator = addTypingIndicator();
  let assistantMessage = null;
  let streamedAnswer = "";
  let streamedSources = [];
  queryInput.value = "";
  queryInput.style.height = "";
  sendButton.disabled = true;
  setStatus("Thinking", "loading");

  try {
    const result = await askQuestionStream(
      query,
      (answer) => {
        streamedAnswer = answer;

        if (!assistantMessage) {
          typingIndicator.remove();
          assistantMessage = addStreamingAssistantMessage();
        }

        updateStreamingAssistantMessage(assistantMessage, streamedAnswer, streamedSources);
      },
      (sources) => {
        streamedSources = sources;

        if (!assistantMessage) {
          typingIndicator.remove();
          assistantMessage = addStreamingAssistantMessage();
        }

        updateStreamingAssistantMessage(
          assistantMessage,
          streamedAnswer || "No answer returned.",
          streamedSources
        );
      }
    );

    if (!assistantMessage) {
      typingIndicator.remove();
      addMessage("assistant", result.answer || "No answer returned.", result.sources || []);
    }

    setStatus("Ready");
  } catch (error) {
    typingIndicator.remove();
    addMessage("assistant", `Error: ${error.message}`);
    setStatus("Request failed", "error");
  } finally {
    sendButton.disabled = false;
    queryInput.focus();
  }
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);

  if (!files.length) {
    fileLabel.textContent = "Upload your documents";
    selectedFiles.textContent = "";
    return;
  }

  fileLabel.textContent = "Files selected";
  selectedFiles.textContent = files.map((file) => file.name).join(", ");
  uploadButtonText.textContent = "Uploading";
  uploadResult.className = "upload-result";
  uploadResult.textContent = `${files.length} file(s) selected. Starting upload...`;
  startUpload();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  startUpload();
});

uploadButton.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.value = "";
    fileInput.click();
  }
});

uploadButton.addEventListener("click", () => {
  if (isUploading) {
    return;
  }

  fileInput.value = "";
  fileInput.click();
});

loadSourceStatus();
