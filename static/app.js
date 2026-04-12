const form = document.getElementById("caption-form");
const imageInput = document.getElementById("image-input");
const uploadText = document.getElementById("upload-text");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const resultMeta = document.getElementById("result-meta");
const captionOutput = document.getElementById("caption-output");
const compareResults = document.getElementById("compare-results");
const submitButton = document.getElementById("submit-button");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderSingleResult(payload) {
  compareResults.innerHTML = `
    <article class="result-card">
      <p class="result-title">${escapeHtml(payload.mode_label)}</p>
      <p class="result-time">${payload.elapsed_ms} ms</p>
      <p class="result-caption">${escapeHtml(payload.caption)}</p>
    </article>
  `;
}

function renderAllResults(payload) {
  const cards = payload.results
    .map(
      (item) => `
      <article class="result-card">
        <p class="result-title">${escapeHtml(item.mode_label)}</p>
        <p class="result-time">${item.elapsed_ms} ms</p>
        <p class="result-caption">${escapeHtml(item.caption)}</p>
      </article>
    `,
    )
    .join("");

  compareResults.innerHTML = cards;
}

function setPreview(file) {
  if (!file) {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewPlaceholder.style.display = "block";
    uploadText.textContent = "Drop image here or click to browse";
    return;
  }

  uploadText.textContent = `Selected: ${file.name}`;
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
}

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  setPreview(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = imageInput.files;
  if (!file) {
    resultMeta.textContent = "Validation error";
    compareResults.innerHTML = `<p class="caption">Please choose an image first.</p>`;
    return;
  }

  const formData = new FormData(form);
  submitButton.disabled = true;
  submitButton.textContent = "Generating...";
  resultMeta.textContent = "Request in progress";
  compareResults.innerHTML = `<p class="caption">Running model inference, please wait...</p>`;

  try {
    const response = await fetch("/api/caption", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Unknown API error");
    }

    if (Array.isArray(payload.results)) {
      resultMeta.textContent = `Compared ${payload.count} modes | total ${payload.elapsed_ms} ms`;
      renderAllResults(payload);
    } else {
      resultMeta.textContent = `${payload.mode_label} | ${payload.elapsed_ms} ms`;
      renderSingleResult(payload);
    }
  } catch (error) {
    resultMeta.textContent = "Request failed";
    compareResults.innerHTML = `<p class="caption">${escapeHtml(error.message)}</p>`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Generate Caption";
  }
});
