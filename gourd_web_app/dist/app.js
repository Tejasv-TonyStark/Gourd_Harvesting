// Class IDs from the four-class best_wb YOLO11n checkpoint.
const CLASSES = ["bitter gourd", "bottle gourd", "ivy gourd", "snake gourd"];
const DETECTION_THRESHOLD = 0.25;
const MATURITY_THRESHOLD = 0.46;
const MODEL_BASE = "models/";

// Lead with the human motivation before asking visitors to try the technology.
document.querySelector(".hero")?.after(document.querySelector("#story"));

const $ = (id) => document.getElementById(id);
const els = {
  uploadTab: $("uploadTab"), cameraTab: $("cameraTab"), uploadPanel: $("uploadPanel"),
  cameraPanel: $("cameraPanel"), previewPanel: $("previewPanel"), loadingPanel: $("loadingPanel"),
  resultsPanel: $("resultsPanel"), fileInput: $("fileInput"), dropZone: $("dropZone"),
  video: $("cameraVideo"), cameraEmpty: $("cameraEmpty"), startCamera: $("startCamera"),
  capture: $("captureButton"), resultCanvas: $("resultCanvas"), sourceCanvas: $("sourceCanvas"),
  analyse: $("analyseButton"), chooseAnother: $("chooseAnother"), scanAgain: $("scanAgain"),
  imageBadge: $("imageBadge"), loadingTitle: $("loadingTitle"), loadingText: $("loadingText"),
  progressBar: $("progressBar"), progressLabel: $("progressLabel"), resultList: $("resultList"),
  resultHeadline: $("resultHeadline")
};

let stream = null;
let detectorSession = null;
let classifierSession = null;
let modelsPromise = null;
let currentImageName = "field photo";

document.querySelectorAll(".reveal").forEach((node) => new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => { if (entry.isIntersecting) { entry.target.classList.add("visible"); observer.unobserve(entry.target); } });
}, { threshold: 0.12 }).observe(node));

function setTab(which) {
  const upload = which === "upload";
  els.uploadTab.classList.toggle("active", upload);
  els.cameraTab.classList.toggle("active", !upload);
  els.uploadTab.setAttribute("aria-selected", upload);
  els.cameraTab.setAttribute("aria-selected", !upload);
  els.uploadPanel.hidden = !upload;
  els.cameraPanel.hidden = upload;
  els.uploadPanel.classList.toggle("active", upload);
  els.cameraPanel.classList.toggle("active", !upload);
  els.previewPanel.hidden = true;
  els.resultsPanel.hidden = true;
  if (upload) stopCamera();
}

els.uploadTab.addEventListener("click", () => setTab("upload"));
els.cameraTab.addEventListener("click", () => setTab("camera"));
els.fileInput.addEventListener("change", () => els.fileInput.files[0] && useFile(els.fileInput.files[0]));
["dragenter", "dragover"].forEach((event) => els.dropZone.addEventListener(event, (e) => { e.preventDefault(); els.dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => els.dropZone.addEventListener(event, (e) => { e.preventDefault(); els.dropZone.classList.remove("dragging"); }));
els.dropZone.addEventListener("drop", (e) => { const file = e.dataTransfer.files[0]; if (file?.type.startsWith("image/")) useFile(file); });

async function useFile(file) {
  if (file.size > 20 * 1024 * 1024) return showError("Please choose an image smaller than 20 MB.");
  currentImageName = file.name;
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  placeImage(bitmap);
  bitmap.close();
}

function placeImage(image) {
  const maxSide = 1600;
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  els.sourceCanvas.width = Math.round(image.width * scale);
  els.sourceCanvas.height = Math.round(image.height * scale);
  const sourceCtx = els.sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceCtx.drawImage(image, 0, 0, els.sourceCanvas.width, els.sourceCanvas.height);
  els.resultCanvas.width = els.sourceCanvas.width;
  els.resultCanvas.height = els.sourceCanvas.height;
  els.resultCanvas.getContext("2d").drawImage(els.sourceCanvas, 0, 0);
  els.uploadPanel.hidden = true;
  els.cameraPanel.hidden = true;
  els.previewPanel.hidden = false;
  els.loadingPanel.hidden = true;
  els.resultsPanel.hidden = true;
  els.imageBadge.textContent = "Ready to analyse";
}

els.startCamera.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
    els.video.srcObject = stream;
    els.cameraEmpty.hidden = true;
    els.capture.disabled = false;
  } catch (error) { showError("Camera access was unavailable. Check browser permission or upload a photo instead."); }
});

els.capture.addEventListener("click", () => {
  if (!els.video.videoWidth) return;
  currentImageName = "camera capture";
  placeImage(els.video);
  stopCamera();
});

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  els.video.srcObject = null;
  els.cameraEmpty.hidden = false;
  els.capture.disabled = true;
}

function resetScanner() {
  stopCamera();
  els.fileInput.value = "";
  els.uploadPanel.hidden = false;
  els.cameraPanel.hidden = true;
  els.previewPanel.hidden = true;
  els.loadingPanel.hidden = true;
  els.resultsPanel.hidden = true;
  els.analyse.hidden = false;
  els.chooseAnother.textContent = "Choose another";
  setTab("upload");
  document.querySelector("#scanner").scrollIntoView({ behavior: "smooth", block: "start" });
}
els.chooseAnother.addEventListener("click", resetScanner);
els.scanAgain.addEventListener("click", resetScanner);

function progress(percent, title, detail) {
  els.progressBar.style.width = `${percent}%`;
  els.progressLabel.textContent = `${Math.round(percent)}%`;
  if (title) els.loadingTitle.textContent = title;
  if (detail) els.loadingText.textContent = detail;
}

async function fetchModel(url, start, end) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${url}`);
  const total = Number(response.headers.get("content-length")) || 1;
  if (!response.body) return response.arrayBuffer();
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    progress(start + (received / total) * (end - start), "Loading the research models…", "This one-time download keeps your image processing on this device.");
  }
  const data = new Uint8Array(received); let offset = 0;
  chunks.forEach((chunk) => { data.set(chunk, offset); offset += chunk.length; });
  return data.buffer;
}

async function fetchModelParts(urls, start, end) {
  const buffers = [];
  const span = (end - start) / urls.length;
  for (let i = 0; i < urls.length; i++) buffers.push(await fetchModel(urls[i], start + i * span, start + (i + 1) * span));
  const total = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const joined = new Uint8Array(total); let offset = 0;
  buffers.forEach((buffer) => { joined.set(new Uint8Array(buffer), offset); offset += buffer.byteLength; });
  return joined.buffer;
}

async function loadModels() {
  if (detectorSession && classifierSession) return;
  if (modelsPromise) return modelsPromise;
  modelsPromise = (async () => {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    const detectorBytes = await fetchModelParts([
      `${MODEL_BASE}gourd_detector_best_wb.part1.bin`,
      `${MODEL_BASE}gourd_detector_best_wb.part2.bin`,
      `${MODEL_BASE}gourd_detector_best_wb.part3.bin`
    ], 3, 52);
    const classifierBytes = await fetchModel(`${MODEL_BASE}gourd_maturity_mobilenetv3.onnx`, 52, 72);
    progress(76, "Starting the vision engine…", "Preparing detection and maturity analysis.");
    // Keep YOLO post-processing on WASM so browser inference matches the
    // validated ONNX CPU output.
    detectorSession = await ort.InferenceSession.create(detectorBytes, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all"
    });
    const classifierProviders = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    try {
      classifierSession = await ort.InferenceSession.create(classifierBytes, { executionProviders: classifierProviders, graphOptimizationLevel: "all" });
    } catch (classifierError) {
      classifierSession = await ort.InferenceSession.create(classifierBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    progress(82, "Models ready", "Looking for gourds in your photograph.");
  })().catch((error) => { modelsPromise = null; throw error; });
  return modelsPromise;
}

function detectorInput() {
  const size = 640;
  const scale = Math.min(size / els.sourceCanvas.width, size / els.sourceCanvas.height);
  const drawW = Math.round(els.sourceCanvas.width * scale), drawH = Math.round(els.sourceCanvas.height * scale);
  const padX = Math.floor((size - drawW) / 2), padY = Math.floor((size - drawH) / 2);
  const canvas = document.createElement("canvas"); canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, size, size);
  ctx.drawImage(els.sourceCanvas, padX, padY, drawW, drawH);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const data = new Float32Array(3 * size * size), plane = size * size;
  for (let i = 0; i < plane; i++) { data[i] = rgba[i * 4] / 255; data[plane + i] = rgba[i * 4 + 1] / 255; data[2 * plane + i] = rgba[i * 4 + 2] / 255; }
  return { tensor: new ort.Tensor("float32", data, [1, 3, size, size]), scale, padX, padY };
}

function classifierInput(box) {
  const [x1, y1, x2, y2] = box;
  const width = Math.max(1, x2 - x1), height = Math.max(1, y2 - y1);
  const scale = 256 / Math.min(width, height);
  const resizedW = Math.round(width * scale), resizedH = Math.round(height * scale);
  const temp = document.createElement("canvas"); temp.width = resizedW; temp.height = resizedH;
  temp.getContext("2d").drawImage(els.sourceCanvas, x1, y1, width, height, 0, 0, resizedW, resizedH);
  const crop = document.createElement("canvas"); crop.width = 224; crop.height = 224;
  crop.getContext("2d").drawImage(temp, Math.floor((resizedW - 224) / 2), Math.floor((resizedH - 224) / 2), 224, 224, 0, 0, 224, 224);
  const rgba = crop.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, 224, 224).data;
  const data = new Float32Array(3 * 224 * 224), plane = 224 * 224;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < plane; i++) {
    data[i] = (rgba[i * 4] / 255 - mean[0]) / std[0];
    data[plane + i] = (rgba[i * 4 + 1] / 255 - mean[1]) / std[1];
    data[2 * plane + i] = (rgba[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", data, [1, 3, 224, 224]);
}

function softmaxPair(a, b) { const max = Math.max(a, b), ea = Math.exp(a - max), eb = Math.exp(b - max); return eb / (ea + eb); }

async function analyse() {
  els.previewPanel.hidden = true; els.resultsPanel.hidden = true; els.loadingPanel.hidden = false;
  progress(2, "Preparing the research models…", "The first visit downloads the models. Later scans are faster.");
  try {
    await loadModels();
    progress(84, "Detecting gourds…", "YOLO11n is locating and identifying visible produce.");
    const prep = detectorInput();
    const detectorOutput = await detectorSession.run({ [detectorSession.inputNames[0]]: prep.tensor });
    const output = detectorOutput[detectorSession.outputNames[0]];
    const detections = [];
    for (let i = 0; i < output.data.length; i += 6) {
      const confidence = output.data[i + 4];
      if (confidence < DETECTION_THRESHOLD) continue;
      const cls = Math.round(output.data[i + 5]);
      let x1 = (output.data[i] - prep.padX) / prep.scale, y1 = (output.data[i + 1] - prep.padY) / prep.scale;
      let x2 = (output.data[i + 2] - prep.padX) / prep.scale, y2 = (output.data[i + 3] - prep.padY) / prep.scale;
      x1 = Math.max(0, Math.min(els.sourceCanvas.width, x1)); y1 = Math.max(0, Math.min(els.sourceCanvas.height, y1));
      x2 = Math.max(0, Math.min(els.sourceCanvas.width, x2)); y2 = Math.max(0, Math.min(els.sourceCanvas.height, y2));
      if (x2 - x1 <= 3 || y2 - y1 <= 3) continue;
      if (cls >= 0 && cls < CLASSES.length) {
        detections.push({ box: [x1, y1, x2, y2], cls, detectionConfidence: confidence });
      }
    }
    for (let i = 0; i < detections.length; i++) {
      progress(86 + ((i + 1) / Math.max(1, detections.length)) * 11, "Estimating maturity…", `Examining detected gourd ${i + 1} of ${detections.length}.`);
      const input = classifierInput(detections[i].box);
      const outputs = await classifierSession.run({ [classifierSession.inputNames[0]]: input });
      const logits = outputs[classifierSession.outputNames[0]].data;
      const matureProbability = softmaxPair(logits[0], logits[1]);
      detections[i].matureProbability = matureProbability;
      detections[i].maturity = matureProbability >= MATURITY_THRESHOLD ? "Mature" : "Immature";
      detections[i].maturityConfidence = matureProbability >= MATURITY_THRESHOLD ? matureProbability : 1 - matureProbability;
      await new Promise(requestAnimationFrame);
    }
    progress(100, "Analysis complete", "Preparing your result.");
    renderResults(detections);
  } catch (error) {
    console.error(error);
    showError("The on-device model could not start in this browser. Try current Chrome or Edge, check your connection for the first model download, then try again.");
  }
}
els.analyse.addEventListener("click", analyse);

function renderResults(items) {
  const canvas = els.resultCanvas, ctx = canvas.getContext("2d");
  ctx.drawImage(els.sourceCanvas, 0, 0);
  const line = Math.max(3, canvas.width / 260);
  ctx.lineWidth = line; ctx.font = `700 ${Math.max(15, canvas.width / 42)}px DM Sans, sans-serif`; ctx.textBaseline = "bottom";
  items.forEach((item, index) => {
    const [x1, y1, x2, y2] = item.box, label = `${index + 1}. ${CLASSES[item.cls]} · ${item.maturity}`;
    ctx.strokeStyle = "#f3ba43"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const metrics = ctx.measureText(label), textH = Math.max(25, canvas.width / 30);
    ctx.fillStyle = "rgba(23,62,42,.92)"; ctx.fillRect(x1, Math.max(0, y1 - textH), metrics.width + 18, textH);
    ctx.fillStyle = "#fffdf6"; ctx.fillText(label, x1 + 9, Math.max(textH - 5, y1 - 5));
  });
  els.imageBadge.textContent = items.length
    ? `${items.length} detection${items.length === 1 ? "" : "s"}`
    : "No detection";
  els.resultHeadline.textContent = items.length
    ? `${items.length} gourd${items.length === 1 ? "" : "s"} analysed`
    : "No gourd detected";
  const resultItems = items.length ? items.map((item, index) => `
    <article class="result-item">
      <span class="result-index">${index + 1}</span>
      <div class="result-value"><small>Detected variety</small><strong>${escapeHtml(CLASSES[item.cls])}</strong><div class="confidence-bar"><span style="width:${item.detectionConfidence * 100}%"></span></div><small>${percent(item.detectionConfidence)} detection confidence</small></div>
      <div class="result-value"><small>Maturity estimate</small><strong>${item.maturity}</strong><div class="confidence-bar"><span style="width:${item.maturityConfidence * 100}%"></span></div><small>${percent(item.maturityConfidence)} classification confidence</small></div>
      <div class="result-value"><small>Suggested next step</small><strong>Verify in the field</strong><small>Check colour, size, firmness, and local guidance</small></div>
    </article>`).join("") : `<article class="result-item"><span class="result-index">?</span><div class="result-value"><small>What happened</small><strong>No supported gourd was found</strong><small>Try a closer, brighter photograph with the whole fruit visible.</small></div></article>`;
  els.resultList.innerHTML = resultItems;
  els.loadingPanel.hidden = true;
  els.previewPanel.hidden = false;
  els.resultsPanel.hidden = false;
  els.analyse.hidden = true;
  els.chooseAnother.textContent = "Choose another photo";
  els.resultsPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(message) {
  els.loadingPanel.hidden = true;
  els.previewPanel.hidden = false;
  els.resultsPanel.hidden = false;
  els.resultHeadline.textContent = "Analysis unavailable";
  els.resultList.innerHTML = `<article class="result-item"><span class="result-index">!</span><div class="result-value"><small>Please try again</small><strong>${escapeHtml(message)}</strong></div></article>`;
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function escapeHtml(value) { const node = document.createElement("span"); node.textContent = value; return node.innerHTML; }

window.addEventListener("pagehide", stopCamera);
