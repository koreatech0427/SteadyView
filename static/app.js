const statePanel = document.querySelector("#statePanel");
const stateViews = Array.from(document.querySelectorAll(".state-view"));
const videoInput = document.querySelector("#videoInput");
const dropZone = document.querySelector(".drop-zone");
const fileBadge = document.querySelector("#fileBadge");
const fileIcon = document.querySelector(".file-icon");
const fileName = document.querySelector("#fileName");
const fileSize = document.querySelector("#fileSize");
const removeFileButton = document.querySelector("#removeFileButton");
const emptyPreview = document.querySelector("#emptyPreview");
const previewVideo = document.querySelector("#previewVideo");
const originalVideo = document.querySelector("#originalVideo");
const resultVideo = document.querySelector("#resultVideo");
const processButton = document.querySelector("#processButton");
const resetButton = document.querySelector("#resetButton");
const comparePlayButton = document.querySelector("#comparePlayButton");
const progressBar = document.querySelector("#progressBar");
const progressPct = document.querySelector("#progressPct");
const progressLabel = document.querySelector("#progressLabel");
const progressEta = document.querySelector("#progressEta");
const progressElapsed = document.querySelector("#progressElapsed");
const steps = document.querySelector("#steps");
const cancelJobButton = document.querySelector("#cancelJobButton");
const downloadLink = document.querySelector("#downloadLink");
const alertBox = document.querySelector("#alertBox");
const featureCheckboxes = Array.from(document.querySelectorAll("input[name='feature']"));

let selectedFile = null;
let previewUrl = null;
let resultUrl = null;
let activeJobId = null;
let pollingTimer = null;
let progressTimer = null;
let displayedProgress = 0;
let processStartedAt = null;
let previewFallbackTimer = null;
let comparePlaying = false;
let syncingComparison = false;
let cancelRequested = false;

// Cloudflare 경유 시 한 번에 큰 파일을 보내면 막힐 수 있어서 64MiB부터 분할 업로드를 사용한다.
const CHUNKED_UPLOAD_THRESHOLD = 64 * 1024 * 1024;
const UPLOAD_CHUNK_SIZE = 16 * 1024 * 1024;
const featureOrder = ["Superresolution", "Stabilization", "Upright Correction"];
/*
const displayStepLabels = ["업로드 준비", "보정 처리", "초해상도 처리", "결과 정리"];
const stepLabels = ["영상 분석", "프레임 추출", "AI 모델 적용", "영상 보정"];

*/
const stepLabels = [
  "\uc5c5\ub85c\ub4dc \uc900\ube44",
  "\uc601\uc0c1 \ubd84\uc11d",
  "\uc601\uc0c1 \ubcf4\uc815",
  "\uacb0\uacfc \uc815\ub9ac",
];

setUiState("input");

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file) {
    videoInput.files = event.dataTransfer.files;
    handleFile(file);
  }
});

videoInput.addEventListener("change", () => {
  const [file] = videoInput.files;
  if (file) {
    handleFile(file);
  }
});

removeFileButton.addEventListener("click", () => {
  resetSelectedFile();
  hideAlert();
});

featureCheckboxes.forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    hideAlert();
    clearResult();
  });
});

processButton.addEventListener("click", async () => {
  hideAlert();
  if (!selectedFile) {
    showAlert("먼저 영상을 업로드하세요.");
    return;
  }

  const option = getSelectedOption();
  if (!option) {
    showAlert("복원 옵션을 하나 이상 선택하세요.");
    return;
  }

  setUiState("processing");
  setBusy(true);
  cancelRequested = false;
  updateCancelButton(false);
  processStartedAt = Date.now();
  startProgress();

  try {
    const job = selectedFile.size >= CHUNKED_UPLOAD_THRESHOLD
      ? await createChunkedJob(selectedFile, option)
      : await createDirectJob(selectedFile, option);
    activeJobId = job.id;
    updateCancelButton(false);
    updateProgress(job.progress || 0, job.message);
    pollJob(activeJobId);
  } catch (error) {
    stopPolling();
    stopProgress();
    setUiState("input");
    setBusy(false);
    updateCancelButton(false);
    showAlert(error.message);
  }
});

async function createDirectJob(file, option) {
  const formData = new FormData();
  formData.append("option", option);
  formData.append("file", file);

  updateProgress(1, "영상 업로드 중...");
  const response = await fetch("/api/jobs", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json();
}

async function createChunkedJob(file, option) {
  const totalChunks = Math.ceil(file.size / UPLOAD_CHUNK_SIZE);
  const initForm = new FormData();
  initForm.append("file_name", file.name);
  initForm.append("file_size", String(file.size));
  initForm.append("total_chunks", String(totalChunks));

  updateProgress(0, `영상 업로드 준비 중... 0/${totalChunks}`);
  const initResponse = await fetch("/api/uploads", {
    method: "POST",
    body: initForm,
  });
  if (!initResponse.ok) {
    throw new Error(await readError(initResponse));
  }

  const upload = await initResponse.json();
  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * UPLOAD_CHUNK_SIZE;
    const end = Math.min(start + UPLOAD_CHUNK_SIZE, file.size);
    const chunkForm = new FormData();
    chunkForm.append("chunk_index", String(index));
    chunkForm.append("chunk", file.slice(start, end), `${file.name}.part${index}`);

    const uploadPercent = Math.max(1, Math.round(((index + 1) / totalChunks) * 5));
    updateProgress(uploadPercent, `영상 업로드 중... ${index + 1}/${totalChunks}`);
    const chunkResponse = await fetch(`/api/uploads/${upload.id}/chunks`, {
      method: "POST",
      body: chunkForm,
    });
    if (!chunkResponse.ok) {
      throw new Error(await readError(chunkResponse));
    }
  }

  const completeForm = new FormData();
  completeForm.append("option", option);
  updateProgress(5, "업로드 파일을 합치는 중...");
  const completeResponse = await fetch(`/api/uploads/${upload.id}/complete`, {
    method: "POST",
    body: completeForm,
  });
  if (!completeResponse.ok) {
    throw new Error(await readError(completeResponse));
  }

  return completeResponse.json();
}

resetButton.addEventListener("click", resetAll);
if (cancelJobButton) {
  cancelJobButton.addEventListener("click", cancelActiveJob);
}

comparePlayButton.addEventListener("click", async () => {
  hideAlert();
  if (!previewUrl || !resultUrl) {
    showAlert("비교하려면 원본과 결과 영상이 모두 필요합니다.");
    return;
  }

  if (comparePlaying) {
    pauseComparison();
    return;
  }

  await playComparison();
});

[originalVideo, resultVideo].forEach((video) => {
  video.addEventListener("pause", () => {
    if (comparePlaying && !video.ended) {
      pauseComparison();
    }
  });

  video.addEventListener("ended", () => {
    if (comparePlaying) {
      pauseComparison();
    }
  });

  video.addEventListener("seeked", () => {
    if (!comparePlaying || syncingComparison) {
      return;
    }

    const peer = video === originalVideo ? resultVideo : originalVideo;
    syncVideoTime(peer, video.currentTime);
  });
});

async function handleFile(file) {
  selectedFile = file;
  hideAlert();
  clearResult();
  setUiState("input");

  const extension = file.name.split(".").pop() || "MP4";
  fileIcon.textContent = extension.slice(0, 4).toUpperCase();
  fileBadge.classList.remove("hidden");
  dropZone.classList.add("hidden");
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;

  setPreview(file);
  scheduleServerPreviewFallback(file);
}

async function loadServerPreview(file) {
  if (file !== selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/preview", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const previewBlob = await response.blob();
    if (file === selectedFile) {
      setPreview(previewBlob);
    }
  } catch (error) {
    if (file === selectedFile) {
      showAlert(error.message);
    }
  }
}

function scheduleServerPreviewFallback(file) {
  clearPreviewFallback();

  const needsConversion = !file.type.includes("mp4") && !file.name.toLowerCase().endsWith(".mp4");
  if (needsConversion) {
    loadServerPreview(file);
    return;
  }

  previewFallbackTimer = window.setTimeout(() => {
    if (file === selectedFile && previewVideo.readyState < HTMLMediaElement.HAVE_METADATA) {
      loadServerPreview(file);
    }
  }, 1200);
}

function clearPreviewFallback() {
  if (previewFallbackTimer) {
    window.clearTimeout(previewFallbackTimer);
    previewFallbackTimer = null;
  }
}

function getSelectedOption() {
  const selectedFeatures = featureOrder.filter((feature) =>
    featureCheckboxes.some((checkbox) => checkbox.value === feature && checkbox.checked),
  );
  return selectedFeatures.join(" + ");
}

function setPreview(blob) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(blob);
  previewVideo.src = previewUrl;
  originalVideo.src = previewUrl;
  emptyPreview.classList.add("hidden");
  previewVideo.classList.remove("hidden");
  updateCompareButton();
}

function setResult(url, sourceName) {
  if (resultUrl) {
    if (resultUrl.startsWith("blob:")) {
      URL.revokeObjectURL(resultUrl);
    }
  }
  const outputName = `steadyview_${sourceName.replace(/\.[^.]+$/, ".mp4")}`;
  resultUrl = url;
  resultVideo.src = resultUrl;
  downloadLink.href = resultUrl;
  downloadLink.download = outputName;
  comparePlaying = false;
  setUiState("complete");
  updateCompareButton();
}

function clearResult() {
  pauseComparison();
  stopPolling();
  stopProgress();
  activeJobId = null;
  processStartedAt = null;
  cancelRequested = false;
  updateCancelButton(false);
  if (resultUrl) {
    if (resultUrl.startsWith("blob:")) {
      URL.revokeObjectURL(resultUrl);
    }
    resultUrl = null;
  }
  resultVideo.removeAttribute("src");
  updateCompareButton();
}

function resetSelectedFile() {
  clearResult();
  clearPreviewFallback();
  selectedFile = null;
  videoInput.value = "";
  fileBadge.classList.add("hidden");
  dropZone.classList.remove("hidden");
  emptyPreview.classList.remove("hidden");
  previewVideo.classList.add("hidden");
  previewVideo.removeAttribute("src");
  originalVideo.removeAttribute("src");
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  updateCompareButton();
}

function resetAll() {
  resetSelectedFile();
  hideAlert();
  setUiState("input");
}

async function playComparison() {
  const startTime = originalVideo.ended || resultVideo.ended ? 0 : originalVideo.currentTime;
  syncVideoTime(originalVideo, startTime);
  syncVideoTime(resultVideo, startTime);

  try {
    await Promise.all([originalVideo.play(), resultVideo.play()]);
    comparePlaying = true;
    updateCompareButton();
  } catch {
    pauseComparison();
    showAlert("브라우저가 동시 재생을 막았습니다. 버튼을 다시 눌러주세요.");
  }
}

function pauseComparison() {
  comparePlaying = false;
  originalVideo.pause();
  resultVideo.pause();
  updateCompareButton();
}

function syncVideoTime(video, time) {
  if (!Number.isFinite(time) || Number.isNaN(video.duration)) {
    return;
  }

  syncingComparison = true;
  if (Math.abs(video.currentTime - time) > 0.05) {
    video.currentTime = time;
  }
  window.setTimeout(() => {
    syncingComparison = false;
  }, 120);
}

function updateCompareButton() {
  comparePlayButton.disabled = !previewUrl || !resultUrl;
  comparePlayButton.textContent = comparePlaying ? "비교 영상 일시정지" : "비교 영상 동시 재생";
}

function updateCancelButton(isCancelling) {
  if (!cancelJobButton) {
    return;
  }

  cancelJobButton.disabled = !activeJobId || Boolean(isCancelling);
  cancelJobButton.textContent = isCancelling ? "중단 요청 중..." : "처리 중단";
}

async function pollJob(jobId) {
  stopPolling();

  const check = async () => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const job = await response.json();
      if (job.id !== activeJobId) {
        return;
      }

      updateProgress(job.progress || displayedProgress, job.message);
      updateCancelButton(job.status === "cancelling" || job.cancel_requested);

      if (job.status === "done") {
        stopPolling();
        finishProgress(job.message);
        updateCancelButton(false);
        await loadJobResult(job.id);
        setBusy(false);
        return;
      }

      if (job.status === "cancelled") {
        stopPolling();
        stopProgress();
        activeJobId = null;
        cancelRequested = false;
        setUiState("input");
        setBusy(false);
        updateCancelButton(false);
        showAlert(job.message || "영상 처리를 중단했습니다.");
        return;
      }

      if (job.status === "failed") {
        throw new Error(job.message || "영상 처리에 실패했습니다.");
      }
    } catch (error) {
      stopPolling();
      stopProgress();
      activeJobId = null;
      setUiState("input");
      setBusy(false);
      updateCancelButton(false);
      showAlert(error.message);
    }
  };

  await check();
  if (activeJobId === jobId) {
    pollingTimer = window.setInterval(check, 1500);
  }
}

async function cancelActiveJob() {
  if (!activeJobId || cancelRequested) {
    return;
  }

  cancelRequested = true;
  updateCancelButton(true);
  updateProgress(displayedProgress, "처리 중단 요청을 보냈습니다.");

  try {
    const response = await fetch(`/api/jobs/${activeJobId}/cancel`, {
      method: "POST",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const job = await response.json();
    updateProgress(job.progress || displayedProgress, job.message);
  } catch (error) {
    cancelRequested = false;
    updateCancelButton(false);
    showAlert(error.message);
  }
}

async function loadJobResult(jobId) {
  const resultUrl = `/api/jobs/${jobId}/result`;
  try {
    const response = await fetch(resultUrl, { method: "HEAD", cache: "no-store" });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
  } catch (error) {
    // 대용량 결과 파일은 Cloudflare/브라우저가 HEAD 확인을 끊는 경우가 있어,
    // job이 완료된 상태라면 결과 URL을 그대로 연결한다.
    console.warn("Result HEAD check failed; continuing with result URL.", error);
  }

  setResult(resultUrl, selectedFile.name);
  activeJobId = null;
  updateCancelButton(false);
}

function stopPolling() {
  if (pollingTimer) {
    window.clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

function startProgress() {
  stopProgress();
  displayedProgress = 0;
  renderSteps(0);
  updateElapsedTime();
  updateProgress(0);
  progressTimer = window.setInterval(updateElapsedTime, 1000);
}

function finishProgress(message) {
  stopProgress();
  updateProgress(100, message);
}

function stopProgress() {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
}

function updateElapsedTime() {
  if (!progressElapsed) {
    return;
  }

  if (!processStartedAt) {
    progressElapsed.textContent = "\uacbd\uacfc \uc2dc\uac04 0\ucd08";
    return;
  }

  const elapsedSeconds = Math.max(Math.round((Date.now() - processStartedAt) / 1000), 0);
  progressElapsed.textContent = `\uacbd\uacfc \uc2dc\uac04 ${formatDuration(elapsedSeconds)}`;
}

function updateProgress(percent, message) {
  displayedProgress = Math.max(displayedProgress, Number(percent) || 0);
  const visiblePercent = Math.min(displayedProgress, 100);
  const phase = Math.min(Math.floor(visiblePercent / 25), 3);
  const progressHeadline = buildProgressHeadline(visiblePercent, message, phase);
  renderSteps(phase);
  progressBar.style.width = `${visiblePercent}%`;
  progressPct.textContent = `${visiblePercent}%`;
  progressLabel.textContent = progressHeadline;
  progressEta.textContent = buildProgressDetail(visiblePercent, message);
}

function buildProgressHeadline(percent, message, phase) {
  if (percent >= 100) {
    return "\uc644\ub8cc";
  }

  const stageMessage = extractStageMessage(message);
  if (stageMessage) {
    return stageMessage;
  }

  return `${stepLabels[phase]}...`;
}

function extractStageMessage(message) {
  if (!message) {
    return "";
  }

  const normalizedMessage = String(message).trim();
  if (!normalizedMessage) {
    return "";
  }

  const koreanStage = normalizedMessage.match(/^(.+?\.\.\.)/);
  if (koreanStage) {
    return koreanStage[1];
  }

  const englishStageMap = [
    ["Analyzing upright angles", "\uc601\uc0c1 \ubd84\uc11d \uc911..."],
    ["Calculating auto-crop", "\uc790\ub3d9 \ud06c\ub86d \uacc4\uc0b0 \uc911..."],
    ["Rendering upright result", "\uacb0\uacfc \uc601\uc0c1 \ub80c\ub354\ub9c1 \uc911..."],
    ["Starting superresolution processing", "\ucd08\ud574\uc0c1\ub3c4 \ucc98\ub9ac \uc2dc\uc791..."],
    ["Superresolution processing completed", "\ucd08\ud574\uc0c1\ub3c4 \ucc98\ub9ac \uc644\ub8cc..."],
  ];
  const matchedStage = englishStageMap.find(([source]) => normalizedMessage.startsWith(source));
  if (matchedStage) {
    return matchedStage[1];
  }

  return normalizedMessage;
}

function buildProgressDetail(percent, message) {
  if (percent >= 100) {
    return message || "\uc644\ub8cc\ub418\uc5c8\uc2b5\ub2c8\ub2e4.";
  }

  if (!processStartedAt || percent < 5) {
    return message || "\ucc98\ub9ac \uc900\ube44 \uc911\uc785\ub2c8\ub2e4.";
  }

  const elapsedSeconds = Math.max((Date.now() - processStartedAt) / 1000, 1);
  const totalSeconds = Math.round(elapsedSeconds / (percent / 100));
  const timeText = `\uc608\uc0c1 \ucd1d \uc18c\uc694 ${formatDuration(totalSeconds)}`;
  return message ? `${message} ${timeText}` : timeText;
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(Math.round(totalSeconds), 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes <= 0) {
    return `${remainder}\ucd08`;
  }
  return `${minutes}\ubd84 ${remainder}\ucd08`;
}

function renderSteps(doneThrough) {
  steps.replaceChildren();
  for (let index = 0; index < 4; index += 1) {
    const item = document.createElement("span");
    if (index <= doneThrough) {
      item.classList.add("done");
    }
    steps.appendChild(item);
  }
}

function setBusy(isBusy) {
  processButton.disabled = isBusy;
  processButton.textContent = isBusy ? "처리 중..." : "복원 시작";
}

function setUiState(nextState) {
  statePanel.dataset.state = nextState;
  stateViews.forEach((view) => {
    const isActive = view.dataset.view === nextState;
    view.classList.toggle("hidden", !isActive);
    view.classList.toggle("is-active", isActive);
  });
}

function showAlert(message) {
  alertBox.textContent = message;
  alertBox.classList.remove("hidden");
}

function hideAlert() {
  alertBox.classList.add("hidden");
  alertBox.textContent = "";
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || "요청에 실패했습니다.";
  } catch {
    return "요청에 실패했습니다.";
  }
}

