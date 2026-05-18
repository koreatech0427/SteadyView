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
const steps = document.querySelector("#steps");
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

const featureOrder = ["Superresolution", "Stabilization", "Upright Correction"];
/*
const displayStepLabels = ["업로드 준비", "보정 처리", "초해상도 처리", "결과 정리"];
const stepLabels = ["영상 분석", "프레임 추출", "AI 모델 적용", "영상 보정"];

*/
const stepLabels = [
  "\uc5c5\ub85c\ub4dc \uc900\ube44",
  "\ubcf4\uc815 \ucc98\ub9ac",
  "\ucd08\ud574\uc0c1\ub3c4 \ucc98\ub9ac",
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

  const formData = new FormData();
  formData.append("option", option);
  formData.append("file", selectedFile);

  setUiState("processing");
  setBusy(true);
  processStartedAt = Date.now();
  startProgress();

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const job = await response.json();
    activeJobId = job.id;
    updateProgress(job.progress || 0, job.message);
    pollJob(activeJobId);
  } catch (error) {
    stopPolling();
    stopProgress();
    setUiState("input");
    setBusy(false);
    showAlert(error.message);
  }
});

resetButton.addEventListener("click", resetAll);

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

      if (job.status === "done") {
        stopPolling();
        finishProgress(job.message);
        await loadJobResult(job.id);
        setBusy(false);
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
      showAlert(error.message);
    }
  };

  await check();
  if (activeJobId === jobId) {
    pollingTimer = window.setInterval(check, 1500);
  }
}

async function loadJobResult(jobId) {
  const resultUrl = `/api/jobs/${jobId}/result`;
  const response = await fetch(resultUrl, { method: "HEAD", cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response));
  }

  setResult(resultUrl, selectedFile.name);
  activeJobId = null;
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
  updateProgress(0);
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

function updateProgress(percent, message) {
  displayedProgress = Math.max(displayedProgress, Number(percent) || 0);
  const visiblePercent = Math.min(displayedProgress, 100);
  const phase = Math.min(Math.floor(visiblePercent / 25), 3);
  renderSteps(phase);
  progressBar.style.width = `${visiblePercent}%`;
  progressPct.textContent = `${visiblePercent}%`;
  progressLabel.textContent = visiblePercent === 100 ? "\uc644\ub8cc" : `${stepLabels[phase]}...`;
  progressEta.textContent = buildProgressDetail(visiblePercent, message);
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
  const remainingSeconds = Math.max(totalSeconds - Math.round(elapsedSeconds), 0);
  const timeText = `\uc608\uc0c1 \ucd1d \uc18c\uc694 ${formatDuration(totalSeconds)} · \ub0a8\uc740 \uc2dc\uac04 ${formatDuration(remainingSeconds)}`;
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

