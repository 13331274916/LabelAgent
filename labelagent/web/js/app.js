/* LabelAgent 前端逻辑：五个模块（标注 / 清洗 / 数据集 / 环境 / 训练与消融） */
"use strict";

/* ------------------------------------------------------------------ */
/* 基础工具                                                            */
/* ------------------------------------------------------------------ */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && typeof opts.body !== "string") {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({ ok: false, error: `HTTP ${resp.status}` }));
  if (!resp.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data.data;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function downloadFile(path, filename) {
  const a = document.createElement("a");
  a.href = `/api/annotation/download?path=${encodeURIComponent(path)}`;
  a.download = filename || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ------------------------------------------------------------------ */
/* Tab 切换                                                            */
/* ------------------------------------------------------------------ */
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("#panel-" + tab.dataset.tab).classList.add("active");
    refreshOverview();
  });
});

async function refreshOverview() {
  try {
    const d = await api("/api/overview");
    $("#overview").textContent = `图像 ${d.images} · 标注 ${d.annotations} · 类别 ${d.classes.length}`;
  } catch (e) { /* ignore */ }
}

/* ================================================================== */
/* 🏷️ 标注模块                                                        */
/* ================================================================== */
let currentImageId = null;
let agentPollTimer = null;

async function loadProviders() {
  const providers = await api("/api/annotation/providers");
  const sel = $("#ann-provider");
  sel.innerHTML = "";
  providers.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.name}${p.available ? "" : "（不可用：" + p.message + "）"}`;
    sel.appendChild(opt);
  });
  sel.onchange = () => {
    const p = providers.find((x) => x.id === sel.value);
    $("#ann-apikey-row").classList.toggle("hidden", !(p && p.needs_api_key));
  };
}

async function loadGallery() {
  const images = await api("/api/annotation/images");
  const g = $("#gallery");
  g.innerHTML = "";
  images.forEach((it) => {
    const div = document.createElement("div");
    div.className = "thumb" + (it.id === currentImageId ? " active" : "");
    div.innerHTML = `
      <img src="/api/annotation/image/${it.id}/file" alt="">
      <span class="badge">${it.annotation_count}</span>
      ${it.is_blurry ? '<span class="warn">模糊</span>' : ""}
      ${it.duplicate_of ? '<span class="warn">重复</span>' : ""}
      <div class="cap">${escapeHtml(it.filename)}</div>`;
    div.onclick = () => selectImage(it.id);
    g.appendChild(div);
  });
  if (!currentImageId && images.length) selectImage(images[0].id);
  else if (currentImageId) showImage(currentImageId);
  else $("#ann-list").innerHTML = '<div class="muted">尚未导入图像</div>';
}

async function selectImage(id) {
  currentImageId = id;
  await showImage(id);
  await loadGallery();
}

async function showImage(id) {
  const item = await api(`/api/annotation/image/${id}`);
  const img = new Image();
  img.onload = () => {
    const canvas = $("#viewer");
    const maxW = canvas.parentElement.clientWidth - 4;
    const scale = Math.min(1, maxW / img.naturalWidth);
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const colors = ["#ff5c5c", "#4fc3f7", "#81c784", "#ffb74d", "#ba68c8", "#fff176"];
    item.annotations.forEach((ann, i) => {
      const c = colors[i % colors.length];
      ctx.strokeStyle = c;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ann.points.forEach((p, k) => {
        const x = p.x * scale, y = p.y * scale;
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = c;
      ctx.font = "bold 13px sans-serif";
      ctx.fillText(ann.label, ann.points[0]?.x * scale + 2, Math.max(14, ann.points[0]?.y * scale - 4));
    });
  };
  img.src = `/api/annotation/image/${id}/file`;
  img.onerror = () => { /* 文件不可读时忽略 */ };

  // 标注列表（修订）
  const list = $("#ann-list");
  list.innerHTML = `<div class="muted">${escapeHtml(item.filename)} · ${item.annotations.length} 个标注</div>`;
  item.annotations.forEach((ann) => {
    const row = document.createElement("div");
    row.className = "ann-item";
    row.innerHTML = `
      <span style="color:${["#ff5c5c", "#4fc3f7", "#81c784", "#ffb74d", "#ba68c8"][item.annotations.indexOf(ann) % 5]}">■</span>
      <input value="${escapeHtml(ann.label)}" data-ann="${ann.id}">
      <span class="muted">${ann.confidence ? (ann.confidence * 100).toFixed(0) + "%" : ""}</span>
      <button class="del" data-del="${ann.id}" title="删除标注">✕</button>`;
    row.querySelector("input").onchange = async (e) => {
      try {
        await api(`/api/annotation/image/${item.id}/annotations/${ann.id}`, {
          method: "PATCH", body: { label: e.target.value },
        });
        await showImage(item.id);
        await loadGallery();
      } catch (err) { alert(err.message); }
    };
    row.querySelector(".del").onclick = async () => {
      await api(`/api/annotation/image/${item.id}/annotations/${ann.id}`, { method: "DELETE" });
      await showImage(item.id);
      await loadGallery();
    };
    list.appendChild(row);
  });
}

$("#btn-prev").onclick = async () => {
  const images = await api("/api/annotation/images");
  const idx = images.findIndex((i) => i.id === currentImageId);
  if (idx > 0) await selectImage(images[idx - 1].id);
};
$("#btn-next").onclick = async () => {
  const images = await api("/api/annotation/images");
  const idx = images.findIndex((i) => i.id === currentImageId);
  if (idx >= 0 && idx < images.length - 1) await selectImage(images[idx + 1].id);
};
$("#btn-refresh").onclick = async () => {
  await loadGallery();
  if (currentImageId) await showImage(currentImageId);
};

$("#btn-import-folder").onclick = async () => {
  const folder = $("#ann-folder").value.trim();
  if (!folder) return alert("请输入文件夹路径");
  const r = await api("/api/annotation/import-folder", { method: "POST", body: { folder } });
  $("#import-msg").textContent = `导入成功：${r.imported} 张（项目总计 ${r.total} 张）`;
  await loadGallery();
  await refreshOverview();
};

$("#ann-upload").onchange = async () => {
  const fd = new FormData();
  [...$("#ann-upload").files].forEach((f) => fd.append("files", f));
  const r = await api("/api/annotation/upload", { method: "POST", body: fd });
  $("#import-msg").textContent = `上传导入：${r.imported} 张（项目总计 ${r.total} 张）`;
  await loadGallery();
  await refreshOverview();
};

$("#btn-demo").onclick = async () => {
  const count = parseInt($("#demo-count").value, 10) || 12;
  const r = await api("/api/demo/generate", { method: "POST", body: { count } });
  $("#import-msg").textContent = `已生成 ${r.generated} 张演示图像（项目总计 ${r.total} 张）`;
  await loadGallery();
  await refreshOverview();
};

$("#btn-agent").onclick = async () => {
  const body = {
    provider_id: $("#ann-provider").value,
    prompt: $("#ann-prompt").value,
    label: $("#ann-label").value || "scratch_defect",
    max_objects: parseInt($("#ann-max-objects").value, 10) || 5,
    api_key: $("#ann-apikey").value || null,
  };
  try {
    await api("/api/annotation/agent/start", { method: "POST", body });
    $("#btn-agent").disabled = true;
    $("#btn-agent-stop").classList.remove("hidden");
    pollAgentStatus();
  } catch (e) { alert(e.message); }
};

$("#btn-agent-stop").onclick = async () => {
  // 停止：当前实现通过刷新状态终止轮询（Agent 任务不可中断，任务粒度小）
  $("#btn-agent-stop").classList.add("hidden");
};

async function pollAgentStatus() {
  const s = await api("/api/annotation/agent/status");
  $("#agent-status").textContent =
    `进度 ${s.done}/${s.total}${s.current_image ? " · " + s.current_image : ""} · ${(s.log[s.log.length - 1] || "")}`;
  if (!s.running) {
    $("#btn-agent").disabled = false;
    $("#btn-agent-stop").classList.add("hidden");
    $("#agent-status").textContent += "（完成）";
    await loadGallery();
    if (currentImageId) await showImage(currentImageId);
    return;
  }
  setTimeout(pollAgentStatus, 800);
}

$("#btn-export-single").onclick = async () => {
  if (!currentImageId) return alert("请先选择一张图像");
  const r = await api(`/api/annotation/image/${currentImageId}/export`, {
    method: "POST", body: { format: $("#export-format").value },
  });
  $("#export-msg").innerHTML = `已导出：<a href="/api/annotation/download?path=${encodeURIComponent(r.path)}" download="${r.filename}">${escapeHtml(r.filename)}</a>`;
};

$("#btn-export-zip").onclick = async () => {
  const r = await api("/api/annotation/export-zip", {
    method: "POST", body: { formats: ["labelme", "voc", "yolo", "coco", "csv"] },
  });
  $("#export-msg").innerHTML = `打包完成（${r.filename}）：<a href="/api/annotation/download?path=${encodeURIComponent(r.path)}" download="${r.filename}">下载 ZIP</a>`;
};

/* ================================================================== */
/* 🧹 清洗模块                                                        */
/* ================================================================== */
$("#btn-diagnose").onclick = async () => {
  const r = await api("/api/cleaning/diagnose", {
    method: "POST",
    body: { blur_threshold: parseFloat($("#clean-blur-thresh").value) || 100 },
  });
  const box = $("#diagnose-result");
  box.innerHTML = `
    <div class="score" style="color:${r.health_score > 80 ? "var(--ok)" : r.health_score > 50 ? "var(--warn)" : "var(--danger)"}">${r.health_score}</div>
    <div class="muted">健康得分 / 100 · 耗时 ${r.elapsed_ms} ms · ${r.scanned} 张</div>
    <div class="items">
      <div class="item"><b>${r.blurry_count}</b>模糊图</div>
      <div class="item"><b>${r.duplicate_count}</b>重复图</div>
      <div class="item"><b>${r.oob_count}</b>越界标注</div>
      <div class="item"><b>${r.empty_count}</b>空图</div>
      <div class="item"><b>${r.anomalies}</b>异常项</div>
      <div class="item"><b>${r.scanned}</b>已扫描</div>
    </div>`;
  await loadGallery();
};

$("#btn-clean").onclick = async () => {
  const r = await api("/api/cleaning/auto-clean", {
    method: "POST",
    body: {
      remove_duplicates: $("#clean-dup").checked,
      fix_oob: $("#clean-oob").checked,
      remove_blurry: $("#clean-blurry").checked,
      remove_empty: $("#clean-empty").checked,
      duplicate_method: $("#clean-method").value,
    },
  });
  $("#clean-result").textContent =
    `清洗完成：保留 ${r.kept} 张，移除 ${r.removed} 张，已修复越界标注 ${r.fixed_oob} 条`;
  await loadGallery();
  await refreshOverview();
};

$("#btn-cache-clear").onclick = async () => {
  const r = await api("/api/cleaning/cache", { method: "DELETE" });
  alert(`已清空缓存 ${r.cleared} 条`);
  refreshCacheInfo();
};

async function refreshCacheInfo() {
  try {
    const c = await api("/api/cleaning/cache");
    $("#cache-info").textContent = `缓存：${c.entries} 条（${c.file}）`;
  } catch (e) { /* ignore */ }
}

/* ================================================================== */
/* 📂 数据集模块                                                      */
/* ================================================================== */
$("#btn-stats").onclick = async () => {
  const s = await api("/api/dataset/stats");
  const tb = $("#stats-table tbody");
  tb.innerHTML = "";
  Object.entries(s.per_class).forEach(([cls, n]) => {
    tb.insertAdjacentHTML("beforeend", `<tr><td>${escapeHtml(cls)}</td><td>${n}</td></tr>`);
  });
  $("#stats-table").insertAdjacentHTML("beforeend",
    `<tr><td><b>合计</b></td><td><b>${s.annotation_count}</b> 标注 / ${s.image_count} 图</td></tr>`);
};

$("#ds-ratio").oninput = (e) => { $("#ds-ratio-label").textContent = parseFloat(e.target.value).toFixed(2); };

$("#btn-split").onclick = async () => {
  const r = await api("/api/dataset/split", {
    method: "POST",
    body: {
      train_ratio: parseFloat($("#ds-ratio").value),
      dataset_name: $("#ds-name").value.trim() || "defect_dataset_split_v1",
    },
  });
  $("#split-result").innerHTML = `
    数据集 <b>${escapeHtml(r.dataset_name)}</b> · Train ${r.train_count} 张 / Val ${r.val_count} 张<br>
    dataset.yaml：<a href="/api/annotation/download?path=${encodeURIComponent(r.yaml_path)}" download="dataset.yaml">下载</a>
    <br><span class="muted">${escapeHtml(r.yaml_path)}</span>`;
  refreshDatasetList();
};

async function refreshDatasetList() {
  try {
    const list = await api("/api/dataset/datasets");
    $("#ds-list").innerHTML = list.length
      ? list.map((d) => `<div>📁 ${escapeHtml(d.name)} ${d.has_yaml ? "（含 dataset.yaml）" : ""}</div>`).join("")
      : "尚未生成数据集";
  } catch (e) { $("#ds-list").textContent = "—"; }
}

/* ================================================================== */
/* 🐍 环境模块                                                        */
/* ================================================================== */
$("#btn-detect-env").onclick = async () => {
  const envs = await api("/api/environment/detect");
  const box = $("#env-list");
  box.innerHTML = "";
  envs.forEach((e) => {
    const div = document.createElement("div");
    div.className = "env-item";
    div.innerHTML = `<span class="p">${escapeHtml(e.path)}</span><span class="v">${escapeHtml(e.version)} · ${e.package_count} 包</span>`;
    div.onclick = async () => {
      $("#env-path").value = e.path;
      await loadPackages(e.path);
    };
    box.appendChild(div);
  });
};

$("#btn-import-env").onclick = async () => {
  const path = $("#env-path").value.trim();
  if (!path) return alert("请输入 Python 可执行文件路径");
  const r = await api("/api/environment/import", { method: "POST", body: { path } });
  alert(`导入成功：Python ${r.version}，${r.package_count} 个依赖包`);
  $("#tr-python").value = path;
};

$("#btn-env-packages").onclick = async () => {
  const path = $("#env-path").value.trim();
  if (!path) return alert("请输入 Python 可执行文件路径");
  await loadPackages(path);
};

async function loadPackages(path) {
  const pkgs = await api("/api/environment/packages", { method: "POST", body: { path } });
  const tb = $("#pkg-table tbody");
  tb.innerHTML = "";
  pkgs.forEach((p) => {
    tb.insertAdjacentHTML("beforeend",
      `<tr><td>${escapeHtml(p.name)}</td><td>${escapeHtml(p.version)}</td><td>${p.required ? "⚠️ 训练相关" : "—"}</td></tr>`);
  });
  if (!pkgs.length) tb.innerHTML = '<tr><td colspan="3" class="muted">未读取到依赖包</td></tr>';
}

/* ================================================================== */
/* 🚀 训练与消融模块                                                  */
/* ================================================================== */
let trainPollTimer = null;

async function loadTrainingOptions() {
  const o = await api("/api/training/options");
  fillSelect("#tr-arch", o.model_archs);
  fillSelect("#tr-loss", o.loss_functions);
  fillSelect("#tr-optimizer", o.optimizers);
  fillSelect("#tr-scheduler", o.schedulers);
  $("#tr-strategies").innerHTML = Object.entries(o.strategies).map(([k, v]) =>
    `<label class="check"><input type="checkbox" data-strategy="${k}"> ${v}</label>`).join("");
  // 默认开启若干策略（与桌面版一致）
  ["warmup", "ema", "amp", "mosaic"].forEach((k) => {
    const cb = document.querySelector(`[data-strategy="${k}"]`);
    if (cb) cb.checked = true;
  });
}

function fillSelect(sel, items) {
  const el = $(sel);
  el.innerHTML = "";
  items.forEach((it) => {
    const opt = document.createElement("option");
    opt.value = it; opt.textContent = it;
    el.appendChild(opt);
  });
}

function readTrainingConfig() {
  const strategies = {};
  $$("[data-strategy]").forEach((cb) => { strategies[cb.dataset.strategy] = cb.checked; });
  return {
    model_arch: $("#tr-arch").value,
    loss: $("#tr-loss").value,
    optimizer: $("#tr-optimizer").value,
    scheduler: $("#tr-scheduler").value,
    epochs: parseInt($("#tr-epochs").value, 10) || 12,
    batch: parseInt($("#tr-batch").value, 10) || 8,
    img_size: parseInt($("#tr-imgsize").value, 10) || 640,
    lr: parseFloat($("#tr-lr").value) || 0.01,
    two_stage: $("#tr-twostage").checked,
    stage1_epochs: parseInt($("#tr-stage1").value, 10) || 8,
    stage2_epochs: parseInt($("#tr-stage2").value, 10) || 4,
    strategies,
  };
}

$("#btn-train-start").onclick = async () => {
  const config = readTrainingConfig();
  const mode = $("#tr-mode").value;
  if (mode === "external") config.python_exe = $("#tr-python").value.trim() || null;
  try {
    await api("/api/training/start", {
      method: "POST",
      body: {
        config,
        mode,
        dataset_yaml: $("#tr-dataset-yaml").value.trim() || null,
      },
    });
    $("#btn-train-start").disabled = true;
    $("#btn-train-stop").classList.remove("hidden");
    pollTrainingStatus();
  } catch (e) { alert(e.message); }
};

$("#btn-train-stop").onclick = async () => {
  await api("/api/training/stop", { method: "POST" });
};

async function pollTrainingStatus() {
  const s = await api("/api/training/status");
  if (!s) { $("#btn-train-start").disabled = false; return; }
  $("#train-progress").style.width = s.progress + "%";
  $("#train-progress-text").textContent = s.progress + "%";
  $("#train-epoch").textContent = `Epoch ${s.current_epoch} / ${s.total_epochs}`;
  $("#train-stage").textContent = s.stage ? `Stage ${s.stage}` : "Stage -";
  $("#train-console").textContent = s.log.slice(-60).join("\n");
  drawLossChart(s.loss_history);
  if (s.artifacts && Object.keys(s.artifacts).length) {
    $("#train-artifacts").innerHTML = "产物：" + Object.entries(s.artifacts).map(([n, p]) =>
      `<a href="/api/training/download?path=${encodeURIComponent(p)}" download="${n}">${n}</a>`).join(" · ");
  }
  if (s.status === "finished" || s.status === "failed" || s.status === "stopped") {
    $("#btn-train-start").disabled = false;
    $("#btn-train-stop").classList.add("hidden");
    if (s.status === "finished") refreshAblation();
    return;
  }
  setTimeout(pollTrainingStatus, 700);
}

function drawLossChart(losses) {
  const canvas = $("#loss-chart");
  if (!losses || losses.length < 2) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#2c3754"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    ctx.beginPath(); ctx.moveTo(0, (h - 20) * i / 4 + 10); ctx.lineTo(w, (h - 20) * i / 4 + 10); ctx.stroke();
  }
  const maxLoss = Math.max(...losses) * 1.05, minLoss = Math.min(...losses) * 0.8;
  const range = Math.max(1e-6, maxLoss - minLoss);
  ctx.strokeStyle = "#4f8cff"; ctx.lineWidth = 2; ctx.beginPath();
  losses.forEach((v, i) => {
    const x = (i / (losses.length - 1)) * w;
    const y = h - 10 - ((v - minLoss) / range) * (h - 30);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8fa0bf"; ctx.font = "11px sans-serif";
  ctx.fillText(`Loss: ${losses[losses.length - 1].toFixed(4)}（min ${minLoss.toFixed(4)}）`, 8, 18);
}

/* ---- 消融实验 ---- */
async function refreshAblation() {
  const st = await api("/api/training/ablation/status");
  renderAblationGroups(st.groups);
  renderAblationResults(st.results);
}

function renderAblationGroups(groups) {
  const tb = $("#abl-table tbody");
  tb.innerHTML = "";
  groups.forEach((g) => {
    tb.insertAdjacentHTML("beforeend", `
      <tr>
        <td>${escapeHtml(g.name)}</td><td>${escapeHtml(g.model_arch)}</td>
        <td>${g.warmup ? "✓" : "✗"}</td><td>${g.ema ? "✓" : "✗"}</td><td>${g.mosaic ? "✓" : "✗"}</td>
        <td><button data-del-group="${g.id}" class="danger">删除</button></td>
      </tr>`);
  });
  $$("[data-del-group]").forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/training/ablation/groups/${btn.dataset.delGroup}`, { method: "DELETE" });
      refreshAblation();
    };
  });
}

function renderAblationResults(results) {
  const tb = $("#abl-result-table tbody");
  const table = $("#abl-result-table");
  if (!results.length) { table.classList.add("hidden"); return; }
  table.classList.remove("hidden");
  tb.innerHTML = "";
  results.forEach((r) => {
    tb.insertAdjacentHTML("beforeend", `
      <tr>
        <td>${escapeHtml(r.name)}</td><td>${escapeHtml(r.model_arch)}</td>
        <td>${r.val_loss}</td><td><b>${r.mAP50}</b></td><td>${r.precision}</td>
        <td>${r.relative_improvement != null ? r.relative_improvement + "%" : "—"}</td>
        <td>${r.is_best ? "🏆 最佳" : ""}</td>
      </tr>`);
  });
  $("#abl-result").textContent = results.find((r) => r.is_best)
    ? `最佳实验：${results.find((r) => r.is_best).name}（mAP@0.5 = ${results.find((r) => r.is_best).mAP50}）`
    : "";
}

$("#btn-abl-add").onclick = async () => {
  const body = {
    name: $("#abl-name").value.trim() || "experiment",
    model_arch: $("#abl-arch").value,
    warmup: $("#abl-warmup").checked,
    ema: $("#abl-ema").checked,
    mosaic: $("#abl-mosaic").checked,
  };
  try {
    await api("/api/training/ablation/groups", { method: "POST", body });
    await refreshAblation();
  } catch (e) { alert(e.message); }
};

$("#btn-abl-run").onclick = async () => {
  try {
    await api("/api/training/ablation/run", {
      method: "POST",
      body: { mode: $("#tr-mode").value, dataset_yaml: $("#tr-dataset-yaml").value.trim() || null },
    });
    const poll = setInterval(async () => {
      const st = await api("/api/training/ablation/status");
      renderAblationResults(st.results);
      if (!st.running) { clearInterval(poll); renderAblationResults(st.results); }
    }, 900);
  } catch (e) { alert(e.message); }
};

$("#btn-abl-import").onclick = async () => {
  const files = $("#abl-csv").files;
  if (!files.length) return alert("请选择 results.csv 文件（可多选）");
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  fd.append("base_name", "external");
  const r = await api("/api/training/ablation/import-csv", { method: "POST", body: fd });
  await refreshAblation();
  alert(`导入成功：${r.imported} 个实验结果`);
};

$("#btn-abl-summary").onclick = async (e) => {
  e.preventDefault();
  const resp = await fetch("/api/training/ablation/summary.csv");
  const blob = await resp.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ablation_summary.csv";
  a.click();
  URL.revokeObjectURL(a.href);
};

$("#btn-abl-plot").onclick = async (e) => {
  e.preventDefault();
  const img = $("#abl-plot-img");
  img.src = `/api/training/ablation/plot.png?t=${Date.now()}`;
  img.classList.remove("hidden");
};

/* ================================================================== */
/* 初始化                                                              */
/* ================================================================== */
(async function init() {
  await loadProviders();
  await loadTrainingOptions();
  await loadGallery();
  await refreshOverview();
  await refreshCacheInfo();
  await refreshDatasetList();
  await refreshAblation();
  setInterval(refreshOverview, 5000);
})();
