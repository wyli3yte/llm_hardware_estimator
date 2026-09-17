const form = document.getElementById("estimate-form");
const statusBox = document.getElementById("status");
const decisionBox = document.getElementById("decision");
const optionsBox = document.getElementById("options");
const table = document.getElementById("gpu-table");
const download = document.getElementById("download");

function fillSelect(id, values, selected) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === selected) option.selected = true;
    select.appendChild(option);
  });
}

function formatPercent(value) {
  if (value === null || value === undefined) return "";
  return `${Math.round(value * 100)}%`;
}

async function loadOptions() {
  const response = await fetch("/api/options");
  const options = await response.json();
  fillSelect("model", options.models, "qwen3-32b");
  fillSelect("scenario", options.scenarios, "rag");
  fillSelect("weight_precision", options.precisions, "bf16");
  fillSelect("goal", options.goals, "balanced");
  fillSelect("target", options.targets, "domestic");
}

function renderResult(result) {
  statusBox.textContent = "已生成推荐方案。";
  decisionBox.innerHTML = `
    <strong>${result.decision.summary}</strong>
    <span>生产建议显存：${result.decision.memory}</span>
    <span>主要风险：${result.decision.risk}</span>
  `;

  optionsBox.className = "option-grid";
  optionsBox.innerHTML = result.options.map((item) => `
    <div class="option">
      <b>${item.name}</b>
      <span>${item.gpu_count} × ${item.gpu_model}</span><br />
      <span>${item.note}</span>
    </div>
  `).join("");

  table.innerHTML = `
    <thead>
      <tr>
        <th>排名</th>
        <th>等级</th>
        <th>厂商</th>
        <th>GPU型号</th>
        <th>是否满足</th>
        <th>推荐卡数</th>
        <th>显存余量</th>
        <th>主要瓶颈</th>
        <th>多卡风险</th>
        <th>采购建议</th>
      </tr>
    </thead>
    <tbody>
      ${result.gpu_candidates.map((item) => `
        <tr>
          <td>${item.rank}</td>
          <td>${item.level}</td>
          <td>${item.vendor}</td>
          <td>${item.gpu_model}</td>
          <td>${item.fit_status}</td>
          <td>${item.suggested_gpu_count}</td>
          <td>${formatPercent(item.memory_margin_ratio)}</td>
          <td>${item.main_bottleneck}</td>
          <td>${item.multi_gpu_risk}</td>
          <td>${item.decision_note}</td>
        </tr>
      `).join("")}
    </tbody>
  `;

  download.href = result.download_url;
  download.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  statusBox.textContent = "正在生成方案...";
  download.hidden = true;

  const payload = Object.fromEntries(new FormData(form).entries());
  payload.concurrency = Number(payload.concurrency || 4);

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "生成失败");
    renderResult(result);
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

loadOptions().catch((error) => {
  statusBox.textContent = error.message;
});
