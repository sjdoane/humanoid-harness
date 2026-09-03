const views = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");

function showView(id) {
  views.forEach((view) => view.classList.toggle("is-visible", view.id === id));
  navItems.forEach((item) => item.classList.toggle("is-active", item.dataset.view === id));
  const heading = document.querySelector(`#${id} h1`);
  if (heading) heading.focus?.({ preventScroll: true });
}

navItems.forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));

async function loadJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

async function loadStatus() {
  try {
    const status = await loadJson("/api/status");
    document.querySelector("#progress").textContent = status.implemented_capability.join("; ");
    document.querySelector("#bottleneck").textContent = status.current_bottleneck;
    document.querySelector("#next-step").textContent = status.next_gate;
    document.querySelector("#evidence-class").textContent = status.evidence_class;
  } catch (error) {
    document.querySelector("#progress").textContent = "Status endpoint unavailable.";
    document.querySelector("#bottleneck").textContent = error.message;
    document.querySelector("#next-step").textContent = "Run humanoid-harness doctor.";
  }
}

async function loadGraphStats() {
  const graphState = document.querySelector("#graph-state");
  try {
    const stats = await loadJson("/api/research/stats");
    if (stats.state !== "built") {
      graphState.textContent = "Index not built. Run: humanoid-harness research build";
      return;
    }
    document.querySelector("#paper-count").textContent = stats.papers;
    document.querySelector("#mechanism-count").textContent = stats.mechanisms;
    document.querySelector("#failure-count").textContent = stats.failure_modes;
    graphState.textContent = `${stats.evidence_items} source-located evidence records · schema v${stats.schema_version}`;
  } catch (error) {
    graphState.textContent = error.message;
  }
}

function conditionLabel(value) {
  return {
    nominal: "Nominal",
    lateral_velocity: "Lateral +1 m/s",
    pitch_velocity_falsifier: "Pitch +1 rad/s",
  }[value] || value;
}

function appendComparison(row) {
  const root = document.querySelector("#comparison-body");
  const tableRow = document.createElement("tr");
  const condition = document.createElement("th");
  condition.scope = "row";
  condition.textContent = conditionLabel(row.condition);
  const baseline = document.createElement("td");
  baseline.textContent = row.baseline_no_collapse + "/" + row.n;
  const candidate = document.createElement("td");
  candidate.textContent = row.candidate_no_collapse + "/" + row.n;
  const delta = document.createElement("td");
  delta.textContent = (row.delta >= 0 ? "+" : "") + row.delta;
  const check = document.createElement("td");
  const label = document.createElement("span");
  if (row.passed === true) {
    label.className = "check-label check-pass";
    label.textContent = "met";
  } else if (row.passed === false) {
    label.className = "check-label check-fail";
    label.textContent = "missed";
  } else {
    label.className = "check-label check-neutral";
    label.textContent = "descriptive";
  }
  const expectation = document.createElement("small");
  expectation.textContent = row.expectation;
  check.append(label, expectation);
  tableRow.append(condition, baseline, candidate, delta, check);
  root.append(tableRow);
}

async function loadLocalExploration() {
  const badge = document.querySelector("#exploration-badge");
  try {
    const evidence = await loadJson("/api/exploration/latest");
    if (evidence.state !== "available") {
      badge.textContent = "no admitted behavioral trace";
      badge.className = evidence.state === "rejected" ? "badge badge-blocked" : "badge badge-neutral";
      document.querySelector("#stage-detail").textContent =
        evidence.state === "rejected"
          ? "A local bundle exists but failed integrity validation."
          : "No internally reconciled local exploration is available.";
      return;
    }

    badge.textContent = "local exploration · not admitted";
    badge.className = "badge badge-warning";
    const video = document.querySelector("#exploration-video");
    video.src = evidence.media.video;
    video.poster = evidence.media.poster;
    video.hidden = false;
    document.querySelector("#empty-figure").hidden = true;
    document.querySelector("#rollout-stage").classList.add("has-video");
    document.querySelector("#stage-title").textContent =
      "Four-arm lateral-disturbance rollout";
    document.querySelector("#stage-detail").textContent =
      "Bundle-declared first seed; visible execution only, not formal oracle evidence.";
    document.querySelector("#receipt-evidence").textContent = "local exploratory";
    document.querySelector("#receipt-reference").textContent = "Tier-K · not admitted";
    document.querySelector("#receipt-seed").textContent = String(evidence.visual.seed);
    document.querySelector("#receipt-condition").textContent =
      conditionLabel(evidence.visual.condition);
    document.querySelector("#receipt-selection").textContent = evidence.visual.selection;

    document.querySelector("#exploration-results").hidden = false;
    document.querySelector("#exploration-question").textContent = evidence.question;
    document.querySelector("#comparison-body").replaceChildren();
    evidence.comparisons.forEach(appendComparison);
    document.querySelector("#phase-duty").textContent =
      evidence.diagnostics.nominal_phase_correction_percent + "%";
    document.querySelector("#fallback-duty").textContent =
      evidence.diagnostics.nominal_fallback_percent + "%";
    document.querySelector("#run-count").textContent =
      evidence.completed_runs + " / " + evidence.scheduled_runs;

    const figure = document.querySelector("#holdout-figure");
    figure.src = evidence.media.figure;
    figure.alt = evidence.alt_text;
    figure.hidden = false;
    const limitations = document.querySelector("#exploration-limitations");
    limitations.replaceChildren();
    [...evidence.not_admitted_reasons, ...evidence.audit_limitations].forEach((value) => {
      const item = document.createElement("li");
      item.textContent = value;
      limitations.append(item);
    });
    document.querySelector("#manifest-sha").textContent = evidence.receipts.manifest_sha256;
    document.querySelector("#runs-sha").textContent = evidence.receipts.runs_sha256;
    document.querySelector("#video-sha").textContent = evidence.receipts.video_sha256;
  } catch (error) {
    badge.textContent = "local evidence unavailable";
    badge.className = "badge badge-blocked";
    document.querySelector("#stage-detail").textContent = error.message;
  }
}

function renderResults(payload) {
  const root = document.querySelector("#results");
  root.replaceChildren();
  if (!payload.results.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No matching research records.";
    root.append(empty);
    return;
  }
  payload.results.forEach((result) => {
    const article = document.createElement("article");
    article.className = "result";

    const kind = document.createElement("span");
    kind.className = "result-kind";
    kind.textContent = result.kind;
    const title = document.createElement("h3");
    title.textContent = result.title;
    const description = document.createElement("p");
    description.textContent = result.description;
    article.append(kind, title, description);

    if (result.source_url) {
      const link = document.createElement("a");
      link.href = result.source_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = `Open source${result.paper_id ? ` · ${result.paper_id}` : ""}`;
      article.append(link);
    }
    root.append(article);
  });
}

document.querySelector("#research-search").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = new FormData(event.currentTarget).get("q");
  const root = document.querySelector("#results");
  root.textContent = "Searching…";
  try {
    renderResults(await loadJson(`/api/research/query?q=${encodeURIComponent(query)}`));
  } catch (error) {
    root.textContent = error.message;
  }
});

loadStatus();
loadGraphStats();
loadLocalExploration();
