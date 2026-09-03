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

const probe002aConditions = [
  { id: "C_exact", label: "Exact reference" },
  { id: "C_zero_input", label: "Zero input" },
  { id: "C_shuffle_input", label: "Deterministic shuffle" },
  { id: "C_shift_input", label: "+250-frame shift" },
];

function requireProbeText(value, field) {
  if (typeof value !== "string" || !value) {
    throw new Error(`Experiment 002A ${field} is invalid.`);
  }
  return value;
}

function requireProbeCount(value, field) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`Experiment 002A ${field} is invalid.`);
  }
  return value;
}

function probeCount(value, total, field) {
  return `${requireProbeCount(value, field)} / ${requireProbeCount(total, "snapshot count")}`;
}

function probeMagnitude(value, field) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`Experiment 002A ${field} is invalid.`);
  }
  if (value === 0) return "0.000000";
  if (value < 0.0001) return value.toExponential(3);
  return value.toFixed(6);
}

function appendProbe002aCondition(row, displayLabel) {
  const tableRow = document.createElement("tr");
  const label = document.createElement("th");
  label.scope = "row";
  label.textContent = displayLabel;

  const input = document.createElement("td");
  input.textContent = probeCount(
    row.policy_input_changed_count_vs_exact,
    row.snapshot_count,
    "policy-input count",
  );
  const actor = document.createElement("td");
  actor.textContent = probeCount(
    row.actor_output_changed_count_vs_exact,
    row.snapshot_count,
    "actor-output count",
  );
  const control = document.createElement("td");
  control.textContent = probeCount(
    row.composed_action_changed_count_vs_exact,
    row.snapshot_count,
    "composed-control count",
  );
  const critic = document.createElement("td");
  critic.textContent = probeCount(
    row.critic_output_changed_count_vs_exact,
    row.snapshot_count,
    "critic-output count",
  );
  const rms = document.createElement("td");
  rms.className = "probe-magnitude";
  rms.textContent = probeMagnitude(
    row.composed_action_rms_delta_vs_exact,
    "control RMS delta",
  );

  tableRow.append(label, input, actor, control, critic, rms);
  document.querySelector("#probe-002a-body").append(tableRow);
}

function renderProbe002aUnavailable(payload) {
  const badge = document.querySelector("#probe-002a-badge");
  badge.className =
    payload.state === "rejected" ? "badge badge-blocked" : "badge badge-neutral";
  badge.textContent = payload.state === "rejected" ? "receipt rejected" : "no report";
  document.querySelector("#probe-002a-state").textContent =
    payload.detail || "No locally validated Experiment 002A report is available.";
  document.querySelector("#probe-002a-results").hidden = true;
}

function renderProbe002a(payload) {
  if (payload.state !== "available") {
    renderProbe002aUnavailable(payload);
    return;
  }
  if (!Array.isArray(payload.conditions) || payload.conditions.length !== 4) {
    throw new Error("Experiment 002A must contain exactly four reference arms.");
  }
  if (
    payload.experiment_id !== "experiment_002a" ||
    payload.evidence_class !== "exploratory" ||
    payload.snapshot_count !== 8 ||
    payload.condition_count !== 4 ||
    payload.independent_unit !== "one_existing_local_checkpoint" ||
    payload.formal_oracle_comparison_authorized !== false ||
    payload.tracker_behavior_or_stability_established !== false ||
    payload.oracle_quality_established !== false
  ) {
    throw new Error("Experiment 002A claim or design boundary is invalid.");
  }
  const conditions = new Map(payload.conditions.map((row) => [row.condition_id, row]));
  if (
    conditions.size !== 4 ||
    !probe002aConditions.every((condition) => conditions.has(condition.id))
  ) {
    throw new Error("Experiment 002A reference-arm identities are incomplete.");
  }

  const badge = document.querySelector("#probe-002a-badge");
  if (payload.mechanistic_gate_passed === true) {
    badge.className = "badge badge-warning";
    badge.textContent = "passed · exploratory";
  } else {
    badge.className = "badge badge-blocked";
    badge.textContent = "not passed · exploratory";
  }
  document.querySelector("#probe-002a-state").textContent =
    requireProbeText(payload.claim, "claim");
  document.querySelector("#probe-002a-snapshots").textContent = String(
    requireProbeCount(payload.snapshot_count, "snapshot count"),
  );
  document.querySelector("#probe-002a-conditions").textContent = String(
    requireProbeCount(payload.condition_count, "condition count"),
  );
  document.querySelector("#probe-002a-checkpoints").textContent =
    payload.independent_unit === "one_existing_local_checkpoint" ? "1" : "—";

  const tableBody = document.querySelector("#probe-002a-body");
  tableBody.replaceChildren();
  probe002aConditions.forEach((condition) =>
    appendProbe002aCondition(conditions.get(condition.id), condition.label),
  );

  const limitations = document.querySelector("#probe-002a-limitations");
  limitations.replaceChildren();
  if (Array.isArray(payload.limitations)) {
    payload.limitations.forEach((value) => {
      if (typeof value !== "string" || !value) return;
      const item = document.createElement("li");
      item.textContent = value;
      limitations.append(item);
    });
  }

  const receipts = payload.receipts;
  if (!receipts || typeof receipts !== "object") {
    throw new Error("Experiment 002A integrity receipt is missing.");
  }
  document.querySelector("#probe-002a-authority").textContent = requireProbeText(
    payload.authority,
    "authority",
  );
  document.querySelector("#probe-002a-report-sha").textContent = requireProbeText(
    receipts.report_sha256,
    "report SHA-256",
  );
  document.querySelector("#probe-002a-analysis-sha").textContent =
    requireProbeText(receipts.analysis_payload_sha256, "analysis SHA-256");
  document.querySelector("#probe-002a-design-sha").textContent = requireProbeText(
    receipts.design_sha256,
    "design SHA-256",
  );
  document.querySelector("#probe-002a-observations-sha").textContent =
    requireProbeText(receipts.matched_observation_set_sha256, "observation-set SHA-256");
  document.querySelector("#probe-002a-controller-sha").textContent =
    requireProbeText(receipts.residual_controller_content_sha256, "checkpoint SHA-256");
  document.querySelector("#probe-002a-projection-sha").textContent =
    requireProbeText(receipts.source_projection_receipt_sha256, "projection SHA-256");
  document.querySelector("#probe-002a-license").textContent = requireProbeText(
    payload.dataset_license_status,
    "dataset status",
  );
  document.querySelector("#probe-002a-next-gate").textContent = requireProbeText(
    payload.next_gate,
    "next gate",
  );
  document.querySelector("#probe-002a-results").hidden = false;
}

async function loadProbe002a() {
  try {
    renderProbe002a(await loadJson("/api/experiments/002a"));
  } catch (error) {
    renderProbe002aUnavailable({ state: "rejected", detail: error.message });
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
loadProbe002a();
