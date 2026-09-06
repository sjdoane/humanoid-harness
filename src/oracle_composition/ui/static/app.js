const views = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");
let g1LearningLoaded = false;
let g1LearningLoading = false;

function showView(id) {
  views.forEach((view) => view.classList.toggle("is-visible", view.id === id));
  navItems.forEach((item) => item.classList.toggle("is-active", item.dataset.view === id));
  const heading = document.querySelector(`#${id} h1`);
  if (heading) heading.focus?.({ preventScroll: true });
}

navItems.forEach((item) =>
  item.addEventListener("click", () => {
    showView(item.dataset.view);
    if (item.dataset.view === "g1-learning" && !g1LearningLoaded) loadG1Learning();
  }),
);

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

function requireE0Text(value, field) {
  if (typeof value !== "string" || !value) {
    throw new Error(`E0 TQC ${field} is invalid.`);
  }
  return value;
}

function requireE0Count(value, field, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`E0 TQC ${field} is invalid.`);
  }
  return value;
}

function requireE0Number(value, field, minimum = 0) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(`E0 TQC ${field} is invalid.`);
  }
  return value;
}

function formatGiB(value, field) {
  return `${(requireE0Number(value, field, 1) / 1073741824).toFixed(2)} GiB`;
}

function requireE0Digest(value, field) {
  const digest = requireE0Text(value, field);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`E0 TQC ${field} is invalid.`);
  }
  return digest;
}

function renderE0TqcUnavailable(payload) {
  const badge = document.querySelector("#e0-tqc-badge");
  badge.className =
    payload.state === "rejected" ? "badge badge-blocked" : "badge badge-neutral";
  badge.textContent = payload.state === "rejected" ? "receipt rejected" : "no receipt";
  document.querySelector("#e0-tqc-state").textContent =
    payload.detail || "No locally validated E0 resource receipt is available.";
  document.querySelector("#e0-tqc-results").hidden = true;
}

function renderE0Tqc(payload) {
  if (payload.state !== "available") {
    renderE0TqcUnavailable(payload);
    return;
  }
  const expectedWorkerSeeds = [92001, 92002, 92003, 92004, 92005];
  if (
    payload.evidence_class !== "E0_resource_only" ||
    payload.gate_passed !== true ||
    payload.environment_id !== "Humanoid-v5" ||
    payload.seed !== 92001 ||
    payload.seed_role !== "permanently_excluded_disposable_calibration" ||
    payload.checkpoint_emitted !== false ||
    payload.model_disposition !== "discarded_unserialized" ||
    payload.eligible_for_controller_training !== false ||
    payload.eligible_for_behavioral_evaluation !== false ||
    !Array.isArray(payload.worker_seeds) ||
    payload.worker_seeds.length !== expectedWorkerSeeds.length ||
    !payload.worker_seeds.every((value, index) => value === expectedWorkerSeeds[index])
  ) {
    throw new Error("E0 TQC claim or design boundary is invalid.");
  }

  const steps = requireE0Count(payload.observed_environment_steps, "environment steps", 1);
  const updates = requireE0Count(payload.observed_gradient_updates, "gradient updates", 1);
  const throughput = requireE0Number(
    payload.environment_steps_per_second,
    "environment throughput",
    0,
  );
  const throughputGate = requireE0Number(
    payload.minimum_environment_steps_per_second,
    "throughput gate",
    0,
  );
  const windowCount = requireE0Count(
    payload.sustained_throughput_window_count,
    "throughput window count",
    1,
  );
  if (throughput < throughputGate) {
    throw new Error("E0 TQC throughput does not satisfy its gate.");
  }

  const badge = document.querySelector("#e0-tqc-badge");
  badge.className = "badge badge-warning";
  badge.textContent = "passed · E0 resource only";
  document.querySelector("#e0-tqc-state").textContent = requireE0Text(payload.claim, "claim");
  document.querySelector("#e0-tqc-steps").textContent = steps.toLocaleString("en-US");
  document.querySelector("#e0-tqc-updates").textContent = updates.toLocaleString("en-US");
  document.querySelector("#e0-tqc-throughput").textContent = throughput.toFixed(1);
  document.querySelector("#e0-tqc-wall").textContent = requireE0Number(
    payload.training_wall_seconds,
    "training wall time",
    0,
  ).toFixed(1);
  document.querySelector("#e0-tqc-throughput-row").textContent =
    `${throughput.toFixed(1)} steps/s`;
  document.querySelector("#e0-tqc-throughput-gate").textContent =
    `≥ ${throughputGate.toFixed(0)} steps/s`;
  document.querySelector("#e0-tqc-window-count").textContent =
    `${windowCount} × 10k-step windows passed`;
  document.querySelector("#e0-tqc-rss").textContent = formatGiB(
    payload.peak_rss_bytes,
    "sampled peak RSS",
  );
  document.querySelector("#e0-tqc-rss-gate").textContent =
    `≤ ${formatGiB(payload.sampled_peak_rss_failure_threshold_bytes, "RSS threshold")}`;
  document.querySelector("#e0-tqc-replay").textContent = formatGiB(
    payload.replay_buffer_allocation_bytes,
    "replay allocation",
  );
  document.querySelector("#e0-tqc-disk").textContent = formatGiB(
    payload.minimum_free_disk_bytes,
    "minimum free disk",
  );
  document.querySelector("#e0-tqc-disk-gate").textContent =
    `≥ ${formatGiB(payload.minimum_free_disk_gate_bytes, "free-disk gate")}`;
  document.querySelector("#e0-tqc-claim").textContent = requireE0Text(payload.claim, "claim");
  document.querySelector("#e0-tqc-next-gate").textContent = requireE0Text(
    payload.next_gate,
    "next gate",
  );

  if (!Array.isArray(payload.limitations) || payload.limitations.length < 5) {
    throw new Error("E0 TQC claim limitations are incomplete.");
  }
  const limitations = document.querySelector("#e0-tqc-limitations");
  limitations.replaceChildren();
  payload.limitations.forEach((value) => {
    const item = document.createElement("li");
    item.textContent = requireE0Text(value, "limitation");
    limitations.append(item);
  });

  const receipts = payload.receipts;
  if (!receipts || typeof receipts !== "object") {
    throw new Error("E0 TQC integrity receipt is missing.");
  }
  document.querySelector("#e0-tqc-authority").textContent = requireE0Text(
    payload.authority,
    "authority",
  );
  document.querySelector("#e0-tqc-receipt-sha").textContent = requireE0Digest(
    receipts.receipt_sha256,
    "receipt SHA-256",
  );
  document.querySelector("#e0-tqc-design-sha").textContent = requireE0Digest(
    receipts.design_sha256,
    "design SHA-256",
  );
  document.querySelector("#e0-tqc-source-sha").textContent = requireE0Digest(
    receipts.calibration_source_sha256,
    "source SHA-256",
  );
  document.querySelector("#e0-tqc-tree-sha").textContent = requireE0Digest(
    receipts.run_source_tree_sha256,
    "run source-tree SHA-256",
  );
  document.querySelector("#e0-tqc-runtime-sha").textContent = requireE0Digest(
    receipts.runtime_sha256,
    "runtime SHA-256",
  );
  document.querySelector("#e0-tqc-lock-sha").textContent = requireE0Digest(
    receipts.dependency_lock_sha256,
    "dependency-lock SHA-256",
  );
  document.querySelector("#e0-tqc-seed").textContent = String(payload.seed);
  document.querySelector("#e0-tqc-worker-seeds").textContent = payload.worker_seeds.join(", ");

  const figureWrap = document.querySelector("#e0-tqc-figure-wrap");
  if (payload.media && typeof payload.media.figure === "string") {
    document.querySelector("#e0-tqc-figure").src = payload.media.figure;
    document.querySelector("#e0-tqc-figure-sha").textContent = requireE0Digest(
      receipts.figure_sha256,
      "figure SHA-256",
    );
    figureWrap.hidden = false;
  } else {
    figureWrap.hidden = true;
  }
  document.querySelector("#e0-tqc-results").hidden = false;
}

async function loadE0Tqc() {
  try {
    renderE0Tqc(await loadJson("/api/experiments/e0-tqc"));
  } catch (error) {
    renderE0TqcUnavailable({ state: "rejected", detail: error.message });
  }
}

function g1Text(value, field) {
  if (typeof value !== "string" || !value) throw new Error(`G1 ${field} is invalid.`);
  return value;
}

function g1Count(value, field) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`G1 ${field} is invalid.`);
  }
  return value;
}

function g1Metric(value, digits = 3) {
  if (value === null) return "not observed";
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("G1 metric is invalid.");
  }
  return value.toFixed(digits);
}

function appendG1Line(root, label, value) {
  const line = document.createElement("small");
  const term = document.createElement("span");
  term.textContent = `${label}: `;
  line.append(term, document.createTextNode(value));
  root.append(line);
}

function appendG1Run(row) {
  const tableRow = document.createElement("tr");
  if (row.state === "rejected") {
    const run = document.createElement("th");
    run.scope = "row";
    run.textContent = g1Text(row.run_id, "rejected run ID");
    const detail = document.createElement("td");
    detail.colSpan = 3;
    const label = document.createElement("span");
    label.className = "check-label check-fail";
    label.textContent = "registration rejected";
    const reason = document.createElement("small");
    reason.textContent = g1Text(row.detail, "rejection detail");
    detail.append(label, reason);
    tableRow.append(run, detail);
    document.querySelector("#g1-learning-body").append(tableRow);
    return;
  }
  if (row.state !== "available") throw new Error("G1 run state is invalid.");

  const source = document.createElement("th");
  source.scope = "row";
  const runId = document.createElement("strong");
  runId.className = "run-id";
  runId.textContent = g1Text(row.run_id, "run ID");
  source.append(runId);
  appendG1Line(source, "selected", g1Text(row.selected_label, "selected label"));

  const gate = document.createElement("td");
  const passed = row.full_task_development_gate_passed;
  if (typeof passed !== "boolean") throw new Error("G1 gate result is invalid.");
  const gateLabel = document.createElement("span");
  gateLabel.className = `check-label ${passed ? "check-pass" : "check-fail"}`;
  gateLabel.textContent = passed ? "PASS" : "FAIL";
  gate.append(gateLabel);
  const passedCount = g1Count(row.development_gates.passed, "passed gate count");
  const totalCount = g1Count(row.development_gates.total, "gate count");
  appendG1Line(gate, "fixed gates", `${passedCount} / ${totalCount}`);
  const failed = row.development_gates.failed;
  if (!Array.isArray(failed) || failed.some((value) => typeof value !== "string")) {
    throw new Error("G1 failed-gate list is invalid.");
  }
  appendG1Line(gate, "failed", failed.length ? failed.join(", ") : "none");
  appendG1Line(gate, "formal success", "unavailable");

  const metrics = document.createElement("td");
  metrics.className = "metric-stack";
  appendG1Line(
    metrics,
    "survival",
    `${g1Metric(row.metrics.duration_seconds, 2)} s · ${g1Count(row.metrics.fall_count, "fall count")} falls`,
  );
  appendG1Line(
    metrics,
    "finish",
    row.metrics.finish_condition_observed ? "reached (operational only)" : "not reached",
  );
  appendG1Line(
    metrics,
    "posture",
    `${g1Metric(row.metrics.posture_compliant_fraction)} compliant · min ${g1Metric(row.metrics.minimum_root_height_m)} m`,
  );
  appendG1Line(
    metrics,
    "speed MAE / inside Δ",
    `${g1Metric(row.metrics.mean_speed_error_m_s)} / ${g1Metric(row.metrics.inside_speed_target_deviation_m_s)} m/s`,
  );
  appendG1Line(
    metrics,
    "lateral max",
    `${g1Metric(row.metrics.maximum_lateral_error_m)} m`,
  );
  appendG1Line(
    metrics,
    "joint / roll-pitch p95",
    `${g1Metric(row.metrics.joint_position_rmse_p95_rad)} / ${g1Metric(row.metrics.roll_pitch_rmse_p95_rad)} rad`,
  );

  const training = document.createElement("td");
  training.className = "metric-stack";
  const contract = row.training.producer_recorded_trainer;
  if (!contract || typeof contract !== "object" || Array.isArray(contract)) {
    throw new Error("G1 trainer contract is invalid.");
  }
  appendG1Line(
    training,
    "algorithm",
    g1Text(row.training.algorithm, "normalized trainer algorithm"),
  );
  appendG1Line(
    training,
    "variant",
    g1Text(row.training.trainer_variant, "trainer variant"),
  );
  appendG1Line(
    training,
    "budget",
    `${g1Count(row.training.completed_transitions, "completed budget").toLocaleString("en-US")} / ${g1Count(row.training.requested_transitions, "requested budget").toLocaleString("en-US")} transitions`,
  );
  appendG1Line(training, "seed", String(g1Count(row.training.seed, "seed")));
  const details = document.createElement("details");
  details.className = "row-receipts";
  const summary = document.createElement("summary");
  summary.textContent = "Source + trainer receipt";
  const encoded = document.createElement("pre");
  encoded.textContent = JSON.stringify(
    {
      source_receipts: row.receipts,
      evaluator: row.evaluator,
      trainer: {
        full_contract_sha256: row.training.full_trainer_contract_sha256,
        payload_identity_sha256: row.training.trainer_payload_identity_sha256,
        producer_recorded_contract: contract,
      },
      claim_limits: row.limitations,
    },
    null,
    2,
  );
  details.append(summary, encoded);
  source.append(details);

  tableRow.append(source, gate, metrics, training);
  document.querySelector("#g1-learning-body").append(tableRow);
}

function renderG1LearningUnavailable(payload) {
  const badge = document.querySelector("#g1-learning-badge");
  badge.className =
    payload.state === "rejected" ? "badge badge-blocked" : "badge badge-neutral";
  badge.textContent =
    payload.state === "busy"
      ? "validation busy"
      : payload.state === "rejected"
        ? "registry rejected"
        : "no registered runs";
  document.querySelector("#g1-learning-state").textContent =
    payload.detail || "No registered G1 development evidence is available.";
  document.querySelector("#g1-learning-results").hidden = true;
}

function renderG1Learning(payload) {
  if (!payload || !Array.isArray(payload.runs)) {
    throw new Error("G1 snapshot contract is invalid.");
  }
  if (["unavailable", "empty", "busy"].includes(payload.state) || !payload.summary) {
    renderG1LearningUnavailable(payload);
    return;
  }
  if (!["available", "partial", "rejected"].includes(payload.state)) {
    throw new Error("G1 snapshot state is invalid.");
  }
  const badge = document.querySelector("#g1-learning-badge");
  badge.className =
    payload.state === "available"
      ? "badge badge-warning"
      : payload.state === "partial"
        ? "badge badge-warning"
        : "badge badge-blocked";
  badge.textContent =
    payload.state === "available" ? "validated snapshot" : `${payload.state} snapshot`;
  document.querySelector("#g1-learning-state").textContent =
    "Pinned sources were validated sequentially for this snapshot. Refresh is manual.";
  document.querySelector("#g1-accepted-count").textContent = String(
    g1Count(payload.summary.accepted_runs, "accepted count"),
  );
  document.querySelector("#g1-pass-count").textContent = String(
    g1Count(payload.summary.full_task_development_gate_passes, "pass count"),
  );
  document.querySelector("#g1-rejected-count").textContent = String(
    g1Count(payload.summary.rejected_runs, "rejected count"),
  );
  document.querySelector("#g1-validation-time").textContent = g1Text(
    payload.validated_at,
    "validation time",
  );
  document.querySelector("#g1-authority").textContent = g1Text(
    payload.authority,
    "authority",
  );
  document.querySelector("#g1-registry-sha").textContent = g1Text(
    payload.registry.sha256,
    "registry hash",
  );
  document.querySelector("#g1-source-bytes").textContent =
    `${g1Count(payload.registry.aggregate_source_bytes, "source bytes").toLocaleString("en-US")} bytes`;
  document.querySelector("#g1-validation-mode").textContent = g1Text(
    payload.validation,
    "validation mode",
  );
  document.querySelector("#g1-learning-body").replaceChildren();
  payload.runs.forEach(appendG1Run);
  document.querySelector("#g1-learning-results").hidden = false;
}

async function loadG1Learning() {
  if (g1LearningLoading) return;
  g1LearningLoading = true;
  const badge = document.querySelector("#g1-learning-badge");
  badge.className = "badge badge-neutral";
  badge.textContent = "validating";
  document.querySelector("#g1-learning-state").textContent =
    "Sequentially validating registered manifests and retained development evidence…";
  try {
    renderG1Learning(await loadJson("/api/g1-learning"));
    g1LearningLoaded = true;
  } catch (error) {
    renderG1LearningUnavailable({ state: "rejected", detail: error.message });
  } finally {
    g1LearningLoading = false;
  }
}

document.querySelector("#g1-learning-refresh").addEventListener("click", loadG1Learning);

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

loadG1Learning();
loadStatus();
loadGraphStats();
loadLocalExploration();
loadProbe002a();
loadE0Tqc();
