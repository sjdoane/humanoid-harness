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
