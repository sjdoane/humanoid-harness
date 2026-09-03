#!/usr/bin/env python3
"""Build the metadata-only oracle-composition candidate registry.

This utility reads version-pinned public metadata. It does not screen,
classify, include, or extract evidence from any paper.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "humanoid-harness-literature-review/0.1"
REQUEST_INTERVAL_SECONDS = 3.0


@dataclass(frozen=True)
class CandidateSeed:
    arxiv_id: str
    expected_version: int
    search_string_id: str
    discovery_lanes: str
    provenance: str
    full_text_available: str = "yes"
    administrative_note: str = "none"
    expected_withdrawn: bool = False
    supplementary_source: str | None = None
    evidence_boundary: str | None = None


SEEDS = (
    CandidateSeed("2108.13264", 4, "D0_LEGACY_SCOPE_SEED", "evaluation", "legacy_snapshot"),
    CandidateSeed(
        "2303.05711", 2, "D0_LEGACY_SCOPE_SEED", "composition;switching", "legacy_snapshot"
    ),
    CandidateSeed("2403.04205", 3, "D0_LEGACY_SCOPE_SEED", "oracle;composition", "legacy_snapshot"),
    CandidateSeed("2403.10506", 2, "D0_LEGACY_SCOPE_SEED", "benchmark", "legacy_snapshot"),
    CandidateSeed("2406.06005", 2, "D0_LEGACY_SCOPE_SEED", "contact;sequencing", "legacy_snapshot"),
    CandidateSeed(
        "2409.14393", 1, "D0_LEGACY_SCOPE_SEED", "reference;composition", "legacy_snapshot"
    ),
    CandidateSeed("2410.01030", 3, "D0_LEGACY_SCOPE_SEED", "oracle;preference", "legacy_snapshot"),
    CandidateSeed("2502.08844", 1, "D0_LEGACY_SCOPE_SEED", "benchmark;runtime", "legacy_snapshot"),
    CandidateSeed("2507.07356", 3, "D0_LEGACY_SCOPE_SEED", "tracking", "legacy_snapshot"),
    CandidateSeed("2510.05070", 2, "D0_LEGACY_SCOPE_SEED", "tracking;residual", "legacy_snapshot"),
    CandidateSeed("2511.07820", 4, "D0_LEGACY_SCOPE_SEED", "tracking", "legacy_snapshot"),
    CandidateSeed(
        "2601.12799", 1, "D0_LEGACY_SCOPE_SEED", "instruction;composition", "legacy_snapshot"
    ),
    CandidateSeed(
        "2602.15060", 2, "D0_LEGACY_SCOPE_SEED", "closed_loop;tracking", "legacy_snapshot"
    ),
    CandidateSeed("2603.03279", 1, "D0_LEGACY_SCOPE_SEED", "multi_skill", "legacy_snapshot"),
    CandidateSeed(
        "2604.17335", 2, "D0_LEGACY_SCOPE_SEED", "generation;tracking", "legacy_snapshot"
    ),
    CandidateSeed(
        "2606.10340", 1, "D0_LEGACY_SCOPE_SEED", "generation;composition", "legacy_snapshot"
    ),
    CandidateSeed("2606.20705", 1, "D0_LEGACY_SCOPE_SEED", "hierarchy;residual", "legacy_snapshot"),
    CandidateSeed(
        "2606.22998", 2, "D0_LEGACY_SCOPE_SEED", "controller_aware;generation", "legacy_snapshot"
    ),
    CandidateSeed("2606.27581", 1, "D0_LEGACY_SCOPE_SEED", "contact;tracking", "legacy_snapshot"),
    CandidateSeed("2607.12114", 1, "D0_LEGACY_SCOPE_SEED", "gait;transition", "legacy_snapshot"),
    CandidateSeed(
        "2607.04837",
        3,
        "D0_LEGACY_SCOPE_SEED",
        "administrative_hold",
        "legacy_snapshot",
        "no",
        "arXiv withdrawal banner present and current PDF unavailable; retained for audit only",
        True,
    ),
    CandidateSeed("2608.02385", 1, "D0_LEGACY_SCOPE_SEED", "recovery;tracking", "legacy_snapshot"),
    CandidateSeed(
        "2608.02653", 1, "D0_LEGACY_SCOPE_SEED", "multi_skill;distillation", "legacy_snapshot"
    ),
    CandidateSeed(
        "2608.07746", 1, "D0_LEGACY_SCOPE_SEED", "long_horizon;latent_skill", "legacy_snapshot"
    ),
    CandidateSeed("2608.18234", 2, "D0_LEGACY_SCOPE_SEED", "robust;interaction", "legacy_snapshot"),
    CandidateSeed(
        "1804.02717", 3, "D1_COMPOSITION_SWITCHING", "phase;reference;recovery", "discovery"
    ),
    CandidateSeed("1905.09808", 1, "D1_COMPOSITION_SWITCHING", "composable_policy", "discovery"),
    CandidateSeed(
        "2205.01906", 2, "D1_COMPOSITION_SWITCHING", "skill_embedding;composition", "discovery"
    ),
    CandidateSeed(
        "2305.02195", 1, "D1_COMPOSITION_SWITCHING", "latent_skill;composition", "discovery"
    ),
    CandidateSeed("2305.06456", 3, "D1_COMPOSITION_SWITCHING", "tracking;recovery", "discovery"),
    CandidateSeed(
        "2309.11351", 1, "D1_COMPOSITION_SWITCHING", "conditional_skill;composition", "discovery"
    ),
    CandidateSeed(
        "2310.04582",
        2,
        "D1_COMPOSITION_SWITCHING",
        "motion_representation;composition",
        "discovery",
    ),
    CandidateSeed(
        "2310.10198", 3, "D1_COMPOSITION_SWITCHING", "discrete_motion;composition", "discovery"
    ),
    CandidateSeed("2408.00776", 2, "D1_COMPOSITION_SWITCHING", "contact;multi_gait", "discovery"),
    CandidateSeed(
        "2509.22442", 1, "D1_COMPOSITION_SWITCHING", "policy_composition;long_horizon", "discovery"
    ),
    CandidateSeed(
        "2510.22632", 1, "D1_COMPOSITION_SWITCHING", "motion_matching;environment", "discovery"
    ),
    CandidateSeed(
        "2602.15827", 2, "D1_COMPOSITION_SWITCHING", "motion_matching;skill_chain", "discovery"
    ),
    CandidateSeed(
        "2604.01064", 1, "D1_COMPOSITION_SWITCHING", "policy_switching;long_horizon", "discovery"
    ),
    CandidateSeed(
        "2604.14834", 1, "D1_COMPOSITION_SWITCHING", "skill_graph;switching", "discovery"
    ),
    CandidateSeed(
        "2604.21355", 2, "D1_COMPOSITION_SWITCHING", "policy_gating;transition", "discovery"
    ),
    CandidateSeed(
        "2607.24083", 1, "D1_COMPOSITION_SWITCHING", "hybrid_prior;composition", "discovery"
    ),
    CandidateSeed("2201.04439", 1, "D2_PHASE_RECOVERY", "local_phase", "discovery"),
    CandidateSeed("2308.12751", 1, "D2_PHASE_RECOVERY", "phase_manifold;transition", "discovery"),
    CandidateSeed("2407.18946", 1, "D2_PHASE_RECOVERY", "phase_manifold;alignment", "discovery"),
    CandidateSeed("2510.14454", 1, "D2_PHASE_RECOVERY", "adaptive_phase;time_warp", "discovery"),
    CandidateSeed("2512.09423", 2, "D2_PHASE_RECOVERY", "phase_manifold", "discovery"),
    CandidateSeed("2601.23080", 1, "D2_PHASE_RECOVERY", "tracking;recovery", "discovery"),
    CandidateSeed("2602.13656", 1, "D2_PHASE_RECOVERY", "tracking;fall_recovery", "discovery"),
    CandidateSeed("2603.27756", 2, "D2_PHASE_RECOVERY", "state_conditioned_reference", "discovery"),
    CandidateSeed("2604.22911", 1, "D2_PHASE_RECOVERY", "contact;recovery", "discovery"),
    CandidateSeed("2606.12814", 1, "D2_PHASE_RECOVERY", "tracking;fall_recovery", "discovery"),
    CandidateSeed("2104.02180", 2, "D3_FEASIBILITY_ADMISSION", "motion_prior", "discovery"),
    CandidateSeed("2208.07363", 3, "D3_FEASIBILITY_ADMISSION", "dataset;control", "discovery"),
    CandidateSeed("2406.08858", 1, "D3_FEASIBILITY_ADMISSION", "retargeting;tracking", "discovery"),
    CandidateSeed(
        "2410.21229", 2, "D3_FEASIBILITY_ADMISSION", "whole_body_controller", "discovery"
    ),
    CandidateSeed("2412.13196", 2, "D3_FEASIBILITY_ADMISSION", "retargeting;tracking", "discovery"),
    CandidateSeed(
        "2508.08241", 4, "D3_FEASIBILITY_ADMISSION", "guided_reference;control", "discovery"
    ),
    CandidateSeed(
        "2509.15443", 2, "D3_FEASIBILITY_ADMISSION", "retargeting;kinodynamic", "discovery"
    ),
    CandidateSeed(
        "2510.02252", 1, "D3_FEASIBILITY_ADMISSION", "retargeting;feasibility", "discovery"
    ),
    CandidateSeed("2602.20375", 1, "D3_FEASIBILITY_ADMISSION", "reference;goal", "discovery"),
    CandidateSeed(
        "2603.09956", 2, "D3_FEASIBILITY_ADMISSION", "retargeting;contact;dynamics", "discovery"
    ),
    CandidateSeed("2603.22201", 3, "D3_FEASIBILITY_ADMISSION", "retargeting;dynamics", "discovery"),
    CandidateSeed("2606.03476", 1, "D3_FEASIBILITY_ADMISSION", "retargeting;physics", "discovery"),
    CandidateSeed(
        "2607.06052", 1, "D3_FEASIBILITY_ADMISSION", "benchmark;contact;force", "discovery"
    ),
    CandidateSeed(
        "2506.14770",
        2,
        "D5_TARGETED_WEB_DISCOVERY",
        "future_reference_window;tracking;adaptive_sampling",
        "agent_host_web_search_2026-09-03",
        supplementary_source=(
            "https://github.com/zixuan417/humanoid-general-motion-tracking/"
            "tree/2a590de25a1eb08e47491977a738549c22f16e1f"
        ),
        evidence_boundary=(
            "candidate_and_official_code_metadata_only_not_registered_screened_extracted_"
            "or_evidence_of_harness_implementation"
        ),
    ),
    CandidateSeed(
        "2509.13833",
        3,
        "D5_TARGETED_WEB_DISCOVERY",
        "disturbance_adaptation;tracking;recovery",
        "agent_host_web_search_2026-09-03",
        supplementary_source=(
            "https://github.com/GalaxyGeneralRobotics/OpenTrack/"
            "tree/cb9b751993a2483e5d1805a2565ddbfe950c04c9"
        ),
        evidence_boundary=(
            "candidate_and_official_code_metadata_only_not_registered_screened_extracted_"
            "or_evidence_of_harness_implementation"
        ),
    ),
)


DOI_ROWS = (
    (
        "10.1145/383259.383287",
        "Composable Controllers for Physics-Based Character Animation",
        "Petros Faloutsos; Michiel van de Panne; Demetri Terzopoulos",
        2001,
        "ACM SIGGRAPH",
        "https://www.dgp.toronto.edu/~pfal/papers/siggraph2001.pdf",
        "control_graph;composition",
    ),
    (
        "10.1145/566654.566605",
        "Motion Graphs",
        "Lucas Kovar; Michael Gleicher; Frédéric Pighin",
        2002,
        "ACM SIGGRAPH",
        "https://graphics.cs.wisc.edu/Papers/2002/KGP02/mograph.pdf",
        "motion_graph;composition",
    ),
    (
        "10.1145/566654.566607",
        "Interactive Control of Avatars Animated With Human Motion Data",
        "Jehee Lee; Jinxiang Chai; Paul S. A. Reitsma; Jessica K. Hodgins; Nancy S. Pollard",
        2002,
        "ACM SIGGRAPH",
        "https://graphics.cs.cmu.edu/projects/Avatar/avatar.pdf",
        "motion_graph;interactive_control",
    ),
    (
        "10.1145/1230100.1230123",
        "Parametric Motion Graphs",
        "Rachel Heck; Michael Gleicher",
        2007,
        "ACM Symposium on Interactive 3D Graphics and Games",
        "https://graphics.cs.wisc.edu/Papers/2007/HG07/PMGFullPaper.pdf",
        "motion_graph;parameterization",
    ),
    (
        "10.1145/1276377.1276510",
        "Construction and Optimal Search of Interpolated Motion Graphs",
        "Alla Safonova; Jessica K. Hodgins",
        2007,
        "ACM SIGGRAPH",
        "https://graphics.cs.cmu.edu/projects/interpolated_motion_graphs/interpolated_motion_graphs.pdf",
        "motion_graph;search",
    ),
    (
        "10.1145/2366145.2366173",
        "Terrain Runner: Control, Parameterization, Composition, and Planning for Highly Dynamic Motions",
        "Libin Liu; KangKang Yin; Michiel van de Panne; Baining Guo",
        2012,
        "ACM Transactions on Graphics",
        "https://www.cs.ubc.ca/~van/papers/2012-TOG-TerrainRunner.pdf",
        "control;composition;planning",
    ),
    (
        "10.1111/cgf.12607",
        "Controller Design for Multi-Skilled Bipedal Characters",
        "Michael Firmin; Michiel van de Panne",
        2015,
        "Computer Graphics Forum",
        "https://www.cs.ubc.ca/~van/papers/2015-CGF-multiskilled/2015-CGF-multiskilled.pdf",
        "control_graph;multi_skill",
    ),
    (
        "10.1145/2893476",
        "Guided Learning of Control Graphs for Physics-Based Characters",
        "Libin Liu; Michiel van de Panne; KangKang Yin",
        2016,
        "ACM Transactions on Graphics",
        "https://www.cs.ubc.ca/~van/papers/2016-TOG-controlGraphs/2016-TOG-controlGraphs.pdf",
        "control_graph;learning",
    ),
    (
        "10.1145/3072959.3073663",
        "Phase-Functioned Neural Networks for Character Control",
        "Daniel Holden; Taku Komura; Jun Saito",
        2017,
        "ACM Transactions on Graphics",
        "https://www.research.ed.ac.uk/files/35467734/phasefunction.pdf",
        "phase;character_control",
    ),
    (
        "10.1145/3083723",
        "Learning to Schedule Control Fragments for Physics-Based Characters Using Deep Q-Learning",
        "Libin Liu; Jessica K. Hodgins",
        2017,
        "ACM Transactions on Graphics",
        "https://graphics.cs.cmu.edu/wp/wp-content/uploads/2017/07/a29-liu.pdf",
        "scheduler;control_fragment",
    ),
    (
        "10.1145/3386569.3392440",
        "Learned Motion Matching",
        "Daniel Holden; Oussama Kanoun; Maksym Perepichka; Tiberiu Popa",
        2020,
        "ACM Transactions on Graphics",
        "https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/2EVXGPN6ynTrrJaaAUHmZS/060e223cce6a84e47e925105cac5e17f/Learned_Motion_Matching.pdf",
        "motion_matching;state_conditioning",
    ),
    (
        "10.1145/3386569.3392450",
        "Local Motion Phases for Learning Multi-Contact Character Movements",
        "Sebastian Starke; Yiwei Zhao; Taku Komura; Kazi Zaman",
        2020,
        "ACM Transactions on Graphics",
        "https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf",
        "local_phase;multi_contact",
    ),
    (
        "10.1145/3528223.3530178",
        "DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds",
        "Sebastian Starke; Ian Mason; Taku Komura",
        2022,
        "ACM Transactions on Graphics",
        "https://i.cs.hku.hk/~taku/deepphase.pdf",
        "phase_manifold",
    ),
    (
        "10.1145/3528233.3530735",
        "Learning Soccer Juggling Skills with Layer-wise Mixture-of-Experts",
        "Zhaoming Xie; Sebastian Starke; Hung Yu Ling; Michiel van de Panne",
        2022,
        "ACM SIGGRAPH",
        "https://zhaomingxie.github.io/projects/SoccerJuggle/SoccerJuggle.pdf",
        "multi_skill;mixture_of_experts",
    ),
)


CSV_FIELDS = (
    "candidate_id",
    "database",
    "search_string_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "abstract_available",
    "full_text_available",
    "candidate_relevance",
    "workflow_stage_fit",
    "grounding_relevance",
    "conceptual_synthesis_relevance",
    "empirical_evaluation",
    "notes",
)


def _meta(html_text: str, name: str, *, property_name: bool = False) -> str | None:
    attribute = "property" if property_name else "name"
    match = re.search(rf'<meta {attribute}="{re.escape(name)}" content="([^"]*)"', html_text)
    return html.unescape(match.group(1)) if match else None


def _fetch(seed: CandidateSeed, *, attempts: int = 4) -> dict[str, str | int]:
    url = f"https://arxiv.org/abs/{seed.arxiv_id}v{seed.expected_version}"
    allowed_pdf_urls = {
        f"https://arxiv.org/pdf/{seed.arxiv_id}",
        f"https://arxiv.org/pdf/{seed.arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{seed.arxiv_id}v{seed.expected_version}",
        f"https://arxiv.org/pdf/{seed.arxiv_id}v{seed.expected_version}.pdf",
    }
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8", "replace")
            title = _meta(text, "citation_title")
            date = _meta(text, "citation_date")
            current_url = _meta(text, "og:url", property_name=True) or ""
            pdf_url = _meta(text, "citation_pdf_url")
            authors = [
                html.unescape(value)
                for value in re.findall(r'<meta name="citation_author" content="([^"]*)"', text)
            ]
            if not title or not date or not authors:
                raise ValueError(f"incomplete arXiv metadata for {seed.arxiv_id}")
            withdrawn = "paper has been withdrawn" in text.lower() or "(withdrawn)" in text.lower()
            if withdrawn != seed.expected_withdrawn:
                raise ValueError(f"{seed.arxiv_id} withdrawal state changed: observed={withdrawn}")
            if current_url != url:
                raise ValueError(
                    f"{seed.arxiv_id} observed metadata identity changed: {current_url!r}"
                )
            if pdf_url is not None and pdf_url not in allowed_pdf_urls:
                raise ValueError(f"{seed.arxiv_id} has a mismatched PDF URL: {pdf_url!r}")
            if seed.full_text_available == "yes" and pdf_url is None:
                raise ValueError(f"{seed.arxiv_id} has no discoverable PDF URL")
            supplementary_note = (
                f"; supplementary_source={seed.supplementary_source}; "
                "supplementary_source_role=official_code_only"
                if seed.supplementary_source
                else ""
            )
            boundary_note = (
                f"; evidence_boundary={seed.evidence_boundary}" if seed.evidence_boundary else ""
            )
            return {
                "candidate_id": f"arxiv:{seed.arxiv_id}v{seed.expected_version}",
                "database": "arXiv",
                "search_string_id": seed.search_string_id,
                "title": title,
                "authors": "; ".join(authors),
                "year": int(date[:4]),
                "venue": "arXiv",
                "doi": f"10.48550/arXiv.{seed.arxiv_id}",
                "url": url,
                "abstract_available": "yes",
                "full_text_available": seed.full_text_available,
                "candidate_relevance": "not_assessed",
                "workflow_stage_fit": "not_assessed",
                "grounding_relevance": "not_assessed",
                "conceptual_synthesis_relevance": "not_assessed",
                "empirical_evaluation": "not_assessed",
                "notes": (
                    "candidate-only metadata; no screening or inclusion decision; "
                    f"discovery_lanes={seed.discovery_lanes}; provenance={seed.provenance}; "
                    f"administrative_note={seed.administrative_note}"
                    f"{supplementary_note}{boundary_note}"
                ),
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(max(REQUEST_INTERVAL_SECONDS, 2**attempt))
    raise RuntimeError(f"failed to fetch {seed.arxiv_id}: {error}")


def _doi_row(record: tuple[str, str, str, int, str, str, str]) -> dict[str, str | int]:
    doi, title, authors, year, venue, open_pdf, lanes = record
    return {
        "candidate_id": f"doi:{doi}",
        "database": "publisher record plus author manuscript",
        "search_string_id": "D4_FOUNDATION_CHAIN",
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "url": f"https://doi.org/{doi}",
        "abstract_available": "yes",
        "full_text_available": "yes",
        "candidate_relevance": "not_assessed",
        "workflow_stage_fit": "not_assessed",
        "grounding_relevance": "not_assessed",
        "conceptual_synthesis_relevance": "not_assessed",
        "empirical_evaluation": "not_assessed",
        "notes": (
            "candidate-only metadata; no screening or inclusion decision; "
            f"discovery_lanes={lanes}; open_manuscript={open_pdf}"
        ),
    }


def build(output: Path) -> None:
    rows = []
    for index, seed in enumerate(SEEDS):
        rows.append(_fetch(seed))
        if index + 1 < len(SEEDS):
            time.sleep(REQUEST_INTERVAL_SECONDS)
    rows.extend(_doi_row(record) for record in DOI_ROWS)
    rows.sort(key=lambda row: str(row["candidate_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
