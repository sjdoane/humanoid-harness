# ADR 0001: Oracle-first scope

- **Status:** accepted for the first Family-A study; superseded as the
  program-wide scope
- **Date:** 2026-09-02

## Decision

Experiment 001 is a tracker-admission prerequisite with no authorable research
factor. Experiment 002 separately establishes time-varying causal reference
use. The first controlled oracle comparison, planned as Experiment 003,
studies reference-oracle composition with task reward frozen. Task-reward
generation remains a separate program workstream and must use a frozen oracle
in its own single-factor study.

## Rationale

The prior repository changed both oracle-like commands and task rewards before
establishing a controller that causally consumed controller-native reference
values. That made oracle attribution impossible. A one-variable study makes the
causal claim identifiable.

## Consequences

- Build and freeze a reference-conditioned tracker before oracle search.
- Use fixed, random/open-loop, manual phase, manual state, and generated oracle
  arms under matched conditions.
- A reward change creates a different experiment family.
- PRAXIST may propose bounded oracle variants only after the protected evaluator
  is operational.
- The program-level two-knob contract is now defined in
  [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md).
