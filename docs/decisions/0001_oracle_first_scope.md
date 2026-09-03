# ADR 0001: Oracle-first scope

- **Status:** accepted
- **Date:** 2026-09-02

## Decision

The new repository studies reference-oracle composition first. Task-reward
generation remains frozen and deferred.

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

