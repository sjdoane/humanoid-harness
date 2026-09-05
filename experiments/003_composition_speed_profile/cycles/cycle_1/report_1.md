# Composition cycle 1

Evidence class: `exploratory_oracle_cycle`. Controller switching stands in for tracker following.

The four cycle-0 rows are carried forward unchanged; only the cycle-1 candidate was evaluated in cycle 1.

| source | arm | episodes | median MAE (m/s) | falls | median switches | median task return |
|---|---|---:|---:|---:|---:|---:|
| cycle 0 baseline | single_fast | 20 | 2.074172 | 0 | 0.0 | 10648.552984 |
| cycle 0 baseline | single_slow | 20 | 3.307116 | 0 | 0.0 | 5772.109180 |
| cycle 0 baseline | playback | 20 | 3.180343 | 20 | 2.0 | 2836.434684 |
| cycle 0 baseline | handwritten | 20 | 3.206216 | 20 | 2.0 | 2954.167925 |
| cycle 1 candidate | cycle_1_candidate | 20 | 3.221507 | 20 | 3.0 | 2995.057374 |

## Outcome

The candidate **failed** the never-fall requirement: 20/20 episodes fell.
Metric outcomes are reported separately because no combined ranking was predeclared. Negative deltas favor the candidate.

| cycle-0 baseline | fall-count delta | fall outcome | median-MAE delta (m/s) | MAE outcome |
|---|---:|---|---:|---|
| single_fast | +20 | worsened | +1.147335 | worsened |
| single_slow | +20 | worsened | -0.085609 | improved |
| playback | +0 | matched | +0.041165 | worsened |
| handwritten | +0 | matched | +0.015291 | worsened |

## Candidate controller switches by episode

A fall is marked only when the episode's first fall boundary occurred after the switch and no more than 100 control steps later.

| seed | step | from behavior | to behavior | v_x at switch (m/s) | first fall within 100 steps |
|---:|---:|---|---|---:|:---:|
| 97001 | 12 | expert | medium | 0.479792 | no |
| 97001 | 13 | medium | expert | 0.557593 | no |
| 97001 | 300 | expert | medium | 4.796521 | yes |
| 97002 | 11 | expert | medium | 0.348124 | no |
| 97002 | 14 | medium | expert | 0.631872 | no |
| 97002 | 300 | expert | medium | 5.472233 | yes |
| 97003 | 12 | expert | medium | 0.367721 | no |
| 97003 | 14 | medium | expert | 0.655329 | no |
| 97003 | 300 | expert | medium | 5.892096 | yes |
| 97004 | 12 | expert | medium | 0.484465 | no |
| 97004 | 13 | medium | expert | 0.535530 | no |
| 97004 | 300 | expert | medium | 5.518580 | yes |
| 97005 | 11 | expert | medium | 0.373086 | no |
| 97005 | 14 | medium | expert | 0.649959 | no |
| 97005 | 300 | expert | medium | 5.511487 | yes |
| 97006 | 300 | expert | medium | 5.411297 | yes |
| 97007 | 11 | expert | medium | 0.378624 | no |
| 97007 | 14 | medium | expert | 0.602336 | no |
| 97007 | 300 | expert | medium | 4.840107 | yes |
| 97008 | 12 | expert | medium | 0.438193 | no |
| 97008 | 13 | medium | expert | 0.538464 | no |
| 97008 | 300 | expert | medium | 4.837852 | yes |
| 97009 | 300 | expert | medium | 5.193227 | yes |
| 97010 | 300 | expert | medium | 5.533237 | yes |
| 97011 | 11 | expert | medium | 0.289574 | no |
| 97011 | 14 | medium | expert | 0.615315 | no |
| 97011 | 300 | expert | medium | 5.345459 | yes |
| 97012 | 12 | expert | medium | 0.389537 | no |
| 97012 | 14 | medium | expert | 0.490062 | no |
| 97012 | 300 | expert | medium | 5.333373 | yes |
| 97013 | 300 | expert | medium | 5.368107 | yes |
| 97014 | 12 | expert | medium | 0.388099 | no |
| 97014 | 13 | medium | expert | 0.403637 | no |
| 97014 | 300 | expert | medium | 5.368463 | yes |
| 97015 | 300 | expert | medium | 5.145355 | yes |
| 97016 | 300 | expert | medium | 5.448444 | yes |
| 97017 | 12 | expert | medium | 0.400864 | no |
| 97017 | 13 | medium | expert | 0.480719 | no |
| 97017 | 300 | expert | medium | 5.403350 | yes |
| 97018 | 11 | expert | medium | 0.367969 | no |
| 97018 | 14 | medium | expert | 0.657264 | no |
| 97018 | 300 | expert | medium | 5.444186 | yes |
| 97019 | 11 | expert | medium | 0.371495 | no |
| 97019 | 14 | medium | expert | 0.620038 | no |
| 97019 | 300 | expert | medium | 5.206997 | yes |
| 97020 | 12 | expert | medium | 0.376362 | no |
| 97020 | 14 | medium | expert | 0.526652 | no |
| 97020 | 300 | expert | medium | 5.618782 | yes |

## Slow-third behavior fractions by episode

Slow-third control steps: `[300,600)`.

| seed | expert | medium | simple |
|---:|---:|---:|---:|
| 97001 | 0.000000 | 1.000000 | 0.000000 |
| 97002 | 0.000000 | 1.000000 | 0.000000 |
| 97003 | 0.000000 | 1.000000 | 0.000000 |
| 97004 | 0.000000 | 1.000000 | 0.000000 |
| 97005 | 0.000000 | 1.000000 | 0.000000 |
| 97006 | 0.000000 | 1.000000 | 0.000000 |
| 97007 | 0.000000 | 1.000000 | 0.000000 |
| 97008 | 0.000000 | 1.000000 | 0.000000 |
| 97009 | 0.000000 | 1.000000 | 0.000000 |
| 97010 | 0.000000 | 1.000000 | 0.000000 |
| 97011 | 0.000000 | 1.000000 | 0.000000 |
| 97012 | 0.000000 | 1.000000 | 0.000000 |
| 97013 | 0.000000 | 1.000000 | 0.000000 |
| 97014 | 0.000000 | 1.000000 | 0.000000 |
| 97015 | 0.000000 | 1.000000 | 0.000000 |
| 97016 | 0.000000 | 1.000000 | 0.000000 |
| 97017 | 0.000000 | 1.000000 | 0.000000 |
| 97018 | 0.000000 | 1.000000 | 0.000000 |
| 97019 | 0.000000 | 1.000000 | 0.000000 |
| 97020 | 0.000000 | 1.000000 | 0.000000 |
