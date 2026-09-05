# Composition cycle 2

Evidence class: `exploratory_oracle_cycle`. Controller switching stands in for tracker following.

The 5 rows from the cycle-1 report are carried forward unchanged; only the cycle-2 candidate was evaluated in cycle 2.

| source | arm | episodes | median MAE (m/s) | falls | median switches | median task return |
|---|---|---:|---:|---:|---:|---:|
| cycle 0 baseline | single_fast | 20 | 2.074172 | 0 | 0.0 | 10648.552984 |
| cycle 0 baseline | single_slow | 20 | 3.307116 | 0 | 0.0 | 5772.109180 |
| cycle 0 baseline | playback | 20 | 3.180343 | 20 | 2.0 | 2836.434684 |
| cycle 0 baseline | handwritten | 20 | 3.206216 | 20 | 2.0 | 2954.167925 |
| cycle 1 candidate | cycle_1_candidate | 20 | 3.221507 | 20 | 3.0 | 2995.057374 |
| cycle 2 candidate | cycle_2_candidate | 20 | 2.074172 | 0 | 0.0 | 10648.552984 |

## Outcome

The candidate **passed** the never-fall requirement: 0/20 episodes fell.
Metric outcomes are reported separately because no combined ranking was predeclared. Negative deltas favor the candidate.

| prior arm | fall-count delta | fall outcome | median-MAE delta (m/s) | MAE outcome |
|---|---:|---|---:|---|
| single_fast | +0 | matched | +0.000000 | matched |
| single_slow | +0 | matched | -1.232944 | improved |
| playback | -20 | improved | -1.106170 | improved |
| handwritten | -20 | improved | -1.132044 | improved |
| cycle_1_candidate | -20 | improved | -1.147335 | improved |

## Candidate controller switches by episode

A fall is marked only when the episode's first fall boundary occurred after the switch and no more than 100 control steps later.

| seed | step | from behavior | to behavior | v_x at switch (m/s) | first fall within 100 steps |
|---:|---:|---|---|---:|:---:|
| 97001 | n/a | none | none | n/a | no |
| 97002 | n/a | none | none | n/a | no |
| 97003 | n/a | none | none | n/a | no |
| 97004 | n/a | none | none | n/a | no |
| 97005 | n/a | none | none | n/a | no |
| 97006 | n/a | none | none | n/a | no |
| 97007 | n/a | none | none | n/a | no |
| 97008 | n/a | none | none | n/a | no |
| 97009 | n/a | none | none | n/a | no |
| 97010 | n/a | none | none | n/a | no |
| 97011 | n/a | none | none | n/a | no |
| 97012 | n/a | none | none | n/a | no |
| 97013 | n/a | none | none | n/a | no |
| 97014 | n/a | none | none | n/a | no |
| 97015 | n/a | none | none | n/a | no |
| 97016 | n/a | none | none | n/a | no |
| 97017 | n/a | none | none | n/a | no |
| 97018 | n/a | none | none | n/a | no |
| 97019 | n/a | none | none | n/a | no |
| 97020 | n/a | none | none | n/a | no |

## Slow-third behavior fractions by episode

Slow-third control steps: `[300,600)`.

| seed | expert | medium | simple |
|---:|---:|---:|---:|
| 97001 | 1.000000 | 0.000000 | 0.000000 |
| 97002 | 1.000000 | 0.000000 | 0.000000 |
| 97003 | 1.000000 | 0.000000 | 0.000000 |
| 97004 | 1.000000 | 0.000000 | 0.000000 |
| 97005 | 1.000000 | 0.000000 | 0.000000 |
| 97006 | 1.000000 | 0.000000 | 0.000000 |
| 97007 | 1.000000 | 0.000000 | 0.000000 |
| 97008 | 1.000000 | 0.000000 | 0.000000 |
| 97009 | 1.000000 | 0.000000 | 0.000000 |
| 97010 | 1.000000 | 0.000000 | 0.000000 |
| 97011 | 1.000000 | 0.000000 | 0.000000 |
| 97012 | 1.000000 | 0.000000 | 0.000000 |
| 97013 | 1.000000 | 0.000000 | 0.000000 |
| 97014 | 1.000000 | 0.000000 | 0.000000 |
| 97015 | 1.000000 | 0.000000 | 0.000000 |
| 97016 | 1.000000 | 0.000000 | 0.000000 |
| 97017 | 1.000000 | 0.000000 | 0.000000 |
| 97018 | 1.000000 | 0.000000 | 0.000000 |
| 97019 | 1.000000 | 0.000000 | 0.000000 |
| 97020 | 1.000000 | 0.000000 | 0.000000 |
