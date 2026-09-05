# Oracle designer prompt: cycle 0

Evidence label: `exploratory_oracle_cycle`
Task spec SHA-256: `edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7`
Library manifest SHA-256: `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207`

## Task

Run at the fast gait, slow to the slow gait for the middle third, return to the fast gait, and never fall.

Schedule: [0,300): 5.520768616125 m/s, [300,600): 0.885359990858 m/s, [600,1000): 5.520768616125 m/s.
Horizon: 1000 steps at 0.015 seconds per step.
The untouched stock reward is descriptive task_return only and must not be changed.

## Frozen behavior library

| behavior | median m/s | IQR m/s | admitted clips | E3 falls / 1,000 steps |
|---|---:|---:|---:|---:|
| expert | 5.520768616125 | 0.719302410388 | 28 | 0.055555555556 |
| medium | 3.074019875896 | 0.491353037455 | 28 | 0.000000000000 |
| simple | 0.885359990858 | 0.210745240529 | 28 | 0.194444444444 |

Speed uses root-x sidecar boundary differences over 0.015 seconds on the 28 E3-admitted blocks. Fall rates use all 36 predeclared E3 branches and are not performance guarantees.

## Prior cycle report

| arm | episodes | median MAE m/s | falls | median switches | median task return |
|---|---:|---:|---:|---:|---:|
| none | 0 | n/a | n/a | n/a | n/a |

## Oracle JSON contract

Schema: `humanoid_controller_switching_oracle/v1`.
Allowed signals only: `dwell, t, torso_up, v_target, v_x, x_travelled, z_root`.
Guards allow numeric constants, comparisons, `and`, `or`, and `not`; calls, attributes, arithmetic, and unknown names are invalid.
Transitions are checked in ascending priority after the current state's min_dwell is met. At most one controller switch occurs per step.
Recovery is optional, has highest priority, overrides with its behavior while true, and rejoins the suspended state with dwell reset when false.

```json
{
  "behaviors": [
    "expert",
    "medium",
    "simple"
  ],
  "evidence_class": "exploratory_oracle_cycle",
  "initial": "state_name",
  "oracle_id": "cycle_0_candidate",
  "recovery": {
    "behavior": "expert",
    "guard": "z_root < 1.1"
  },
  "schema_version": 1,
  "states": {
    "next_state": {
      "behavior": "expert",
      "min_dwell": 0
    },
    "state_name": {
      "behavior": "expert",
      "min_dwell": 1
    }
  },
  "transitions": [
    {
      "from": "state_name",
      "guard": "t >= 300",
      "priority": 0,
      "to": "next_state"
    }
  ]
}
```

## Rules

- Return one JSON object only; never return or request Python.
- Use only behavior names in the frozen library and only observable allowed signals.
- Do not change the task, schedule, runtime, actors, reward, seeds, horizon, or metrics.
- Every state needs a non-negative integer min_dwell; every transition needs a unique priority among transitions from its source.
- Avoid unreachable states and zero-dwell cycles.
- This executor switches controllers as a stand-in for tracker following. Do not claim tracker or oracle quality from this cycle.

## Human steering

No additional steering text was supplied.
