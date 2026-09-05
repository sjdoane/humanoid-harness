# T1: speed profile

| field | frozen value |
|---|---|
| evidence class | `exploratory_oracle_cycle` |
| task text | Run at the fast gait, slow to the slow gait for the middle third, return to the fast gait, and never fall. |
| fast target | expert admitted-corpus median: `5.520768616125457 m/s` |
| slow candidates | medium: `3.07401987589581 m/s`; simple: `0.8853599908576963 m/s` |
| slow choice | `simple`; its median is farthest below the expert median |
| schedule | expert median on `[0,300)`, simple median on `[300,600)`, expert median on `[600,1000)` |
| library manifest SHA-256 | `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207` |
| task spec SHA-256 | `edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7` |

The speed statistic is the pooled per-step root-x sidecar difference over the
`0.015 s` control period for each actor's clips in the `28` v2 E3-admitted
blocks. The task spec and these targets are frozen before any oracle evaluation.
