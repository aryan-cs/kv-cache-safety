| policy | mean_safety | mean_capability | policy_level_ssei | policy_level_ssei_ci_low | policy_level_ssei_ci_high | policy_level_safety_clusters | policy_level_capability_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.607 | 0.002 |  |  |  |  |  |
| Policy-pinned cache | 0.612 | 0.002 | 0.010 | 0.006 | 0.015 | 5205 | 1300 |
| Random matched | 0.609 | 0.007 | 0.020 | 0.014 | 0.026 | 5205 | 1300 |
| Sink+recent | 0.733 | 0.002 | -0.005 | -0.009 | -0.001 | 5205 | 1300 |
| Window 128 | 0.478 | 0.008 | 0.092 | 0.083 | 0.101 | 5205 | 1300 |
| user_pinned / budget 128 / sink 8 | 0.621 | 0.003 | -0.002 | -0.006 | 0.002 | 5205 | 1300 |
