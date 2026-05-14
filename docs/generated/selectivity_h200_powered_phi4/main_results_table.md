| policy | mean_safety | mean_capability | policy_level_ssei | policy_level_ssei_ci_low | policy_level_ssei_ci_high | policy_level_safety_clusters | policy_level_capability_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.883 | 0.944 |  |  |  |  |  |
| Policy-pinned cache | 0.883 | 0.943 | -0.001 | -0.003 | 0.000 | 5205 | 1300 |
| Random matched | 0.687 | 0.944 | 0.044 | 0.038 | 0.050 | 5205 | 1300 |
| Sink+recent | 0.883 | 0.945 | 0.001 | -0.000 | 0.003 | 5205 | 1300 |
| Window 128 | 0.662 | 0.945 | 0.084 | 0.076 | 0.091 | 5205 | 1300 |
| user_pinned / budget 128 / sink 8 | 0.674 | 0.935 | 0.055 | 0.046 | 0.063 | 5205 | 1300 |
