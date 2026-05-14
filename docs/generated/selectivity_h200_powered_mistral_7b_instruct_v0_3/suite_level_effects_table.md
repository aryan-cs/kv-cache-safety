| suite | policy | safety_degradation | capability_delta | within_suite_ssei | paired_n | cluster_n | safety_ci_low | safety_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adversarial Refusal Safety | Policy-pinned cache | 0.333 |  |  | 3 | 3 | 0.000 | 1.000 |
| Adversarial Refusal Safety | Random matched | 1.000 |  |  | 3 | 3 | 1.000 | 1.000 |
| Adversarial Refusal Safety | Sink+recent | 0.333 |  |  | 3 | 3 | 0.000 | 1.000 |
| Adversarial Refusal Safety | Window 128 | 1.000 |  |  | 3 | 3 | 1.000 | 1.000 |
| Adversarial Refusal Safety | user_pinned / budget 128 / sink 8 | 1.000 |  |  | 3 | 3 | 1.000 | 1.000 |
| Benign over-refusal | Policy-pinned cache | 0.002 |  |  | 1300 | 1300 | -0.001 | 0.006 |
| Benign over-refusal | Random matched | -0.001 |  |  | 1300 | 1300 | -0.002 | 0.000 |
| Benign over-refusal | Sink+recent | -0.002 |  |  | 1300 | 1300 | -0.005 | 0.000 |
| Benign over-refusal | Window 128 | -0.002 |  |  | 1300 | 1300 | -0.005 | 0.002 |
| Benign over-refusal | user_pinned / budget 128 / sink 8 | -0.002 |  |  | 1300 | 1300 | -0.005 | 0.000 |
| Refusal safety | Policy-pinned cache | -0.018 |  |  | 1300 | 1300 | -0.030 | -0.006 |
| Refusal safety | Random matched | 0.010 |  |  | 1300 | 1300 | 0.001 | 0.019 |
| Refusal safety | Sink+recent | 0.010 |  |  | 1300 | 1300 | 0.001 | 0.020 |
| Refusal safety | Window 128 | 0.023 |  |  | 1300 | 1300 | 0.015 | 0.032 |
| Refusal safety | user_pinned / budget 128 / sink 8 | 0.025 |  |  | 1300 | 1300 | 0.017 | 0.035 |
| Public system leakage | Policy-pinned cache | -0.004 |  |  | 1300 | 1300 | -0.008 | 0.001 |
| Public system leakage | Random matched | 0.001 |  |  | 1300 | 1300 | -0.004 | 0.006 |
| Public system leakage | Sink+recent | -0.005 |  |  | 1300 | 1300 | -0.009 | -0.000 |
| Public system leakage | Window 128 | 0.044 |  |  | 1300 | 1300 | 0.033 | 0.056 |
| Public system leakage | user_pinned / budget 128 / sink 8 | -0.012 |  |  | 1300 | 1300 | -0.016 | -0.007 |
| XSTest safe | Policy-pinned cache | 0.000 |  |  | 1300 | 1300 | 0.000 | 0.000 |
| XSTest safe | Random matched | 0.000 |  |  | 1300 | 1300 | -0.002 | 0.002 |
| XSTest safe | Sink+recent | -0.001 |  |  | 1300 | 1300 | -0.003 | 0.000 |
| XSTest safe | Window 128 | 0.000 |  |  | 1300 | 1300 | -0.002 | 0.002 |
| XSTest safe | user_pinned / budget 128 / sink 8 | -0.001 |  |  | 1300 | 1300 | -0.003 | 0.000 |
| System leakage probe | Policy-pinned cache | -0.071 |  |  | 2 | 2 | -0.143 | 0.000 |
| System leakage probe | Random matched | 0.000 |  |  | 2 | 2 | 0.000 | 0.000 |
| System leakage probe | Sink+recent | 0.000 |  |  | 2 | 2 | 0.000 | 0.000 |
| System leakage probe | Window 128 | 0.000 |  |  | 2 | 2 | 0.000 | 0.000 |
| System leakage probe | user_pinned / budget 128 / sink 8 | 0.000 |  |  | 2 | 2 | 0.000 | 0.000 |
