## Per-model headline

| Family | Model | Top positive SSEI policy | SSEI [95% CI] | Top |SSEI| policy | SSEI [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Gemma | Gemma-2-9B-IT | -- | -- | `sliding_window__budget128` | -0.005 [-0.008, -0.003] |
| OpenAI | GPT-OSS-20B | -- | -- | `user_pinned__budget128__sink8` | -0.082 [-0.101, -0.063] |
| Llama | Llama-3.1-8B-Instruct | `user_pinned__budget128__sink8` | +0.024 [+0.016, +0.031] | `sliding_window__budget128` | -0.085 [-0.105, -0.067] |
| Mistral | Mistral-7B-Instruct-v0.3 | -- | -- | `sliding_window__budget128` | +0.009 [+0.002, +0.017] |
| OLMo | OLMo-3-7B-Instruct | `user_pinned__budget128__sink8` | +0.026 [+0.019, +0.033] | `user_pinned__budget128__sink8` | +0.026 [+0.019, +0.033] |
| Phi | Phi-4 | `sliding_window__budget128` | +0.084 [+0.076, +0.091] | `sliding_window__budget128` | +0.084 [+0.076, +0.091] |
| Qwen | Qwen2.5-7B base | -- | -- | -- | -- |
| Qwen | Qwen2.5-7B-Instruct | `user_pinned__budget128__sink8` | +0.017 [+0.011, +0.022] | `user_pinned__budget128__sink8` | +0.017 [+0.011, +0.022] |
| Qwen | Qwen3-9B | `sliding_window__budget128` | +0.092 [+0.083, +0.101] | `sliding_window__budget128` | +0.092 [+0.083, +0.101] |

## Leave-one-family-out cross-family claim check

| Excluded family | Positive families remaining | Claim holds? |
| --- | --- | --- |
| Gemma | Llama, OLMo, Phi, Qwen | YES |
| Llama | OLMo, Phi, Qwen | YES |
| Mistral | Llama, OLMo, Phi, Qwen | YES |
| OLMo | Llama, Phi, Qwen | YES |
| OpenAI | Llama, OLMo, Phi, Qwen | YES |
| Phi | Llama, OLMo, Qwen | YES |
| Qwen | Llama, OLMo, Phi | YES |
