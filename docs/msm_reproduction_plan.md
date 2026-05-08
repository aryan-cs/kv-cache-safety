# Model Spec Midtraining Reproduction Plan

This project treats `chloeli-15/model_spec_midtraining` as the source of truth for MSM follow-ups.

Official sources:

- Code: https://github.com/chloeli-15/model_spec_midtraining
- Paper specs: https://github.com/chloeli-15/model_spec_midtraining/tree/main/spec/paper
- Released adapters: https://huggingface.co/chloeli/collections
- Paper: https://arxiv.org/abs/2605.02087

## Reuse Before Recreating

Do not retrain an MSM model when the same public adapter is already released by the paper authors. Use the released PEFT adapter with its base model and pinned adapter revision.

Released adapters currently cover Qwen 2.5 14B/32B Instruct and Qwen 3 14B/32B for several paper specs and stages, including:

- Rules Spec
- Rules-Augmented Spec
- Value-Augmented Spec
- General Spec
- Philosophy Spec
- MSM-only
- MSM + AFT with CoT
- MSM + AFT without CoT
- AFT-only controls

These are useful as official positive-control arms. They are not exact replacements for the current 7B/8B selectivity panel models.

## Exact Current-Panel Models

For exact current-panel follow-ups, recreate missing adapters with the official code/procedure if no public adapter exists:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen3-8B`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `allenai/Olmo-3-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `microsoft/phi-4`
- `openai/gpt-oss-20b`

Do not start by recreating all of these. First evaluate official Qwen adapters and/or train a single exact-panel pilot, then expand only if the result is informative.

## Official Procedure To Match

Use the paper repo data-generation pipeline:

1. Put or select a spec under `spec/`, preferably from `spec/paper`.
2. Generate MSM documents with `exps/generate_msm_data.sh` / `src/msm/generate_data_from_spec.py`.
3. Generate AFT chat data with `exps/generate_aft_chat.sh` / `src/aft/generate_chat.py`.
4. Preserve all intermediate artifacts, summaries, token counts, filtering outputs, and generated datasets.
5. Fine-tune with PEFT/LoRA using the paper hyperparameters when recreating missing adapters.

Paper hyperparameters to mirror unless explicitly changed and recorded:

- LoRA rank 64, alpha 128.
- Target all attention and MLP projection layers.
- One epoch.
- AdamW, learning rate `1e-4`, cosine schedule, 5% warmup, weight decay `0.01`.
- Max sequence length 8192 for the complex safety/spec experiments.
- 8B-class models fit on one H200; 14B and 32B training may require more GPUs than the UIUC single-H200 instance.

## Evaluation Rule

Every MSM or AFT variant must be evaluated against its matching base model under the same KV-cache selectivity config. Do not compare an official 14B adapter against a 7B baseline.

For official released adapters, record:

- base model id and revision
- adapter id and revision
- tokenizer source and revision
- spec name
- training stage (`msm`, `aft`, `msm_aft_cot`, `msm_aft_no_cot`)
- source URL and paper repo commit used for procedure reference
