# CI Width Planning

Target full CI width: `0.080`

Conservative single-outcome Bernoulli prompt-cluster count: `601`

Conservative two-component SSEI prompt-cluster count: `1201`

Worst-case paired-delta prompt-cluster count: `2401`

| suite | policy | metric | current clusters | sd | required clusters |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  | No pilot deltas available yet. |

The Bernoulli count assumes the maximum single binary-outcome standard deviation of 0.5. The SSEI count assumes independent safety and capability Bernoulli components and is the minimum confirmatory powered-run planning target. The max paired-delta count is a worst-case bound for paired deltas that may be overly conservative for deterministic LLM prompt clusters. Pilot estimates use prompt-cluster deltas from generations.jsonl and should be treated as planning guidance, not final inference.
