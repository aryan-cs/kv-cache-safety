# Evidence-Gated Interpretation

The completed metrics support behavioral cache sensitivity only; selective degradation and causal safety-erasure claims are not supported. Evidence summaries: Safety degradation exceeds zero with a positive lower confidence bound. Best: system_leakage::sliding_window__budget128 estimate 0.429, 95% CI [0.429, 0.429], width 0.000.; No policy-level SSEI interval clears the configured threshold. Best: sliding_window__budget128 estimate 0.092, 95% CI [0.083, 0.101], width 0.018. Log-odds SSEI 1.392, 95% CI [0.321, 3.222], width 2.901.; Cross-family replication failed: no instruction-tuned family cleared the registered selectivity gates.; System-role restoration does not beat matched user-token controls. Best: system_leakage::kv_int4_sim leakage_avoidance_restoration_fraction system -0.000 95% CI [0.000, 0.000] versus user control -0.000 95% CI [0.000, 0.000]; margin 0.000, conservative 95% CI [0.000, 0.000] with width 0.000; system/control CI widths 0.000/0.000.

Only behavioral cache sensitivity passed. The supported claim is cache sensitivity without selective safety degradation or a causal cache-mediated safety mechanism.

Report the negative selective and causal controls, then either narrow the paper to a deployment robustness result or run additional powered diagnostics.
