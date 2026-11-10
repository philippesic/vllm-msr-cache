# 002: Small boundaries and one validator

Accepted. Keep contracts pure, concrete vLLM types behind `vllm_adapter`, and
benchmark execution outside production. Create further responsibilities only
when extraction needs them: retention/admission, prefetch state, scoring,
transfer and observation. No shared-state mixin recreation of the old connector.

One locked command runs locally, in the optional hook and in CI. Each adopted
rule has positive and negative controls in a versioned manifest. GPU and
benchmark-schema gates arrive with their implementations. Current native absence
is visible; new native material blocks until recipes are qualified. This step
does not invent a kernel or a benchmark solely to populate checks.

This implements the foundation of sections 5–6 of the external porting plan;
full responsibility, runtime-parity and native-build gates remain future work.
