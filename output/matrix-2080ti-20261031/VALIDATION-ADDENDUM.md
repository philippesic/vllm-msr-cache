# Separate output-repeatability check

After observing primary single-agent seed2207 output differences, add exactly one ARC-only repeat of that workload/seed/configuration after the frozen primary, admission-stress and component matrices. Label it `repeatability-single-2207-arc`. Do not pool it into primary performance or replace any result. This is an assistant-selected correctness follow-up within the requested comprehensive verification.

The initial differing request had identical prompts, zero native GPU-prefix hits, zero CPU restores and zero CPU stores in both arms; its first different generated token was position58 (zero-based). Five subsequent agent turns inherit different generated history. Twelve independent background outputs matched. This rules out consuming incorrect restored CPU KV as the cause of that initial difference; it does not by itself prove the numerical/batching explanation or certify every later transfer.

Compare the repeat with the original ARC run and the original semantic run by request identity, prompt/output token lists and ordered key chains. Report either outcome. A non-reproduction does not prove a semantic defect; a baseline divergence demonstrates ordinary-run output instability for this test setup. No workload, memory, policy or math changes.
