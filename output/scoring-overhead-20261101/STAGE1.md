# Stage 1: multi-token capture confirmation

The predeclared three-seed comparison failed the 1% throughput goal in both workloads. These results are retained unchanged while alternative callback timing and packed async scoring are screened. No commit/push.

| Workload | Arm | Output tok/s | Mean TTFT ms | p95 TTFT ms | Mean CPU s |
|---|---|---:|---:|---:|---:|
|single|arc|63.982|2089.0|4917.5|23.94|
|single|original|42.946|4209.6|10155.8|35.79|
|single|candidate|63.075|2042.8|4546.7|24.35|
|multi|arc|58.045|6923.2|15225.2|59.57|
|multi|original|42.284|9825.3|23465.1|81.55|
|multi|candidate|56.747|7017.9|15561.3|61.19|

Candidate: attention hook, multi_token capture, stride1; synchronous continuous dense mean scoring. Summary construction and cache updates batched. Equal 640 GPU/199 CPU KV blocks. Sampling changes score trajectories; separate common-key hold-last proxy top10 overlap is 63.1%, not cache-hit quality. ARC restored 1344 tokens in stress, candidate/original restored none. Scored-eviction activity and actual applied score updates are in confirmation-results.json.

Reproduce with confirm.py using exact package snapshots and source pins in confirmation-protocol.json. The initial screening original-source snapshot raced capture edits; this confirmation uses recovered exact pre-edit capture source.
