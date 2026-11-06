# Replay transport repair

The first repeated replay campaign stopped after six complete 64-response cases.
The sixth, GPU-gated removal case, had one failed hint POST with
`ServerDisconnectedError`. All 384 generated responses matched the frozen token
trace; the RPC failure remains a real measurement failure. Source/outputs/logs and
the failed status are preserved in `output/wait-qualification-20261105/stage1-replay/`.

The failed POST began immediately after the preceding model response ended,
before the next model request. The server continued generating and had no matching
engine failure. The replay adapter used aiohttp's default persistent connection
pool; the maintained live adapters deliberately use `TCPConnector(force_close=True)`
to avoid stale keep-alive reuse. The observed disconnect is consistent with that
race, although no packet trace proves its exact transport cause.

Repair: use the same fresh-socket policy as the maintained live adapters. No
request retries, increased deadlines, ignored errors, policy tuning or change to
the frozen prompt/output/delay trace. The accompanying trace-sampling change only
selects diagnostic log keys and remains disabled in primary timing runs.

Run a complete fresh set of three rotated repetitions for all four replay
conditions with this corrected adapter. The incomplete old set remains exploratory
and is not pooled into the fresh set's interval. This replacement is justified by
a transport defect, not by an unfavorable performance outcome. It adds no open-
ended stopping rule. A further failure gets diagnosis and retained evidence.
