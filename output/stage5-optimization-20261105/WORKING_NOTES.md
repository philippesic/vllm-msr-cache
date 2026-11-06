# Active Stage 5 work (not a final result)

User authorized exhaustive sustained optimization across all seven categories. Do not
hand back at a checkpoint. No subagent authorization. No commits/push/deployment.

Preexisting tree was extensively dirty. Full source-before.tar.gz and preexisting.patch
are preserved here. Our changes: new semantic_offload/arc_optimization.py; opt-in
constructor/config in manager.py/spec.py; optional ARCAdvice and reset cleanup in
connector.py. New tests/test_arc_optimization.py. Client knobs and mixed replay in
benchmarks/serving_suite/repo_workload.py and replay_workload.py. Defaults unchanged.
New reusable research modules: stage5_optimization.py (prepare/run tranches),
stage5_report.py, arc_optimization_profile.py, hint_cost_profile.py.
Docs STATUS/product plan active; evidence protocol/category ledger created.

Validation: 83 initial focused checks, 133 bench checks, 126 post-timing/reset checks
passed. Differential fast eviction exercises 1000 random states x 5 operations with
atomic-failure/state equivalence. Calibration complete and offline verified:
wait 105.447 s, linear-wait 104.888 s, ARC 110.430 s; one exploratory pass, identical
inputs/outputs, actual CPU eviction competition. No fast-eviction serving win yet.
CPU hint path median ~0.194 ms at512 keys, digest ~0.579 ms at8192 tokens. Hint RTT
~0.113 seconds mainly before scheduler handling. Isolated measurements are not
critical-path savings. Protected-prefix eviction microbenchmark added but not run.

ACTIVE GPU: exec session 98681 runs policy-round1; holds GPU and legacy reservations
across both eight-arm cells. Log policy-round1.log; first repository then ample.
Do not edit source_identity-covered Python (semantic_offload, tests, serving_suite,
all vendor Python, helpers) or wait_hint_qualification.py while running. New research
modules not imported in run are outside identity, but don't edit stage5_optimization.py
since specs pin it as external input. Calibration exec 10549 finished, tests finished.

First repository partials: ARC109.848, frequent104.538, pressure95105.625,
cooldown2107.035, wait105.093, recent109.788; prefix50 and boost50 pending.
Recent refreshes only470 keys vs wait117421, mostly actionable entries are T2.
Cooldown recomputes more tokens than wait. No statistical winner from single pass.

Prepared NEXT: category-round2-spec.json, sixteen repository arms: controls plus
boost25,prefix25,prefix128,suffix-order,budget512,budget1024,cadence025,cadence2,
cadence5,single,gpu-gate,gpu-bridge,cooldown5,pressure75. Not launched yet.
Continue with adverse/mixed/uneven and tight-document/live followups, plumbing controls,
scheduling 4/8/12 sequences,512/1024/2048 batch,0/2/8/16sec continuation credit with
matched ARC/wait. Every category needs concrete alternatives + evidence-driven
followup before closure; current initial panel is NOT exhaustion. Repeated finalists
3 or6 fixed repetitions on development matrix, then operational/source/config freeze.
Fresh generalization is Stage6, not this development evidence.

Potential repair before operational REPLAY: wait_hint_qualification.offline loops all
successful requests and indexes source_id. cancel_probe writes successful wave=-1
row without source_id. Must exclude explicitly operational probe rows, test ordinary
replay mismatches still fail, then use new source freeze. Or run final operational
LIVE workload without diagnostic_replay (existing path works). Audit-off tier=agent
will fail native receipt accounting; compare audit overhead in a separate matched
operational/live cell with repository_workload=True, both arms same tier/audit mode.
The existing stage4-minimal-performance/spec.json demonstrates supported audit-off
operational plan. Never silently weaken active native accounting to get a pass.

Live specs reused by prepare may contain input_files hashes for historical current
repo source. Refresh hashes for current repo inputs before measurement, preserve
absolute independent public source pins, label all old held-out corpora development.
Some native command argv paths/expected files must be inspected before live runs.

Final: run affected tests + bench-tests + check-light, verify all completed/failed
artifacts, clear owned processes/disposable files (read storage-steward skill), update
category exhaustion ledger with honest unresolved scopes, retain selected default-off
config/source and findings. Render final comparison chart via T3 html_preview then
html_render (tools discovered mcp__t3_code__html_preview/html_render); no server.

Additional measured timing opportunity: initial-hint-readiness.json groups 48 wait
windows in calibration. First hint sees full CPU prefix in32, zero in5;7 windows
later expose more ready keys. If one-shot/slow cadence loses, a bounded initial
readiness retry (e.g.0.25s for <=2s, then slow cadence) is a credible follow-up before
calling timing exhausted. No future-return information needed; use actual ACKs.
Prepared scheduling-* specs (4/6/12seq,512/2048batch,0/2/16s credit), all tight matched
ARC/wait; plumbing-tight spec five arms ARC/wait/linear ARC/linear wait/noop.
Not launched. First pressure panel complete: no >2% candidate point improvement.
Ample panel currently active after pressure panel. Read policy-round1.log.

CORRECTION to initial readiness note: all5 first-zero windows are expired hints
(nomination_deadline_elapsed), NOT CPU availability misses. Only2 other windows
have later higher readiness. Prioritize constant hint window1500/3000ms and bounded
retry on expiry; do not claim a broad pending-store bottleneck from these counts.
Prepared timing-window-spec.json: ARC/wait/window250/window1500/window3000/tool-only
(no queue refresh). Not launched. Add complementary queue-only after current source
freeze: plan hint_during_wait=False suppresses periodic tool/sleep refresh (keep initial
hint and queue refresh), defaultTrue preserves behavior. This is a credible timing
corner before exhaustion. Source changes only between tranches + relevant tests.

CURRENT UPDATE: policy-round1 session98681 FINISHED, all16 cases passed and both
cells verified. No initial variant clears2% overwait; ample wait123.848s vsARC122.458s.
Round2 source boundary completed: balanced_order fixes repetition order (2-arm had
same order under old rotate+reverse; no repeats>1 were executed, so old measurements
unaffected). Tests verify two-arm alternation and 3-arm six-run position/precedence
balance. check_replay_row now handles identified wave=-1 cancel probes separately,
retaining ordinary replay mismatch rejection. Queue-only hint_during_wait=False added
(defaultTrue) to replay/live; explicit authored uneven-wait provenance supported.
Spec/connector reject optional ARC optimization on unsupported semantic policy.
137 focused checks pass. Protected eviction profile completed (cpu-profile-protected).

ACTIVE GPU NOW: exec session53911, category-round2.log, 17-arm
category-round2-extended-spec.json -> category-round2/00-repository. Preserve frozen
source; no edits to stage5_optimization.py or wait_hint_qualification.py or any
source_identity-covered Python until this tranche ends. Current source snapshots
include all recent changes. Existing unused scheduling/plumbing/timing specs had
stage5 module input hash updated before measurement; originals archived in
unexecuted-preparation-v1.tar.gz. timing-window spec pending, as is scheduling and
plumbing. Queue-only included in round2 (17th arm); no need to add it again.
Initial readiness correction stands: all5 zero-first counts were EXPIRED, not ready0.

Credible next refinements if round2 budget/cadence loses: current session_budget is
cumulative across an entire generation (all4 turns). Test a budget_scope='wait'
variant keyed additionally by hint.owner_id, so each real tool-wait gets a fresh
bounded quota; default session scope preserves existing screens. This can protect
legitimate returns without indefinitely promoting an abandoned future. Current
max_state is bounded LRU history, not a forever-per-identity quota guarantee.

Need actual abandoned-advice stress before closing budget/fairness category. Proposed
replay-only diagnostic: selected sessions, after completing all ordinary requests,
spawn background hint publishers for their last known completed prefix and an
explicit abandoned owner; no further request is ever offered. Parent cancels these
owned hint tasks when all real session tasks finish. Expected request count and
work remain identical across arms, and no future info is passed to policy. Explicit
workflow events/provenance identify false-positive hints; check agent_report accepts
unmatched hint owners (do not weaken accounting). Keep timing/replay diagnostics
separate from live quality. Corpus currently has no such behavior; implement only
at a source boundary, with tests, if needed to close the planned abandoned category.

capacity-headroom-spec.json prepared: same8-sequence repository replay,3.5GiB CPU
instead of2.625GiB, ARC/wait. Unequal-resource reference only, not a selectable policy
win. Current ample screen also lowered concurrency4 so cannot serve as matched
throughput headroom reference. At next GPU-free boundary run existing
wait_opportunity_report.case_metrics / returning_transcript_floor on completed cases
for remaining recomputation. Do not convert token headroom linearly into wall-time
headroom; no isolated GPU prefill-cost proof from these counters.
Makefile now includes new ARC tests in default focused test and new ordering/replay
contracts in bench-tests (Makefile is not executed in the current GPU replay).

Round2 GPU bridge first point104.086s vswait106.266 (>2% apparent gain), BUT
activation receipt gpu-bridge-screen-activation.json:202 GPU lookups, ZERO extra
suffix calls/keys. Do not promote or attribute speed to bridge. Retargeted
bridge-opportunity-spec.json (CPU1.3125GiB, GPU2560 blocks; CPU smaller than GPU)
ARC/wait/bridge one pass declared, not launched. This honors inactive-screen rule.
Other partials: cadence2 105.657s/338hints, cadence5 106.723s/149hints vswait633hints;
budget1024 110.504s, worse recomputation60954 vswait54938; cooldown5 105.676s;
gpu-gate105.735s; fast cadence025105.907s/1945hints. No established winner.

CURRENT GPU: exec session24058 runs operating-round3.log. Specs: plumbing-tight,
8 matched scheduling pairs (batch2048/512, seq4/6/12, credit0/2/16), bridge-opportunity,
capacity-headroom.26 cases total. Phase2 DONE17/17 verified; all36 GPU cases to date
(calibration3 + first panel16 + second17) completed, no established new winner.

Source boundary changes now frozen: AdviceOptions.budget_scope ('session' default,
'wait' adds owner_id to budget identity); replay abandoned_hint source flag spawns
owned background future-owner publications after selected real sessions finish,
parent cancels/reaps all at normal workload completion. No abandonment flag reaches
policy. Tests exercise task ownership and unchanged request count. Live optional
hint_coalesce_initial suppresses immediate on-start duplicate after pre-tool advice.
Replay now removes inherited frozen payload priority before optional recomputation,
so credit0 really means FCFS.140 focused tests +138 bench tests passed (priority repair
covered by latter). New reusable generator stage5_followups.py produced followup-plans:
abandoned9 arms, live-publication4 arms, pressure-repeat3armsx3, timing7 arms, audit3.
All pending, NOT in operating-round3. No Python edits while current tranche runs.

At next source boundary, refine hint_coalesce_initial before its first live use:
currently sleeps full interval even after waiting for tool semaphore. Instead compute
remaining interval since last successful *full-prefix* acknowledgement; partial,
expired/error/cancelled or old ACKs should allow immediate retry. Add pure helper/tests.
This is not used by current26 cases. Live 'during-tool' existing mode directly tests
publication/tool overlap; implement replay overlap only if live evidence warrants it.

Completed-work-accounting.json: returning transcript floor14815 tokens; currentwait
remaining recomputation2192–2560 vs ARC~11120; ample cell exactly0 above floor.
Do not convert token fractions to wall-time bounds. Matched8seq extraCPU reference
is queued to measure conditional headroom without confusing concurrency.
Pressure75 point104.162s vswait106.266; same work count54954 vs54938, needs repeats.
GPUbridge inactive despite apparent2% gain; explicit retargeted cell queued.
Decision rule for repeats added to protocol before repeats: primary point>=2% and
95% interval>1, adverse lower throughput bound>=.98, group completion upper<=1.05;
all development conditional, fresh Stage6 still needed. No pooling selecting screens.

Status-turn continuation: user status answered in commentary; sustained search still
active, no final handback. Operating-round3 session24058 remains running. No source
Python changed during tranche. First tight5 passed: ARC181.325, linearARC184.708,
wait178.331, noop185.302, linearWait176.238 seconds. Tight work much larger than
primary: ARC183082 prefill vswait135999; final-episode load and all-transfer bytes
have different meanings under preemption. No new throughput winner established.
Batch2048 ARC185.878/wait182.725; batch512 ARC185.311/wait176.249.
Sequences4 ARC208.074/wait178.299. Cross-setting comparisons are single screening
points against earlier default, not paired repeated evidence. Keep only promising
settings for interleaved comparison against default. Rest of26 underway.
EXHAUSTION.md updated with second-panel findings; progress-report.json recomputed
from available cases. No source changes, tests, or publication since prior note.

At next boundary also enhance stage5_report with per-cohort session completion
from workflow start/end and corpus publish_hints assignments (including ARC controls
using same designated cohorts); ratio candidate/baseline, upper95% guard1.05.
Do not count abandoned publishers as offered requests or extend their own completed
session latency. Test synthetic aggregation. Existing agent report only all-session
latency and replay request identities omit hint_group, so corpus is authoritative.

Remaining concrete ideas to decide at next boundary: live publication overlap that
joins the initial RPC before submitting the next model request (unlike current
during-tool which can cancel initial RPC when a short tool ends); shared initial
owned task must be shielded from refresh-task cancellation and reaped on failures.
Coalescing freshness helper still required before first live coalesced test.
Consider matched ARC/wait prompt-only storage diagnostic (offload_prompt_only),
which trades decode-tail reuse against storage bandwidth/capacity. Existing nearby
qualification docs have no corresponding measured ablation; distinguish this
common offloading change from retention-policy gain. Do not blindly implement a
new mechanism if live evidence or actual overhead makes it speculative.
Need uneven-wait and live tight-document candidate panels; pending plans currently
cover mixed-abandoned, timing, live-repository, pressure3repeats, audit operational.

SOURCE BOUNDARY / ROUND4 CURRENT:
Operating session24058 FINISHED exit1,19 successful cases then unsupported credit16
failed before workload requests; cleanup verified and failed attempt retained/offline
verified. Gateway onlysupports0/2000/8000 (not a numeric max range). Did NOT broaden
contract. Added stage5.preflight gateway validation before model load across all
spec conditions. Fresh8000 reference separately declared.55 successful GPU cases
campaign total plus1 invalid-setting failure. priority-setting-failure.json records.
Operating remaining data: seq6 ARC201.339/wait182.860; seq12 ARC183.293/wait180.861;
credit0 ARC184.589/wait175.680; credit2 ARC184.623/wait180.736. No new winner.
Bridge/capacity never ran in operating-round3 because fail-stops dispatch.

New source now FROZEN in active ROUND4. repo_workload.hint_refresh_delay delays only
latest successful full-prefix fresh ACK; expired/partial/cancel/error retries
immediately. hint_overlap_initial (before-tool mode) creates initial owned RPC task,
shields it from pending/on-running callback cancellation, joins it before the next
model request and reaps it on exception/cancel. during-tool original also available.
announce now collects owned cancelled receipts in finally. Live five-arm panel tests
ARC/wait/during-tool/coalesced-client/joined-overlap. No production defaults changed.
Tests:54 focused owner/coalescing/ARC tests; full bench143 and default focused86 PASS.
Sessions2319,14109,2272 all completed/acknowledged.

stage5_report now computes per-cohort mean session completion using corpus designated
hinted/unhinted groups even in ARC control. Rejects incomplete/duplicate boundaries;
ignores abandoned publication events. Candidate/baseline completion ratios and paired
CIs added to comparisons; throughput ratios remain baseline/candidate. Synthetic
boundary test passes. progress-report.json recomputed successfully all55passed/1fail.

stage5_followups generator extended; NEW followup-plans-v2/ is current. Original
followup-plans/ is unexecuted/superseded and has stale source pins; do not launch it.
V2 includes: abandoned9, live-publication5, pressure-repeat9, timing7,audit3,uneven6,
documents5,tight-cadence4,prompt-storage4,combination6,default-reference2,bridge3,
capacity2. Combination is linear eviction+batch512+FCFS with contemporaneous
individuals and ARC/wait controls. Prompt-storage matches ARC/wait and explicit
nohint alias configuration; shared offloading change, not retention attribution.

CURRENT GPU exec session8088:
python -m benchmarks.research_checks.stage5_optimization run --spec
 followup-plans-v2/default-reference-spec.json bridge-spec.json capacity-spec.json
 live-publication-spec.json abandoned-spec.json
 --output output/stage5-optimization-20261105/followup-round4
(log followup-round4.log).21 cases. No source edits while running. Executed-source
pin includes stage5_optimization.py and stage5_followups.py, plus suite/semantic/
tests/harness/tools/vendor Python. Docs safe. No active functions.exec monitor cell.

Additional saved planning evidence: abandoned-opportunity-planning.json shows ordinary
wait selected owners finish7.5–23.4s before batch end with1–8 later requests, so
abandoned stress has plausible competition. Actual mixed abandoned panel still
must prove refreshed phantom keys and later work; planning receipt is not activation.
After round4 inspect all live quality/outputs, bridge activation and mixed fairness,
then run remaining fixed repeat/timing/uneven/document/plumbing/combination panels
adaptively. Need final repeated selected matrix (or unchanged baseline if no candidate
survives), functional unforced cache comparison and operational checks. Existing
runtime-probe-long/spec.json can be cloned with stage5 workload label and candidate
config options; runner routes plan.runtime_probe to wait_hint_runtime_probe.run_wave.
Do not treat forced replay output identity as KV correctness.

Round4 intermediate findings (source still frozen; session8088 active):
Fresh default tight reference ARC184.015/wait175.122 confirms small prior scheduling
points (~175–176s) do not establish improvement. GPU bridge retarget ARC131.669/
wait131.912/bridge133.907; STILL ZERO extra suffix calls/keys despite592 GPU lookups.
Receipt gpu-bridge-retarget-activation.json. Do not reject generally or attribute
its timing to bridge. Native store path examined at offloading/scheduler.py ~1030;
incremental stores plus root-priority touch suggest few holes but NO invariant proof.
Capacity same8sequences CPU3.5GiB ARC101.999/wait102.274, both52378prefill and no
observed eviction competition. Unequal-resource reference only; supports modest
conditional remaining headroom, not a hard wall-time bound or selectable policy.

Live-publication five arms completed all real tools and16/16 narrow test-outcome
checks EACH. Receipt live-publication-screen.json: wait114.625s/12194outputs/490hints,
joinedOverlap115.984/12042/439, ARC124.074/12164/0,
coalesced114.765/12146/447, duringTool115.988/12071/441.
All48windows had advance ACK for wait/coalesced;22joined and21during. All decoded
outputs natural and differ, no equal-work causal speed claim. No new >2% gain.
Abandoned panel active; first pressure75=105.709, waitBudget1024=109.066 seconds.
Need final actual phantom activation, later offered work and cohort fairness.
Latest user supplied replacement AGENTS preferences, adds Tailscale SSH aliases;
no cancellation/objective change and no new delegation authorization. Continue.

ROUND4 DONE21/21 passed+offline verified; session8088 complete/acknowledged.76 passed
GPU cases total +1 unsupported-setting failure. Abandoned diagnostic ACTUAL phantom
activation: unchangedwait46 publications,4future owners,10233acknowledged keyrefreshes.
ARC111.172s,wait105.457,pressure75105.709,boost50106.646,cadence5107.562,
prefix75109.099,waitBudget512107.586/1024109.066/2048107.320.
Wait mean completion hinted95.538/unhinted90.179 vs ARC97.669/93.750 seconds.
All caps reduced phantom touches but lost ordinary reuse; no gain overwait. See
abandoned-screen.json for per-cohort/per-variant details; single-pass not CI acceptance.

CURRENT GPU exec session66145 (round5).
Round5 command: stage5_optimization run --spec followup-plans-v2/pressure-repeat-spec.json
 timing-spec.json audit-spec.json --output output/stage5-optimization-20261105/timing-repeat-round5
log timing-repeat-round5.log.19 cases: fixed3repeats ARC/wait/pressure75, timing7,
audit3 operational. Same source as round4; NO EDITS during run. No active exec
monitor at launch. Remaining V2 plans after round5: uneven6,documents5,tight-cadence4,
prompt-storage4,combination6 (25 cases), then selected repeated matrix/functional
validation/final evidence/cleanup. Do not blindly run superseded v1 plans.

ROUND5 FINISHED/FAILED, new ROUND6 active:
Session66145 complete exit1. Fixed pressure repeat9cases PASSED/offlineverified.
Then first timing armwindow3000 completed64normal requests but all its hint inputs
were invalid HTTP400 (declared_wait_ms outside frozen API {0,50,250,750}); failed
wait_hint_failure and dispatch stopped/cleanupverified. This is NOT a timing result.
Nomination-window-failure.json retains failure; no hidden retry/relabeling.85passed
GPUcases total +2invalid-setting failures (credit16zero workload, window3000full
workload but invalidhints). Audit and rest of old timing did not run.

Repeated primary results pressure-repeat-assessment.json:
ARC->wait throughput1.061487 CI95[1.049916,1.073186]; ARC->pressure751.061342
CI[1.043727,1.079255]; wait->pressure750.999864 CI[.978523,1.021669]. No pressure
promotion, no pooling with selecting screen. Default passed3 fixed repetitions.

Source boundary repairs/extensions (NOW FROZEN):
- Stage5preflight calls KnownPrefixWaitHint.from_mapping with selected declared
window; prevents unsupported hints before modelstartup. Also catches prompt-only
storage combined with completed-transcript publication or mismatched client scope.
- New default-off wait_hint_cpu_nomination_grace_ms integer0..3000. Constructor
requires native ARC CPU-only ifnonzero. Adds grace to CPU-advice deadline ONLY for
positive declared waits. Zero wait remains expired; GPU staging deadlines/leases
notextended. Original strict client enum unchanged. Nonzero grace receipt explicit;
zero default retains original ackfields. Tests cover bounds,expiry,zero,nonCPU,
noGPUreservation/transfer/jobs andcorrectflags. New variantsgrace750/grace2250,
window50/window250; removed unsupported window1500/window3000 variants fromgenerator.
- publish_hint now calls known_prefix_tokens(previous,completed_transcript=True).
Plan hint_completed_transcript=False explicitly uses previous PROMPT ONLY in both
published hint and continuation metadata; replay and live defaults unchanged.
This is necessary for prompt-only storage because native completed_transcript
publication explicitlyrequires offload_prompt_only=False. Prompt storage panelnow5
arms ARC/wait/arc-prompt-only/wait-prompt-scope(fullstore)/wait-prompt-only. Both
prompt-scope arms disablecompletedtranscriptpublication; separates storagefromscope.
- generatorwrite calls preflight after custom armsadded. NEW followup-plans-v3
current, oldv1/v2 unexecuted plans nowstalesourcepins. DoNOT launchthem.
- First grace CPUtest hadfixturemissingpreservation_events; correctedfixture.
Finaldefaultfocused97 PASS; bench149 PASS. Sessions8545(failedfirsttest),83381,
87518 finished/acknowledged. No currenttestsprocesses.

CURRENT GPU session88659 runs timing-operational-round6.log:
stage5_optimization run --spec followup-plans-v3/timing-spec.json audit-spec.json
 uneven-spec.json --output output/stage5-optimization-20261105/timing-operational-round6
17cases(8timing+3auditoperational+6uneven). NO SOURCE EDITS whileactive. Noactive
execmonitorcellatthisnote. Mustinspectactualgraceactivation/nominationdeadlines,
auditoperationalabort/drain/recovery, and timingworkcounts. Then remainingv3panels:
documents5,tight-cadence4,prompt-storage5,combination6=20cases. Allprepared/pinned.
Then choosefinalrepeatedmatrix and functionalunforcedprobe, finalfreeze/evidence/
cleanup. DoNOTredo completedpressure-repeat9 orround4 simplybecausev3generator
alsoemittedthesespecs. Needfreshfixedcomparisonsforactualnewfinalists ifany.

Extra currentdocs: protocol/ledger stillactive notexhaustionclaim; docsSTATUS says
fixedrepeats/timingactive. T3charttoolmetadata discoveredfunctions tools named
mcp__t3_code__html_preview({html,appearance?,width?}) then
mcp__t3_code__html_render({html,height:preview.contentHeight,title}). Useforfinal
comparisonchart, noserver. CSS themevariables, transparentouterbody, nooutertitlecard.
Noagents authorized/spawned. No commits/push. User AGENTS updatejustaddsTailscaleSSH.

Round6 timing partial: tool-only102.825s/54666prefill/426hints vswait105.936/
55994/625 (~3% screen), cadence05=103.865/54570/1120,grace750103.904/54794/628,
grace2250104.738/55082/628,window250105.114/54602/633. Grace eliminates expired
nomination receipts (0 vswait21), but single-passwork/timingvariation remains.
NEW fixed3repeat spec created WITHOUT sourceedits: followup-plans-v3/
tool-only-repeat-spec.json (ARC/wait/tool-only9cases seed720). Run aftercurrent
round6, before deciding promotion/remainingconfirmation. Mustnotpoolselectingscreen.
No activeexecmonitorcell at thisnote;session88659 stillrunningwindow50.

Round6 intermediary: timing8PASS; tool-only3%point warrantsrepeats. Graceactivity
PROVED usingACKobserved_at_ns (scheduler now_ns; no decision_ns field):grace750
18lateACKs/3262refreshedkeys,grace225015/2738, bothzeroGPUreserved/speculativebytes.
Receiptgrace-activation.json. Auditoperational3PASS inclactualbackendabort,drain,
recovery:ARC110.675s,wait104.711,audit-off102.668(~2%point). Cachehistoriesmissing
bydesignwithauditoff; doNOTattribute cacheworkfromabsence.
Newdeclaredaudit-repeat-spec.json:3repetitions4armsARC/wait/ARC-audit-off/wait-audit-off
(12cases) to separate commonobservabilitycostfromretentiongain. Unexecuted.
Newtiming-finalist-repeat-spec.json:3repetitions4armsARC/wait/tool-only/
lean-combination(tool-only+grace750+linearARC),12cases. This SUPERSEDESunexecuted
9case tool-only-repeat-spec.json BEFORE anyrepeatwasexecuted; actualactivegrace
plusmathematicallyequivalenteviction justifyonecombination. No screenpooling.
Bothnewspecscreatedwithunchangedpinnedsource;canrunnextwithoutsourceboundaryedits.
Round6session88659currentlyuneven6underway(firstcooldown5=95.312s), noactiveexec
monitorcellafterthisnote. Stillneedremaining20v3diagnosticpanelsandfinalfunctional/
selectedmatrix. Keepallfailures/unusedspecsidentifiable. Allsourcefrozenstill.

ROUND6 DONE17/17passed+verified;session88659complete.102passedGPUcases+2invalid
configurationfailures total. Unevenlastresults:wait93.030,pressure7590.983,
ARC94.628,boost5093.029,cooldown595.312 (cadence2inreport). Pressurepoint2.25%
better onUNEVENTOOLWAITS despiteprimaryrepeatsnogain. Separatefixed3repeat spec
uneven-pressure-repeat-spec.json9cases seed723 created, unexecuted; conditional
interaction only/no pooling. This closes a credible newregime signal, notextending
primaryrepeatuntilwin. Audit-repeat12alsoawaitsrun. Currentprogressreportrecomputed.

CURRENT GPU session13180:
stage5_optimization run --spec followup-plans-v3/timing-finalist-repeat-spec.json
 documents-spec.json tight-cadence-spec.json --output
 output/stage5-optimization-20261105/finalists-documents-round7
log finalists-documents-round7.log.21cases(12fixedfinalistrepeats+5livedocuments+
4tightcadence). AllsamecurrentfrozenPython. NO SOURCE EDITS. Noactiveexecmonitorcell
atthisnote. Remainingunexecutedcurrentplansafterround7: audit-repeat12,
uneven-pressure-repeat9,prompt-storage5,combination6 (32cases); thenchosenfinal
checks/fairness/operational-unforcedfunctionalprobe, freeze/evidence/cleanup.
DoNOTlaunchtool-only-repeat(old9supersededbeforeexecution),pressure-repeat(primary
alreadydonev2), oldv1/v2specsstalepins, orpriorpendingoriginaltiminginvalidwindows.

NEW USER GIT AUTHORIZATION supersedes earlierlocal-onlyinstructions:
ReplacementAGENTS explicitlyrequirescoherentmilestonecommitsandpushtoestablished
user-ownedremote. No explicitseparatenopushremains (oldAGENTSpushonlyrequestedwas
replaced; oldprotocolwasourderivedscope, nowupdated). Needcommit/pushatnextcompleted
batch andhandoff. NoPRrequested. Currentbranchmain tracksorigin/main,
remote git@github.com:philippesic/vllm-msr-cache.git, remoteHEADmatcheslocalHEAD at
read. A separateuserworkflowcommittedpolicy/bookkeeping e5437b5 whileweworked;
it ONLY changedAGENTS/.gitignore/docsGitpolicy andauthoredoutputrecords, leaving
existingresearchsourcesdirty. DoNOTrevertthosechanges. It nowtracksour authored
WORKING_NOTES.md andother top-leveloutputMarkdown; rawruns remainignored.

Gitfoundationissue:135pre-Stage5dirty/untrackedfiles(2.43MB) constitutecompleted
research/qualificationdependenciesthatStage5imports. Foundationmanifestcomputed
frompreexisting-status.txt andsource-before.tar.gz, savedfoundation-checkpoint-manifest.json.
NoGitmutationyet;indexcleanwhenchecked. Planannouncedtouser:checkpointexactarchived
foundationseparately,thenStage5delta. Allpreexistingtaskentriescompletedexceptpaused
AWQwhosefilesarealreadyinHEAD/unchanged. DoNOTsilentlycombineunrelatedactivework.
Useexplicit135manifestandGitblob/indexstagingfromtar (workingfilesuntouched),
verifyindexclean/HEADandmanifestbeforecommit. PreservecurrentHEADpolicy/output
bookkeeping (notinpreexistingdirtymanifest). ThenstageonlyStage5paths forowncommit,
pushexistingmainnonforce;read/checkremoteadvanceandcoordinateifneeded.
Foundationpathsmax479KBsource;no bulkoutput/model/cacheincluded. Snapshotexactness
isretentionverification,notfreshbaselinequalification. AllStage5testsrunoncurrenttree.

Round7 primaryfinalist12DONE/pass. timing-finalist-repeat-assessment.json:
wait->tool-only1.01205 CI[.96131,1.06548];wait->lean-combination1.01536
CI[.96365,1.06986]. Neithermeets2%point/usefulgain+CIcriterion; no promotion.
Fourarm3repeatorderrotatedbutnotcompletefourpositioncycle, disclosed. Currentdocs
panelstillrunning (ARC172.619s completed), then4tightcadence. Session13180active,
noexecmonitorcellatthisnote. Needremainingaudit12/unevenrepeat9/promptstorage5/
combination6 andfinalunforcedfunctional/selectedchecks. NewGitworkmustnotmodify
sourcewhilebatchfrozen. Source_identity excludesGitHEAD/index; stillwaitboundary
beforeactualcommitmilestoneasannounced. Latestuseronlyworkflowinstructions, notstop.

LATEST USER STEERING (takes precedence over old exhaustive pending lists):
User asked status, answered117validGPUcases then. User thenexplicitlysaid finisha
pass throughremainingsections but goLESScrazyonremainingworkload andCHASEEVICTION:
CPUtests muchfasterneedrealservingcost/bottleneckexplanation orrepresentativeworkload
reproduction. Acknowledged. DoNOTblindlylaunchall32pendingrepeats. Trimunexecuted
broadpanels, prioritizeactualevictionprofiling. Planupdatedaccordingly.
Currentround7session13180 finishinglasttightARC (allother20casescomplete); NOactive
execmonitorcell. Tightcadence:cadence5177.674/wait173.943/cadence2184.052,ARCpending.
Documentsall5passed:ARC172.619/wait160.411/pressure75161.155/cadence2170.502/
boost50176.186. Needretainnaturaloutputcounts beforeinterpretingdocumenttimings.
AtboundarycheckpointGitfoundation+Stage5asannounced, thenimplementtargetedprofiler.

Proposed targeted eviction approach (notimplemented): opt-in per-case profiling
via runner planarc_eviction_profile/envSEMANTIC_ARC_EVICTION_PROFILE_DIR. Nativeand
linearARC body wall/threadCPU timings, n/protected/queue sizes/success percall.
Bounded snapshots of actual byteskeys/refcounts/queue+ghost order/target before
representative calls; offline replaynativevslinearidenticalevictions and counted
queuevisits (countingseparatefromtiming). Bodytimeexcludes capture/log overhead;
measureobserveroverheadseparately, noCUDAfences. Aggregate storetimeinclusive
separately so doNOTsum nestedintervals. Existing managerprepare_store already has
policy_started/ended wall+threadCPU timestamps. ManagerdoesNOTcurrentlyoverride
on_request_finished; nativehookguaranteesall submit-sidepreparecallsdone bythen,
so canflush compact profiler summary atrequestfinish (one perrequest) andbuffered
percallrecords. Lastsummarymustcoverallrawcalls. Outputunderownedcase, sourcepinned.
Use newreusable benchmarks/research_checks/arc_serving_profile.py, optional import
fromPreservationmanageronlywhenenabled. Env is normallyscrubbed inrunner; explicitly
setper-case onlywhenflagtrue. Source_identitydoesnotincludeallresearchmodules, so
pinnewprofilerasexternalinput inprofile specs. No genericSEMANTIC_PROFILE_DIR:
existingComponentProfiler wrapsotherphasesandatexitcanbe killed; isolateeviction.
Potential small profiling matrix: primary/tight matchednativeARC/wait andlinear
controls; include real longdocument pair or larger realprefillbatch ONLY if activity
justifies. Actualcallshape/bodyshare guidesfollow-ups; don'tmanufacturearbitrary
protectedstatesandcallthemreal. Sumhostbodytimeis directtimeatstake, notastrict
endtoendbound becausescheduling/GPUoverlapcanchange. Needunprofiledvalidationif
profilerfindsmeaningfulopportunity; otherwiseevidence-backedbottleneckexplanation.
Remainingsectionscouldbeonepromptstorage5/combination6 pass andsmallmatchedaudit
control ratherthan12+9morebroadrepeats. No finalstoppingclaimyet.


Checkpoint completed: exact pre-Stage5 foundation committed separately, then
Stage5 sources/results committed; both pushed non-force to established origin/main.
Portable check31pass; profiler focused100pass; bench149pass. New profiler opt-in
in manager and runner; exact bytes snapshots + body timing + request-finish flush.
Current GPU session47102: eviction-round8 sixcases repository/tight ARC/wait/linearwait.
Do not edit executed source during it. Profiler replay module pinned external.
Remaining-plans prepared: prompt-storage5, combination3, matched audit-off2,
functional3 (three waves) with baselineARC/wait and linear+grace750wait.
Latest user narrowing means skip broad audit12/uneven9 repeats. Need inspect profiler
results and decide credible eviction follow-up, finish remaining pass, final evidence,
cleanup and coherent push. No promotion yet. Skill paths moved into p-stack checkout;
recap and storage-steward re-read there. Storage cleanup skill announced. Current
campaign1.9GiB, plenty space; preserve retained evidence/source recovery backup.


Eviction round8 DONE all6pass/offlineverified;129validtotal.37real beforestates
replayed identically; native/linear5-7x forlargercalls, n1linearslightlyslower.
PrimarynativeARC9.97ms/111.13s;wait4.75ms/105.60s;linearwait4.16ms/103.84s.
TightnativeARC82.15ms/184.70s;wait45.57ms/184.97s;linearwait16.05ms/177.22s.
These arePROFILEDdiagnostics notpromotiontimings;bodydiffcannotexplain7secwallgap.
Allrawcalls reconcile; zeroevictionfailures. Evidenceeviction-serving-profile.json
andEVICTION.md. Nativeprimarywait186/213callsevictedoneblock. Actualstate39victims
3074visits->101linear(~7x),64victims2017->64(~5x), soCPUleadisrealbuttinyoverall.
Milestone3756397committed/pushed. Newstage5_closeout.pyreconstructsfinalrecipes;
checkedplan/corpus/conditions/orderidenticaltopredeclaredremainingplans.
CurrentGPU session48494:closeout-round9fivepanels=15cases:
long-batch-profile2(real livedocuments,batch4096),promptstorage5,combo3,auditcontrol2,
functional3(3waves each). No executed-sourceeditsuntilboundary. Newgeneratoroutside
identityandunusedinround8wasaddedwithoutchangingexecutedinputs; nowpinnedincaseinputs.
LongbatchnativeARCcompleted175.036s,465calls,159.423msbody(maxvictims250),~0.091%wall.
LinearARCcontrolactive. Needoffline exactreplaylongbatchafterthissuite(orquietboundary),
assessremainingcases andanycrediblefollowup; finalCPU/checklight,compactresults,
README/finalexhaustion/status+history,chartviahtml_preview+html_render,cleanupinventory,
finalcommit/push. EVICTION.md stillmentionslongbatchpending;updateafterdata.


Closeout round9 performance panels DONE/pass: longdocs2,storage5,combined3,audit2;
141validtotalnow,2retainedinvalidconfigs. LongbatchARC175.036s/159.423msevict,
linearARC178.724s/17.859ms;465calls each,maxvictims250;output7240vs7243.
Tenadditionalexactstatesreplaypending (doafterGPUidleforquietCPUtiming).
StoragefullARC184.561,promptonlyARC183.921,wait172.717,promptscope189.134,
waitpromptonly177.086;nonewins. ComboARC184.712,wait180.114,combined178.268,
~1.04%pointbelow2%usefulmargin. Auditoffwait102.385,ARC108.491,actualabort/drain/
recoverypassedboth;onewave+probesnotfulloperationalqualification.
Session48494nowfinalfunctional3cases/3waves (ARC firstactive). No executedsourceedits.
check-light-final launched (31portabletests+compileall), noCPU/GPUperfclaimforthese
functionalchecks. CPUfocused100/bench149alreadypassedonunchangedexecutedcode;
newcloseoutgeneratorcheckedrecipesexactandrunningcases. Don'trepeatwithoutreason.
EvidenceREADME/EVICTION/EXHAUSTION/overview.html nowdrafted; README/selected-candidate
stillprovisionaluntilallfunctionalpass. EVICTIONlongbatchreplaypendingtextreplace.
Finaltasks:awaitfunctionalcompletion+offlineverify;quietofflineprofilefullrootto
singleeviction-serving-profile.json (47statesexpected), finalstage5_reportresults,
compactvalidation.jsonfunctionalchecks; finishREADMEandcleanfinalcategoryledger,
updateSTATUS+PROJECT_HISTORY+Stage5plancompletedunderusernarrowedscope, selected
candidatefreeze(noexperimentaloverrides), storageinventoryandownedprocess/lockcheck,
chartpreview/render, finalcoherentcommit/push. Existing sourcecheckpoint3756397
alreadyonorigin/main; currentuncommitteddeltaonlyevidence/worknotes/docs.


FINAL CLOSEOUT: all144validGPUcasespassed;2invalidsettingsretainedseparately.
Finalfunctional3cases/9wavesallpass,48assertions; actualCPUrestoreforbothhintarms,
freshcomputedentireprompt/greedyoutputagreement, malformedadvice, nativeabort/drain/
recoverypassed. ActualpinnedCPUbacking/load-storealias/bytegeometryverifiedfrom
allocationreceipts. Eightservingprofiles/47exactstatesallreplayequivalent;long197-
victimstate34380visits->368,~23xCPUgainbuttotalnativeeviction<0.1%workflow.
Quietofflineprofilereplaydone; finalresults.json146attempts(144valid+2invalid).
checklight31pass;focused100/bench149pass. Defaultselectedunchanged, noexperimental
promotion. DocsREADME/categoryledger/selectedcandidate/status/historyupdated.
GPUprocessesgone, bothlocksacquired/released;146emptyownedruntimecontainersremoved.
Retain~2.23GBsource/evidence/recovery;retention-inventory.jsonoutsideGitcontains
freebytesbefore883118055424/after883118628864/delta573440andretainedpaths.
T3html_previewfailedhostmissingbrowserlibraries;staticchartsourceoverview.html
saved, nohostpackagesinstalled. html_renderstilltobeattemptedforfinaldisplay.
Needfinaldocs/linkchecks,stageownfinalevidence/docs/worknotes,commit/pushnonforce,
verifycleantreeandremote,renderchart,finalself-containedhandback. Stage6pending;
do notrestartGPUwork. No pending toolcells orsessions exceptnone(allreportedcomplete).

Finalchartdisplayedwithhtml_renderafterhtml_previewreportedmissinghostbrowser
libraries. No systempackageinstallationneeded;previewimagevalidationunavailable,
staticSVGdata/sourcecheckedandrenderattachmentcreatedsuccessfully.
FinaldocslinkchecksandJSONcountchecks passed:33offlineverificationrecordscover146
attempts,144valid+2invalid;47stateequivalences,48functionalassertions. Cleanupremoved
onlyemptydirectories;sealedfilesuntouched. Stage6pending; no furtherGPUworkneeded.
