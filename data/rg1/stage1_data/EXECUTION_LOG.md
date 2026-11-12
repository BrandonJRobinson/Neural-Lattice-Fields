# RG-1 Stage 1 execution log

- 2026-10-28 ~13:30 CET: first submission (ledger entry 1): arrays 381392-381395, 381418-381420; adjudication 381421. Authorization: AUTHORIZATION_RG1_STAGE1.json (user: "Authorize Stage 1").
- 13:31-13:33: 55 tasks (all of 381419 [P24, tasks 0-19], 381420 tasks 20-50 [P32], and tasks 59, 76, 78, 79) on hipster-cn011 were "CANCELLED by 0" (root/Slurm) about 2 min after start. They wrote no job logs and no runner failure records, so the batch step never ran. Only this user's jobs on cn011 were affected; the node was not drained. Cause not established; treated as a node-side start failure. No data affected: no chain had produced output.
- 13:39: unrun adjudication job 381421 cancelled by the user account (it would have run against incomplete data and double-counted reserved core-hours).
- submit_stage1.sh extended (release re-checksummed; PLAN.json and analysis unchanged, so the authorization still applies): retry while other jobs run (consumed + reserved accounting), --exclude, --only, --no-analysis.
- 13:41: retry (ledger entry 2): production tasks 0-50 resubmitted with the same seeds, excluding hipster-cn011, arrays 381474/381475 (538.5 core-h reserved; ceiling check 48.81 + 538.5 <= 600).
- Pending: tasks 59 (C12_repro chain 7), 76 (S24 chain 0), 78, 79 (S32) + adjudication, to be resubmitted once running controls free their reservations.
- 14:15: accounting corrected: capacity nodes have ThreadsPerCore=2, so Slurm reports AllocCPUS=2 for each single-core chain. The 600 ceiling and the plan are in PHYSICAL core-hours (single-core forecast); Slurm CPU-hours will read ~2x. submit/status scripts now convert by ThreadsPerCore (release re-checksummed; PLAN unchanged).
- 14:16: retry (ledger entry 3): controls 59, 76, 78, 79 + adjudication (afterany on all Stage 1 arrays), excluding cn011. Ceiling check 557.3 + 21.0 = 578.3 <= 600 physical core-hours (reservation basis).

## 2026-10-29T10:03:35Z Adjudication moved off the queue
Job 381532 (8 CPU x 1.5 h) was pending on Resources from submission to 2026-10-29 11:0x; its start estimate kept slipping (09:26, then 12:10). At the user's request it was cancelled before starting (0 core-h used), and the identical command (same release, sha-verified pins, same flags, --workers 8) was run on login node hipster-l0 under nice -n 10, writing to adjudication_local_20261029T100334Z/. Its CPU time is recorded in adjudication_local_20261029T100334Z/time.txt and counts toward the Stage 1 ceiling.
2026-10-29T10:23:32Z Local adjudication finished rc=0, wall 1152 s, child CPU 73m44s user + 9m19s sys = 1.38 core-h. Stage 1 total = 209.1 (Slurm physical) + 1.4 (login) = 210.5 of 600 core-h. Note: the adjudicator's accounting field reports 418.3 in Slurm CPU-hours (AllocCPUS counts both SMT threads). Outputs hashed in adjudication_local_20261029T100334Z/SHA256SUMS.txt.

## 2026-10-29T10:39:32Z Stage 2 submitted
User: "Authorize Stage 2". Release stage2_release_v1 (PLAN 17d79c35…), preflight PREFLIGHT_20261029T103607Z PASS. 148 tasks, 1,287 core-h of time limits (ceiling 1,400), excluding cn011 (Stage 1 cancellations) and cn013 (draining, Stage 1 NODE_FAIL). Ledger: stage2_data/SUBMISSION_LEDGER.json. Locks unchanged (c0535c02…).

## 2026-10-30T07:01:33Z Stage 2 retry after cn012 failure
User: "Resubmit now". Tasks 90-117 (Q32_lo chains 26-41, Q32_hi chains 0-11; 27 NODE_FAIL, task 111 CANCELLED by root, all within about 5 min on hipster-cn012, no captures) resubmitted with the same seeds and unchanged PLAN as array 382426, excluding cn011, cn012, cn013. Ceiling check 712.08 + 329.00 = 1,041.08 <= 1,400 physical core-hours (reservation basis). Locks unchanged (c0535c02...).

## 2026-10-30 Automatic retry rule (user request)
User asked that failed chains be resubmitted automatically as continuations on the same seed. phi4_local2 has no checkpoint or RNG-state output and the binary is pinned, so continuation is not available without changing the sampler; not done. Implemented the automatic part only: stage2_supervisor/auto_retry.py restarts failed chains from sweep 0 on the same seed (bit-identical, matching the plan's restart_policy) via submit_stage2.sh --retry, max 3 Slurm attempts per chain. Captured chains are never resubmitted whatever their QC verdict. The frozen release is unchanged.

## 2026-10-30T09:35:52Z Stage 2 automatic retry
Supervisor resubmitted tasks [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 146, 147] on the same seeds from sweep 0, excluding hipster-cn011,hipster-cn012,hipster-cn013. See stage2_data/SUBMISSION_LEDGER.json.

## 2026-10-30 Healthy-node rule (user request)
User: "Only healthy nodes should be allowed." auto_retry.py now excludes cn011-cn013, any node with a NODE_FAIL or root cancellation in Stage 2, and any node sinfo reports as not plainly up, on new submissions and on queued tasks. Tasks 18-31, 43-63, 146-147 were cancelled by root on cn012 at 09:32 UTC after about 2 min.

## 2026-10-31T02:59:02Z Stage 2 adjudication
All 148 Stage 2 chains captured, QC PASS. analyze_stage2.sh run on login node hipster-l2 under nice -n 10, 8 workers: rc=0, wall 6,513 s, child CPU 670m01s user + 93m16s sys = 12.72 core-h. Output stage2_data/adjudication_20261031T025902Z/ (STAGE2_RESULT.json sha 35091cdd...b15d, SHA256SUMS verified). Stage 2 total = 897.63 (Slurm physical) + 12.72 (login) = 910.4 of 1,400 core-h. Verdict POSITIVE_RUNNING_RESOLVED; pt CONSISTENT_WITH_ONE_LOOP. Lost tasks: 27 NODE_FAIL, 38 root-cancelled, all on cn012, all rerun on the same seeds.
