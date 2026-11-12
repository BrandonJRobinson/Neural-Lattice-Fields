# RG-1 Stage 1 integrated production release (v1)

This release runs the staged exact sampler `bin/phi4_local2` unmodified. Its binary sha256 is `fe5601a3…11c9`, built from `src/phi4_local2.cpp` (`492c2aa2…55c7`), which came from the package `0b833d45…6db`. The analysis uses the package's own `analysis/*.py`, hash-pinned and unmodified. Nothing here touches G2, S1–S6 or CV-1, and nothing uses the latent-HMC pipeline.

## Design (`PLAN.json`, frozen before any data)

| Ensemble | L | m0² | Chains × sweeps | Purpose |
|---|---|---|---|---|
| P24_lo / P24_hi | 24 | −0.6782 / −0.6768 | 10 × 296,000 each | z=3 bracket, symmetric about −0.6775 |
| P32_lo / P32_hi | 32 | −0.6864 / −0.6856 | 16 × 245,000 each | z=3 bracket, symmetric about −0.686 |
| C12_repro | 12 | −0.6223 | 8 × 150,000 | reproduce z=3.0245(11), g=5.805(41) |
| C08_gauss | 8 | +0.5, λ=0 | 4 × 100,000 | g=0, exact ⟨X²⟩ |
| C12_local, C16_local / C16_gibbs | 12, 16 | −0.6223, −0.657 | 4 × 150k/100k | independent local-only kernel vs Gibbs kernel |
| S24_eta4p5, S32_eta4p5 | 24, 32 | −0.6768, −0.6856; η=4.5 | 2 × 60k / 2 × 30k | direct-source check of the reweighting identity |

Every chain runs 5,000 thermalization sweeps with an independent seed.

**Precision target.** Production aims at 2% on g at z=3, taking g=5.4 as a conservative reference. That needs 5.9×10⁶ sweeps at L24 and 7.8×10⁶ at L32, using the measured per-configuration spread and τ_int.

**Bracket width.** The half-widths in m0² follow from dz/dm0² ≈ L²/(2z). With them, each pair straddles z=3 by about ±0.067.

## Resources

- **Allocated:** 593 core-hours. That is the sum of all task time limits plus 12 for the adjudication job, against the 600-hour Stage 1 ceiling.
- **Expected use:** about 430 core-hours. The login node measured 0.118 s per sweep at L32, 0.037 at L24 and 0.0023 at L12.
- **Wall time:** about 11 h with all 80 single-core tasks running at once. They use 80 of the 256 CPUs the capacity QOS allows.
- **Storage:** about 0.5 GB on `/home`.

## Pipeline

1. **Run.** `rg1/run_chain.py` runs one chain in node-local scratch, under a wall-clock guard set 10 minutes below the SLURM limit.
2. **Structural check.** It checks the exit status, that the stdout JSON echoes the planned parameters, and that the file is exactly `32·n_meas` bytes.
3. **Capture before QC.** It then copies the output to `stage1_data/<ens>/chain_NN/`, with fsync and a sha256 compared before and after the copy, plus a manifest and provenance record, and renames the directory atomically.
4. **QC.** Only after that does QC run on the persistent copy:
   - all values finite;
   - acceptance rates in [0.90, 1.00];
   - split-chain stationarity |z| ≤ 4;
   - 8 blocks of at least 50 τ_int each.
5. **Failures.** A failed or timed-out chain leaves no completed file. It is recorded in `stage1_data/failures/` and is never analysed.

**Restart.** Rerunning a task reuses its seed, so the chain is bit-identical. Chains that already have a verified capture are skipped, and a capture that fails verification is never overwritten. `submit_stage1.sh --retry` resubmits only missing chains, and only within the ceiling measured by sacct.

**Adjudication** (`rg1/analyze_stage1.py`, dependent job, rules in its docstring):
- per-ensemble frozen analysis;
- replica scatter;
- interpolation to z=3, with the bracket rule;
- the controls (PASS ≤ 2σ, WARN 2–3σ, FAIL > 3σ);
- the Lee–Yang and Lebowitz checks;
- PT comparison and M1/MCUT/MPT fits, using the frozen scripts;
- the endpoint B₁₂,₃₂;
- the Stage 2 gate: projected MCUT σ_B ≤ b₁/2.5 from measured costs, within 1,400 core-hours, with no production, bracket or control failure.

## Commands

Run the integrated local preflight:

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage1_release_v1/preflight.sh
```

Submit; this refuses without `AUTHORIZATION_RG1_STAGE1.json`:

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage1_release_v1/submit_stage1.sh
```

Check progress (read-only):

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage1_release_v1/status_stage1.sh
```

## Authorization

Every `PRODUCTION_LOCKED` file on this account carries the Mission R text "HIPSTER arrays and multi-volume triviality production are not authorized". RG-1 Stage 1 is exactly that kind of production, even though it never calls a lattice_nnft launcher. The locks stay in place and are hashed into the submission ledger.

Submission requires `AUTHORIZATION_RG1_STAGE1.json`, built from `AUTHORIZATION_TEMPLATE.json`. It must name the lock authority, state that RG-1 Stage 1 is authorized notwithstanding the locks, and give this `PLAN.json` sha256 and a ceiling of 600 core-hours. Stage 2 needs its own record.

## Open item

The contract's cross-sampler control against a kernel-matched latent HMC would require the latent-HMC code path that this assignment excludes. Stage 1 instead uses two substitutes:
- an independent Markov kernel (local-only updates, `zero_mode_gibbs=0`) at L12 and L16;
- the L12 reproduction control.

This substitution is stated in the decision report.
