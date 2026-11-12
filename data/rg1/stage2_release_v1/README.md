# RG-1 Stage 2 production release (v1)

This release runs the same pinned exact sampler as Stage 1, `bin/phi4_local2` (sha256 `fe5601a3…11c9`), unmodified. The runner, capture, QC, frozen-pin and Stage 1 analysis modules are byte-identical to `stage1_release_v1`. The package's own `analysis/*.py` scripts are hash-pinned and used unmodified. Nothing here touches G2, S1–S6 or CV-1, and nothing uses the latent-HMC pipeline.

## Design (`PLAN.json`, frozen before any Stage 2 data)

The allocation is the one the contract predeclared (verdict §9; execution order step 4): **L12 0.2%, L16 0.3%, L24 1.35%, L32 1.35%** relative σ_g after z = 3 interpolation. The cost units come from Stage 1 measurements, pinned by sha256 in `make_plan.py`.

| Ensembles | L | m0² | Chains × sweeps | Role |
|---|---|---|---|---|
| Q12_lo / Q12_hi | 12 | −0.6250 / −0.6223 | 8 × 2.525M each | new z=3 bracket |
| Q16_lo / Q16_hi | 16 | −0.6570 / −0.6535 | 8 × 2.486M each | new z=3 bracket |
| Q24_lo / Q24_hi | 24 | −0.6782 / −0.6768 | 16 × 296k each | pooled with Stage 1 P24 (→ 26 chains per mass) |
| Q32_lo / Q32_hi | 32 | −0.6864 / −0.6856 | 42 × 245k each | pooled with Stage 1 P32 (→ 58 chains per mass) |

- **Tasks:** 148, each with 5,000 thermalization sweeps and an independent seed.
- **Core-hours:** about 923 expected; 1,299 allocated as task time limits, against the authorized 1,400.
- **Margins:** L12 and L16 get a 1.15× margin, because their units are partly forecast and the interpolation weight is uncertain. L24 and L32 get none, because their units were measured on these same ensembles.

## Adjudication

`rg1/analyze_stage2.py` holds the rules, predeclared in its docstring:

- the same per-ensemble rules as Stage 1;
- Stage 2 vs Stage 1 consistency at each L24 and L32 mass;
- L12 reproduction;
- the Lee–Yang and Lebowitz checks;
- the frozen `pt_compare.py` and `running_fits_g.py` scripts;
- L≥16 sensitivity fits;
- 2,000 synthetic campaigns under α=0 and α=1 on L12–32;
- a single claim, decided by MCUT on L12–32 (with M1 required to agree), with no substitution of window or model.

## Commands

Run the local preflight:

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage2_release_v1/preflight.sh
```

Submit; this refuses without `AUTHORIZATION_RG1_STAGE2.json`. Add `--exclude=hipster-cn011,hipster-cn013` to avoid nodes that failed in Stage 1:

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage2_release_v1/submit_stage2.sh
```

Check progress (read-only):

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage2_release_v1/status_stage2.sh
```

Run the adjudication after every chain has ended. It runs on the login node and refuses while Stage 2 jobs are still queued:

```bash
bash /home/brobins/RG1_SCIENTIFIC_RESCUE_20261028/stage2_release_v1/analyze_stage2.sh
```

## Authorization

`AUTHORIZATION_RG1_STAGE2.json` records the user's words "Authorize Stage 2" and the question they answered. The 435 core-hour figure in that question was the Stage 1 gate's minimum, which gives σ_B = b₁/2.5. This plan follows the contract's predeclared allocation instead, expected at about 923 core-hours, still within the authorized 1,400.

The Mission R locks stay in place, and their hashes are written to the ledger.
