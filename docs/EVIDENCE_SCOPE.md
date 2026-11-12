# Replay scope and provenance

This release separates measurement replay, deterministic recombination of saved summaries, and carried historical records. A passing replay verifies the stated calculations and byte identities within that scope; it does not manufacture missing observations or remove diagnostic failures.

| Evidence | Included records | Deterministic replay | Limitation |
| --- | --- | --- | --- |
| RG-1 principal result, R01–R05 | Executed source, original executable hash, stage plans, chain metadata, 200 primary/28 control binaries, original QC/adjudications | All chain QC; conditional moments; eight endpoints and joint jackknife covariance; matched couplings and adopted errors; primary fits; declared Stage-1 controls; selected-row adaptive integration and optional full-data 160-node convergence | Principal zero-source result and listed controls; historical synthetic calibrations and complete radius grid are carried records |
| Raw/conditional/optimized control-variate comparison | 200 primary binaries, source pin, per-chain primitive sufficient statistics, coefficients, covariance, timing and benchmark records | Fresh primitive/block statistics from measurements; eight methods × four blockings; covariance and matched-coupling inference | Archived timing measurements are not new cross-platform timing claims |
| Nonlinear calibration, N01 | Unrounded architecture summary, executed routines, source hashes | Spectrum, mixture/amplitude rescaling and relative-mode RMS recalculation | The relative-mode RMS and norm ratio are different metrics; neither is silently relabeled |
| Coincident cumulants, N02 | Original per-seed summaries for 15 width/law cells, uncertainties, plotting inputs and source | Recombination of variance, fourth and sixth cumulant summaries | Summary-level validation; no unavailable per-configuration history is reconstructed |
| P3 reference kernel, V01–V02 | Final rescue campaign's 72 hash-verified complete-run summaries, original verdict and source/configuration | Original observable-level diagnostics and all 18 summary-based trend contrasts, including simultaneous confidence constructions | Four independent replica summaries per contrast; centered normal replica model assumed; no equilibration or equivalence certification |
| Oscillator and determinant benchmarks, F01–F05 | Saved campaign records, deterministic-reference inputs, source/configuration, verdicts | Saved-run aggregation and comparison tests; deterministic `g = 0` condensate excluded from the current condensate chi-square | The saved HMC runs are not rerun; cross-coupling covariance is not inferred from absent histories |
| Scalar transitions, S01 | Volume/mass grids, per-seed summaries, crossing tables, diagnostics, source/configuration and source pins | Central crossing values, seed combinations, connected-xi conventions and input identities | Existing bootstrap errors are carried, not redrawn; failed and PARTIAL diagnostics retained |
| Spatial-cumulant bounds | Unrounded documented kernel parameters and analytic-check source | Symbolic identities, independent linear programs, finite-sign enumeration, reduced/full finite sums and outward interval enclosure | Exact global `rho_star` is not numerically computed; the documented-kernel `rho_M` lower bound does not replace it |

## RG-1 inference convention

The preserved inference code uses the second-moment finite-volume mass and zero-momentum Binder coupling with `z = L m_P` and `g_R = 3 z^4 U_4`. The eight endpoint values are interpolated to `z = 3`. The adopted endpoint error is the maximum of the pooled-block jackknife error and independent-replica scatter. The archived interpolation/error propagation rule is retained. Reconstructed joint endpoint covariance is available as evidence; replay does not silently replace the historical adopted-error model with a different covariance model.

The primary `MCUT` fit minimizes residuals on the coupling scale for `g_R = 1/(A + B log(L/12) + C/L²)`, using the adopted coupling errors. This differs from a linear least-squares fit to `1/g_R` with transformed errors. The frozen source includes both historical analysis paths; the primary replay imports the stage-2 coupling-scale function.

The principal replay checks selected actual configurations against independent adaptive full-line quadrature. The optional convergence replay evaluates all 224 zero-source chains with 160 nodes and a 16-curvature-width interval, versus the preserved 80-node/11-width rule. These are numerical accuracy checks within this campaign's convex conditional regime, not an all-parameter quadrature theorem.

## Source and environment identity

`provenance/ORIGIN_MANIFEST.json` records recovered-file origins and byte hashes. Byte-identical archival scripts retain their historical paths, output names and warning labels. The public portability adaptations are recorded there; they redirect inputs or extract deterministic summary functions without changing their scientific formulas. `provenance/LEGACY_SOURCE_PINS.json` verifies the executed lower-dimensional source/configuration identities.

The original RG-1 source manifest is itself preserved as a historical record. It also names historical figure files outside this curated replay payload; it is not the public release inventory. Use `manifests/files.json` for the public files. The frozen replay's nine required scientific source pins and the original executable hash are all included and checked. The original compiler record is GCC 8.5.0; the historical Python package versions were not recovered. `requirements.txt` and `provenance/REPLAY_ENVIRONMENT.json` identify the independent release replay environment.

All current replay outputs are written to newly created user-selected directories. Frozen `expected/` records remain inputs for comparison. The release validation record reports which commands were actually completed, rather than treating successful transfer as scientific closure.
