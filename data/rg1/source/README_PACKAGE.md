# Package: NNFT φ⁴ scientific rescue verdict (28 Oct 2026)

Start with `SCIENTIFIC_RESCUE_VERDICT.md`. Nothing here touches the frozen G2/S1–S6 records or any cluster.

## Layout
- `SCIENTIFIC_RESCUE_VERDICT.md`: the verdict.
- `figures/`: the three figures referenced in the verdict.
- `src/phi4_local.cpp`, `src/phi4_local2.cpp`: independent exact sampler. `phi4_local2` additionally skips the uniform-mode Gibbs step when a₂ ≤ 0 (exact; ψ-dependent kernel choice) and counts the skips.
- `analysis/*.py`: all analysis (Python 3, NumPy, SciPy, Matplotlib).
- `results/*.json`: every number quoted in the verdict.
- `SHA256SUMS.txt`: checksums of every file in this package and of every per-configuration data file.
- `phi4_rescue_configdata.zip` (separate): per-configuration binaries, 4 float64 per row: u = mean φ, Σψ², Σψ³, G(p_min).

## Rebuild and rerun (from the package root)
Scripts read and write `data/`. To regenerate tables and figures from the shipped results without resimulating: `mkdir -p data && cp results/*.json data/ && python3 analysis/figures.py`.
```sh
mkdir -p bin data figures
g++ -O3 -march=native -std=c++17 -o bin/phi4_local2 src/phi4_local2.cpp
# usage: L m0sq lambda h n_therm n_meas n_or meas_every seed outfile [zero_mode_gibbs=1]
./bin/phi4_local2 12 -0.6223 0.4 0.0 5000 1200000 2 1 1201 data/L12_h0_s1201.bin 1 > data/L12_h0_s1201.json   # ≈40 min/core
python3 analysis/analyze_ensemble.py out.json 12 -0.6223 0.4 200 data/L12_h0_s1201.bin                      # zero source + reweighting
python3 analysis/l12_followup.py out.json 12 -0.6223 0.4 200 data/L12_h0_s1201.bin                         # bias ratios, LY estimator, LP
python3 analysis/scaling_table.py data/scaling_spec.json data/scaling_final.json
python3 analysis/pt_compare.py data/scaling_final.json data/pt_compare_final.json
python3 analysis/running_fits.py data/scaling_final.json data/running_fits_final.json
```
The source h = η/L³ (η = 4.5 → h = 0.0026041666…, η = 6 → 0.0034722…, at L = 12).

## Archive re-analysis (for 6 Nov)
Export per-volume, per-draw paired values to JSON:
```json
{"volumes":[{"L":12,"draws":[{"z":3.02,"a1":0.1087,"q":{"4.5":0.1061,"6.0":0.1043},"s":{"4.5":0.1012,"6.0":0.0964}}]}]}
```
Then run `python3 analysis/reanalyze_archive_ly.py archive.json out.json --per-draw-lp`. This is a NEW prospective analysis label; the frozen estimator is unchanged.

## Status labels
- THEOREM: Lebowitz; Lee–Yang/Newman product form; Aizenman–Duminil-Copin applicability to the critical pullback.
- DERIVED: pinning law; latent-HMC frequency estimates; cost law.
- MEASURED-LOCAL: everything produced by `src/` on a 2-core sandbox (not HIPSTER, not G2).
- FORECAST: costs and power for RG-1.
