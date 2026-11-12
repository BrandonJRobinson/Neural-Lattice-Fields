# Neural Lattice Fields

Version **v0.1.0** releases recovered HIPSTER code and scientific records for the lattice/NNFT project, with deterministic raw-to-result replay and checks of the spatial-cumulant bounds. The release contains the existing **200 primary and 28 control RG-1 measurement files**, their configurations and provenance, and the saved records used in the legacy analyses. The replay entry points do not run simulations, draw random samples, or render LaTeX/PDFs.

[Versioned release and raw data](https://github.com/BrandonJRobinson/Neural-Lattice-Fields/releases/tag/v0.1.0) · [Evidence scope](docs/EVIDENCE_SCOPE.md) · [Data format](docs/DATA_FORMAT.md) · [Licenses](DATA_LICENSE.md)

## Install and verify the source

Use a little-endian machine with CPython **3.13.2** and filesystem support for directory symlinks (the raw replay recreates the historical input layout). The independent release replay was validated on macOS and uses NumPy 2.3.5, SciPy 1.16.3, SymPy 1.14.0 and mpmath 1.3.0. These are the replay environment versions; they are not asserted to be the historical HIPSTER Python versions.

```sh
git clone --branch v0.1.0 https://github.com/BrandonJRobinson/Neural-Lattice-Fields.git
cd Neural-Lattice-Fields
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/verify_release.py
```

The repository contains about 48 MB of source and compact records. The measurement data are separate release assets: five archives totaling about 3.92 GB, expanding to **4,069,376,000 measurement bytes**. Allow at least 10 GB of free space for downloads, extraction and replay outputs. Git is needed for the clone; the downloader uses system `curl` when available, with a certificate-verified Python HTTPS fallback. No Git LFS or HIPSTER account is required.

## Replay the compact records and analytic checks

```sh
.venv/bin/python tools/replay.py summaries --output work/summaries
```

This command recombines the included estimator block statistics, audits the 72 P3 run summaries, checks the spatial-cumulant identities and sharp bounds, and replays the saved calibration, oscillator, fermion, cumulant and transition records. Its output is `work/summaries/RELEASE_REPLAY_REPORT.json`. This mode does not reconstruct unavailable legacy chain histories or regenerate carried bootstrap uncertainties.

## Download and replay all RG-1 measurements

```sh
.venv/bin/python tools/download_rg1.py
.venv/bin/python tools/verify_release.py --raw --report work/raw-integrity.json
.venv/bin/python tools/replay.py all --output work/full-replay --quadrature
.venv/bin/python tools/verify_estimator_slopes.py --blocks work/full-replay/estimator_raw/recomputed/BLOCK_REPLAY_VERIFICATION.json --reference replay/estimators/CV_BENCHMARK_RESULT.json --output work/full-replay/estimator-slopes.json
```

The downloader verifies both each compressed archive and every extracted measurement file against SHA-256 manifests. It refuses to replace differing measurement files. For an offline transfer, download all five assets and substitute `tools/download_rg1.py --asset-dir /path/to/assets`.

`all --quadrature` runs the compact-record checks, recomputes the RG-1 conditional observables and all 228 chain QC checks, reconstructs the eight mass endpoints and their joint jackknife covariance, interpolates to `z = 3`, refits the primary models, and replays the 28 controls within the documented principal-result scope. It also reruns the 160-node conditional-quadrature convergence check and recomputes the optimized control-variate sufficient statistics from the 200 primary binaries before replaying all eight estimator methods and four blockings. The preserved primary replay uses two workers; full validation runtime is recorded in `provenance/RELEASE_VALIDATION.json`. Each output directory must be new.

The final command also verifies all eight secondary estimator slope fits from those freshly replayed matched couplings and errors. It uses the preserved fitting function and keeps the benchmark joint-jackknife error model separate from the adopted primary envelopes. It does not remeasure the historical timing benchmarks.

To run the two measurement analyses separately:

```sh
.venv/bin/python tools/replay.py rg1 --output work/rg1 --quadrature
.venv/bin/python tools/replay.py estimator-raw --output work/estimator-raw
```

The primary replay reproduces the following frozen conditional estimates, including the adopted uncertainties:

| L/a | g_R at z = 3 | Adopted standard error |
| ---: | ---: | ---: |
| 12 | 5.7812164280 | 0.0107952524 |
| 16 | 5.6428034394 | 0.0183223643 |
| 24 | 5.4101717588 | 0.0741882840 |
| 32 | 5.2665443778 | 0.0758099853 |

For the primary coupling-scale fit of `g_R = 1/(A + B log(L/12) + C/L²)`, the archived result is `B = 0.02102875 ± 0.00668933` and `χ²/dof = 0.0262104/1`. The inference is conditional on the declared finite-volume scheme, interpolation, adopted error rule and fit model; it is not a continuum extrapolation or a model-independent beta function.

## Contents and evidence boundaries

| Path | Contents |
| --- | --- |
| `data/rg1/` | Executed C++ source, frozen Python analysis, stage plans/configurations, chain metadata, QC and adjudications; measurements are populated by the downloader |
| `data/legacy/` | Recovered campaign configurations, source and saved run/summary records |
| `replay/` | Preserved scientific replay scripts and explicitly documented portability adaptations |
| `expected/` | Frozen reference results, distinct from outputs created in `work/` |
| `manifests/` | Per-file sizes/SHA-256 values and the five release-asset manifests |
| `provenance/` | Input origins, source pins, release environment and validation record |
| `tools/` | Verification, download, packaging and read-only replay entry points |

The historical Linux production executable is retained for hash identity and is never executed by these commands. Original sampler and scheduler files are archival source, not replay entry points. Their historical paths and warnings are preserved as provenance. Optional plotting/sampling programs may require additional packages; the pinned replay does not invoke them. Private manuscript drafts, referee reports and access/transfer logs are outside this release.

RG-1 has measurement-level replay. The legacy campaigns have the saved-record coverage specified in [EVIDENCE_SCOPE.md](docs/EVIDENCE_SCOPE.md). In particular, P3 has 72 complete-run summaries, not time histories; its summary-based inference does not certify equilibration or equivalence. The historical synthetic calibration summaries, full radius diagnostic grid and carried transition-bootstrap errors are not regenerated by this release. Saved failures and partial campaign statuses remain in the data.

## Integrity, citation and licensing

`SHA256SUMS.txt` and `manifests/files.json` identify the source and compact-record payload; the manifest excludes itself and the checksum list to avoid a hash cycle. `manifests/raw-files.json` identifies all 228 binaries, and `manifests/assets.json` identifies the compressed assets and embedded notices. The Git tag identifies the complete source tree. See `provenance/ORIGIN_MANIFEST.json` for recovered-file hashes and the few documented adaptations.

Cite **Vishnu Jejjala and Brandon Robinson, Neural Lattice Fields: recovered-record replay and analytic checks, v0.1.0 (2026)**, using the [versioned release](https://github.com/BrandonJRobinson/Neural-Lattice-Fields/releases/tag/v0.1.0). Machine-readable metadata are in [CITATION.cff](CITATION.cff). No DOI is assigned in this release.

Project code is [MIT licensed](LICENSE); project data are [CC BY 4.0](DATA_LICENSE.md). Existing third-party notices retain their terms. The full data license is included in the repository and each measurement archive.
