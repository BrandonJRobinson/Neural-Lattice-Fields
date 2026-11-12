# RG-1 measurement contract

`manifests/raw-files.json` contains exactly 228 records: 200 primary and 28 controls, totaling 4,069,376,000 bytes and 127,168,000 rows. The primary records contain 123,988,000 rows. The files are unchanged recovered measurements, partitioned into five downloadable archives by lattice size.

Each `chain.bin` is a headerless sequence of four little-endian IEEE-754 binary64 values per row (32 bytes):

| Column | Symbol in the preserved reader | Definition |
| ---: | --- | --- |
| 0 | `u` | Uniform field component, `u = V^-1 sum_x phi_x`; the analysis uses `X = L u` |
| 1 | `S2` | `sum_x psi_x²`, where `psi_x = phi_x - u` and `sum_x psi_x = 0` |
| 2 | `S3` | `sum_x psi_x³` |
| 3 | `Gp` | Axis-averaged structure factor at the smallest nonzero lattice momentum, using the executed source normalization |

The precise structure-factor normalization and measurement ordering are defined by `data/rg1/source/src/phi4_local2.cpp` and the preserved `zm.py` reader. The reader uses native NumPy float64, so the public wrapper requires a little-endian host. The row count must match both `chain.json:n_meas` and file length divided by 32.

Companion files retain lattice size, mass, coupling, source, seed, thermalization/update/measurement intervals, elapsed time and QC provenance. Stage `PLAN.json` files identify the intended schedules and control roles. Primary and control membership is explicit in the public raw manifest; it is not inferred from lattice size alone.

The data archive layout starts with `data/rg1/stage1_data/...` or `data/rg1/stage2_data/...`, so extraction is relative to the repository root. The supplied downloader accepts only the manifest's exact regular-file members and two embedded license/attribution notices. It rejects links, unexpected paths and duplicate members, verifies compressed and uncompressed hashes, and refuses to replace different existing measurements.

The original Linux executable is `data/rg1/bin/phi4_local2`, retained without execute permission. Its SHA-256 is `fe5601a3a624572c60bd5c72c81f6cd55a164be64d817507879f6e1b477311c9`. Replay imports the pinned analysis routines; it does not execute this binary, a sampler, a scheduler or an SSH collector.
