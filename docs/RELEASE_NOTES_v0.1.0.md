# v0.1.0 — recovered-record release

This is the first public code/data release for Neural Lattice Fields. It includes the recovered RG-1 production/analysis identities, 200 primary and 28 control measurement binaries in five release assets, individual/archive SHA-256 manifests, frozen results, pinned replay dependencies, legacy saved records and deterministic analytic checks.

The supported entry point is `tools/replay.py`. Its `all --quadrature` mode reprocesses existing measurements through the principal-result and optimized control-variate analyses, checks the full-data quadrature convergence, and runs the compact-record/analytic suite. The scripts do not start simulations, draw random samples, regenerate bootstrap uncertainties, compile LaTeX or render PDFs.

Evidence coverage and historical dependencies are stated in `docs/EVIDENCE_SCOPE.md`. In particular, unavailable legacy time histories are not replaced by summaries, saved diagnostic failures remain visible, and the original synthetic calibration/radius-grid summaries are carried records. Current condensate chi-square recombination excludes the deterministic zero-coupling row.

Project code is MIT licensed and project data are CC BY 4.0, with third-party notices preserved. Cite the versioned GitHub release using `CITATION.cff`. A DOI has not been assigned.
