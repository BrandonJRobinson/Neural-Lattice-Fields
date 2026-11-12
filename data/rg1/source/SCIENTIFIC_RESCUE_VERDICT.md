# Scientific rescue verdict — 4D lattice φ⁴ NNFT programme

**28 October 2026 · independent review · no cluster access used or attempted · frozen S1–S6/G2 records untouched**

> **Verdict: `NEW_RG_DESIGN_ROUTE` (primary).** It is enabled by two new estimator results (`NEW_ESTIMATOR_ROUTE`: exact Lee–Yang structure of the source response, and exact uniform-mode sufficiency). It is complemented by a partial `EXISTING_DATA_ROUTE` for 31 October, which resolves the magnitude puzzles.
>
> **Qualifications.**
> - The frozen four-volume dataset cannot be turned into a running result.
> - The finite-source/zero-source discrepancy is resolved: at L=12 it is a theorem-signed truncation bias; at L=24/32 the October zero-source point values are not reproduced by an independent exact sampler. It is not a physical conflict.
> - A statistically resolved, perturbation-theory-consistent decrease of g_R with L/a at z≈3 is obtained in this session on small lattices (L/a = 6–12). This is a new, separately labelled analysis, theory-assisted and cutoff-confounded. Short runs at L/a = 16–32 sit 1–2σ *above* the one-loop expectation (combined ≈2–2.5σ, p ≈ 0.01–0.03); that tension is exactly what the proposed Stage 1 settles.
> - A cutoff-robust (MCUT) running measurement on the frozen λ=0.4 trajectory is **affordable**: forecast ≈1×10³ core-hours with an efficient exact sampler and the exact zero-source estimator. It is not affordable with the configured latent HMC.
> - It requires one authorization (§9).

Status labels used throughout:
- **[THEOREM]** follows rigorously from published theorems.
- **[DERIVED]** is an analytic derivation, or an approximation stated as such.
- **[MEASURED-LOCAL]** is a new Monte Carlo result from this session's independent sampler (not HIPSTER; not part of G2).
- **[FORECAST]** is a cost or power projection.
- **[ARCHIVE]** is a value quoted from the supplied documents.

No item below is a "scientific PASS" of a frozen gate.

---

## 1. Results first

1. **Positivity of g_R is not evidence. [THEOREM]**
   - For the executed action the Lebowitz inequality gives κ₄(X) ≤ 0 at h=0, for every L, every m₀² and every λ>0. Hence g_R ≥ 0.
   - The October 27 "strongest empirical physics statement" (provisional positive finite-volume interaction) is a sampler consistency check, not a physical result.
   - Only magnitudes and their L-dependence carry information.

2. **The finite-source response has an exact Lee–Yang form, and the extrapolation obstruction is not physical. [THEOREM]**
   - With X = LM, a₁ = ⟨X²⟩₀, η = hL³ and τ = η²a₁/2:
     - q(η)/a₁ = Σ_j w_j/(1+τw_j)
     - g_R = 3z⁴Σ_j w_j²
   - Here w_j = 2/(a₁α_j²) > 0, Σw_j = 1, and ±iα_j are the Lee–Yang zeros, counted with multiplicity.
   - Every positively weighted cubic q+s fit is biased **low**, with a theorem-fixed sign. The October 27 null direction is not of this form.
   - With exact data at the frozen radii, the set of g_R compatible with *any* Lee–Yang measure is an interval of relative width ~10⁻⁵ (L=12, below).

3. **L=12 is resolved at the frozen bare parameters (λ=0.4, m₀² = −0.6223). [MEASURED-LOCAL]** Independent exact sampler, 2.4×10⁶ configurations:
   - **z = 3.0245 ± 0.0011.** Frozen value: 3.02187. [ARCHIVE]
   - **g_R(12) = 5.805 ± 0.041** (0.7%, zero source). This agrees with the project's own direct cumulant, 6.108 ± 0.336 [ARCHIVE], at 0.9σ.
   - The population value of the frozen cubic common-window functional is **4.818 ± 0.034, i.e. 0.830 ± 0.003 of g_R**. The frozen value 4.547 [4.06, 5.04] [ARCHIVE] is this biased functional plus a ~1σ fluctuation.
   - The one-atom Lee–Yang estimator G_q²/G_s recovers **0.9989 ± 0.0002** of g_R at η=4.5 and **0.9968 ± 0.0004** at η=6. It is a rigorous lower bound.

4. **The cubic-window bias is volume-independent at fixed z. [MEASURED-LOCAL]**
   - cubic/g_R = 0.830, 0.826, 0.831, 0.830 at L = 6, 8, 10, 12 (±0.0005–0.004).
   - Consequence: over this range the frozen common window rescales 1/g by ≈1.20. It **cannot manufacture or erase running**; it only rescales B and the perturbative comparator.
   - Constancy at L ≥ 16 is expected (fixed z, nearly fixed a₁), but it is not yet measured precisely there. Stage 1 checks it.

5. **The L=24/32 "largest scientific warning" does not survive. [MEASURED-LOCAL]**
   - Same bare parameters: **g_R(24) = 6.95 ± 1.31** (m₀² = −0.676, z = 3.145 ± 0.013) and **g_R(32) = 7.87 ± 2.13** (z = 2.999 ± 0.029). Batch-means errors are ±1.0 and ±2.5.
   - A second L=24 ensemble at the project's retuned mass (m₀² = −0.6775) gives z = 3.015 ± 0.016 and g_R = 8.27 ± 1.0 (batch ±1.3).
   - The October h=0 point values 14.52 and 19.45 [ARCHIVE] are ≈5σ above these (4.6–5.8σ in units of the independent jackknife or batch errors).
   - They are consistent with upward fluctuations of a very noisy estimator whose quoted intervals were very likely too narrow (§4.3). A sampling defect in those specific ensembles is not excluded and cannot be checked without their per-trajectory data. Either way, there is no physical source/zero-source conflict.

6. **Exact uniform-mode conditioning (RB) is exact but self-limiting; the pinning law is verified. [DERIVED + MEASURED-LOCAL]**
   - The uniform mode u is tied to the UV composite Σψ³ by the 4λuΣψ³ term.
   - Conditional-variance fraction: **f ≈ m_χ²/(m_χ² + Σ)**, with Σ ≈ 0.04–0.046 nearly L-independent.
   - The RB/raw variance ratio for the κ₄ influence is **(1−f)⁴**. Measured f vs prediction:
     - L=12: 0.595 vs 0.612
     - L=16: 0.429 vs 0.439
     - L=24: 0.276 vs 0.279
     - L=32: 0.161 vs 0.163
   - Measured variance ratio vs (1−f)⁴:
     - L=12: 0.0267 vs 0.0268
     - L=16: 0.103 vs 0.107
     - L=24: 0.271 vs 0.274
     - L=32: 0.442 vs 0.495
   - This explains the historical RB/raw degradation 0.19 → 0.78 between L12 and L32.

7. **A resolved, perturbation-theory-consistent decrease of g_R exists on small lattices; the large-volume points are in ≈2–2.5σ tension with it. [MEASURED-LOCAL; new prospective analysis; fits in g-space]**
   - g_R = 6.130(15), 5.992(39), 5.989(59), 5.805(41) at L/a = 6, 8, 10, 12 (z ≈ 3.02–3.04).
   - On L/a = 6–12, the fitted amplitude relative to the scheme-matched exact one-loop finite-lattice kernel at measured z is **α = 0.875 ± 0.067**:
     - α = 0 is excluded by Δχ² = 189.
     - α = 1 sits at 1.9σ, or ≈1σ after the expected −7% two-loop shift.
     - χ² = 1.9/2.
   - The short L ≥ 16 runs give 5.89(21), 6.95(1.31), 8.27(1.27) and 7.87(2.51). Every one lies above the one-loop expectation (pulls +1.0 to +2.3; χ² ≈ 10.4/4, p ≈ 0.03).
   - On the frozen-ladder volumes alone (L = 12–32), α = −0.65 ± 0.60: no resolved running, and 2.5σ below 1.
   - All eight ensembles: α = 0.84 ± 0.06 (χ² 10.8/6). Unrestricted log slope B = 0.0115 ± 0.0015. With a free L⁻² term, B = 0.006 ± 0.010 (unresolved).
   - This is evidence for the expected *finite-lattice* RG behaviour at L/a ≤ 12 only. It is **not** a cutoff-robust running coefficient. Whether the L ≥ 16 excess is a fluctuation or a failure of one-loop finite-size PT at z ≈ 3 (where Weisz–Wolff report PT degrading) is open.

8. **Most of the cost barrier is the sampler and the design, not the physics. [MEASURED-LOCAL + DERIVED + FORECAST]**
   - Local exact sampler cost: 0.146 core-seconds per sweep at L=32, with τ_int ≈ 1 sweep for the κ₄ influence.
   - Configured latent HMC: ≈75 core-seconds per trajectory [ARCHIVE cost table]. With mref² = 0.5 ≫ m_R², derived IR autocorrelations are O(10²–10³) trajectories.
   - A cutoff-robust four-volume running test at λ=0.4 (80% power, free L⁻²) is forecast at **≈1.0×10³ core-hours**. This puts precision on cheap volumes: L=12 at 0.2%, L=16 at 0.3%, L=24/32 at 1.35%. With equal precision everywhere it would cost ≈3.9×10³.
   - The endpoint-first Stage 1 is forecast at ≈4–5×10² core-hours.

**Strongest physical statement now supported (31 October, two separately labelled parts).**
- *Frozen analysis.* Unchanged: running NOT_RESOLVED. The frozen couplings are positive by theorem and are values of a functional biased low by ≈17%.
- *New independent analysis.*
  - At the frozen L=12 parameters, g_R = 5.805(41).
  - Across L/a = 6–12 at z ≈ 3, the finite-volume coupling decreases as predicted by scheme-matched one-loop lattice perturbation theory (α = 0.88 ± 0.07).
  - At L/a = 16–32, short runs give couplings of 6–8 with 3–30% errors, ≈2–2.5σ above that prediction in combination. No running is resolved there.
  - No continuum or cutoff-robust running claim is made.

---

## 2. What can still be established from existing accepted evidence

| Item | Status | Basis |
|---|---|---|
| Frozen G2 four-volume running: B intervals include 0 and PT | Unchanged, NOT_RESOLVED | [ARCHIVE] |
| Sign of g_R (all L) | Theorem; removed as an empirical claim | Lebowitz |
| Frozen g values are a population **lower bound** on g_R (positive-weight GLS) and ≈0.83 g_R at fixed z | Reinterpretation | Lee–Yang [THEOREM] + calibration [MEASURED-LOCAL] |
| Running inferred from the frozen common window is not an artefact of source truncation | New (measured for L = 6–12; expected, not yet verified, at L ≥ 16) | constant bias ratio [MEASURED-LOCAL] |
| L12 magnitude; source vs zero-source conflict at L12 | Resolved | §1.3 |
| L24/L32 source vs zero-source "warning" | Resolved: October zero-source point values not reproduced by an independent exact sampler | §1.5 |
| Lee–Yang re-analysis of archived per-radius draws (a₁, q(η), s(η), z per draw) | Feasible in minutes **if those draws exist** on the Mac. At frozen statistics it yields honest but wide intervals: emulated L12 gives [4.6, 9.6] vs cubic [3.9, 5.4]. | §4.4, `reanalyze_archive_ly.py` |
| Exact same-ensemble reweighting of archived data | **Not possible.** Historical streaming kept block means. Per-configuration (Σψ², Σψ³) exist nowhere in accepted evidence. | code inspection |

**Access limitation.** The October 27 reproducibility ZIP and `independent_results.json` on the user's Mac were not accessible: no folder is connected to this session, and none was requested. October 27 numbers above are quoted from the supplied text. None of this session's new results depends on them except as comparison targets.

---

## 3. Adversarial reconstruction of the October 27 decision

| Oct 27 element | Independent check | Verdict |
|---|---|---|
| "Strongest empirical statement: provisional positive interaction" | Lebowitz: g_R ≥ 0 identically | **Fails as physics.** The sign is a consistency check; the L32 sign-marginality only says the estimator is noisy. |
| Frozen common-window values read as couplings (with caveats) | Lee–Yang sign theorem; L12 calibration | **Fails in interpretation.** They are a ≈17%-low biased functional, uniformly in L. |
| Null direction ⇒ "a usable bound requires a physical constraint the archive does not supply" | The constraint is a theorem. In the physical class, identification at the frozen radii is ~10⁻⁵ (exact data). | **Fails physically.** The formal statement holds only outside the class of φ⁴ measures. |
| Source/zero-source discrepancy blocks magnitudes | L12: bias measured, sign proven. L24/32: October h=0 values not reproduced (≈5σ in independent units) | **Fails.** The discrepancy is explained, not physical. |
| Same-ensemble reweighting impossible without per-config inputs | Two scalars per configuration suffice | **Withstands** for the archive; trivial for any new run. |
| S1 coefficients PASS / remainder NOT_CERTIFIED | One-loop kernel reproduced (0.0192746, L12→32). Weisz–Wolff: finite-size PT deteriorates for z ≲ 3. | **Withstands.** Moreover, the literal frozen MPT kernel fails at small L (L6: 4.6σ). The scheme-matched one does not (1.3σ). |
| S3 STOP; ~2,872× information needed | Conditional on the configured sampler, the cubic estimator and equal allocation | **Withstands as stated.** Its implication of "no affordable route" fails. A ~500× lower measured per-sweep cost, plus exact conditioning on the cheap volumes, makes the required information affordable (§6.1), whatever the unmeasured HMC autocorrelation turns out to be. |
| S5 NOT_TESTABLE_WITH_ARCHIVE | Paired per-config RB/raw measured at L = 6–32 on new data; pinning law explains the deterioration | **Withstands for the archive.** The underlying question is now answered. |
| S6 NOT_RESOLVED | — | **Withstands.** |
| z⁴ propagation ≠ z transport | z(12), z(32) agree with frozen within errors; z(24) = 3.145 vs 3.075 | **Withstands.** Transport is still required for any fixed-z claim. |

---

## 4. Approach I — exact Lee–Yang structure of the source response

### 4.1 Derivation [THEOREM]

The action S = Σ_x[½Σ_μ(φ_{x+μ}−φ_x)² + ½m₀²φ_x² + λφ_x⁴] is equivalent to nearest-neighbour **ferromagnetic** couplings with single-site weight exp[−(4+m₀²/2)φ² − λφ⁴]. This weight is in the Griffiths–Simon class for every real m₀² (Simon–Griffiths 1973) and has the Lee–Yang property (Newman 1974). The Gaussian-latent pullback φ = K_ref^{−1/2}θ is linear with constant Jacobian, so every statement applies verbatim to the NNFT field distribution.

1. **Lebowitz.** u₄ ≤ 0 at h=0, hence κ₄(X) ≤ 0 and g_R = −z⁴κ₄/κ₂² ≥ 0.
2. **Lee–Yang + Hadamard.** Z(η) = ⟨e^{ηX}⟩₀ is even, entire of order 4/3, and vanishes only on Re η = 0. Hence Z(η) = Π_j(1+η²/α_j²) exactly, with no exponential factor because the order is < 2. Then
   - q(η) ≡ ⟨X⟩_η/η = Σ_j 2/(α_j²+η²)
   - s(η) ≡ Var_η X = Σ_j 2(α_j²−η²)/(α_j²+η²)².
3. With w_j = 2/(a₁α_j²) and τ = η²a₁/2:
   - R(τ) := q/a₁ = Σw_j/(1+τw_j)
   - D(τ) := (q−s)/(η²a₁²) = Σw_j²/(1+τw_j)²
   - **U₄ = Σ_j w_j², g_R = 3z⁴Σ_j w_j².**

   The finite-volume coupling is 3z⁴ times the inverse participation number of the normalized Lee–Yang zeros. Check: a two-point X gives Σw_j² = 2/3, the known maximum (g ≤ 2z⁴; Weisz–Wolff eq. 10).
4. **Signed truncation.** Define the secant G_q = 6z⁴(a₁−q)/(η²a₁²) and the tangent G_s = 3z⁴(q−s)/(η²a₁²).
   - For all η > 0: **G_s < G_q < g_R**, both monotonically decreasing.
   - The frozen cubic model asserts G_q = G_s = g. Its s-row equals (G_q + 2G_s)/3.
   - A positive-weight GLS cubic fit therefore lies below g_R.
5. **One-radius brackets.**
   - G_q²/G_s ≤ g_R (Cauchy–Schwarz on ν_j ∝ w_j²; this equals the one-atom Lee–Yang Padé).
   - g_R ≤ G_q(1+τw_max), with w_max ≤ √U₄ because each atom at w carries mass ≥ w (integer multiplicity).
6. **Identification.** Given R and D at several τ, the U₄ compatible with *some* Lee–Yang measure form an interval, computed exactly by a linear programme over measures on (0, √U₄,max] (`ly_bracket.py`).
   - Validated on exactly solvable measures: pure quartic U₄ = 0.27052 ∈ [0.270496, 0.270643].
   - The October 27 polynomial δμ = εη³Π(η²−r²)² has no representation of the form Σ2η/(α²+η²) with unit residues, so it is not realizable.

### 4.2 Numerical confirmation at L=12 [MEASURED-LOCAL]

Same-ensemble exact reweighting (Z₂-mirrored) of the 2.4M-configuration h=0 ensemble; delete-one-block jackknife over 200 blocks.

| η | G_q/g_R | G_s/g_R | (G_q²/G_s)/g_R |
|---|---|---|---|
| 4.5 | 0.9364(13) | 0.8778(24) | 0.99888(15) |
| 6.0 | 0.8934(21) | 0.8008(37) | 0.99677(44) |

- Frozen cubic functional (q+s at 4.5 and 6 with anchor): **0.8299(32)** (equal-N GLS) and **0.8442(29)** (precise anchor).
- Lee–Yang LP bracket on central data: [5.80521, 5.80527]. With 1σ (2σ) data boxes: [5.720, 5.924] ([5.650, 6.001]).
- The exact reweighting identity is checked against **independent direct sourced runs** (3×10⁵ each). All four quantities agree within ≤1.2σ:

| Quantity | Reweighted from h=0 | Direct sourced run |
|---|---|---|
| q(4.5) | 0.106086(45) | 0.106023(111) |
| s(4.5) | 0.101231(62) | 0.101334(118) |
| q(6) | 0.104283(50) | 0.104174(79) |
| s(6) | 0.096410(90) | 0.096425(95) |

![L=12 Lee–Yang structure](figures/fig2_L12_LeeYang.png)

### 4.3 Why the October L24/L32 direct cumulants were outliers [MEASURED-LOCAL + DERIVED]

- The measured per-configuration standard deviation of the g estimator at L=32 is ≈210 (in units of g, RB).
- The October h=0 sample was 7 replicas × 4096 trajectories [ARCHIVE]. The quoted 95% intervals (half-widths ≈ 10–11) are consistent only if τ_int ≲ 5–8 trajectories, using this per-configuration variance (L=24: ≈222 with 4 replicas).
- Derived latent-HMC frequencies for the configured kernel (mref² = 0.5, T ≤ 0.4) give ω ≈ 0.13 for the zero-mode collective direction and 0.30 for p_min at L=32. That implies τ_int ~ 650 and ~130 trajectories respectively [DERIVED; not measured on HIPSTER].
- The October intervals were therefore very likely too narrow, and the point values (14.5, 19.4) are consistent with fluctuations. The independent exact-sampler values at identical bare parameters are 6.95(1.31) at L=24 and 7.87(2.13) at L=32; the retuned L=24 mass gives 8.27(1.27).

### 4.4 What Lee–Yang does *not* buy: no free lunch from finite sources [MEASURED-LOCAL]

Emulating the frozen design at L=12 (independent h=0 anchor plus independent source runs at η=4.5 and 6; block bootstrap; 400 paired draws):

| Statistics per ensemble | Cubic GLS (95%) | Lee–Yang LP per draw (95%) | Truth |
|---|---|---|---|
| 2×10⁴, raw (≈ frozen L12 precision) | 4.64 [3.87, 5.45] | 7.3 [4.6, 9.6] | 5.805 |
| 2×10⁴, RB | 4.80 [4.49, 5.15] | 5.36 [4.71, 7.07] | 5.805 |
| 6×10⁵ anchor + 3×10⁵ per radius, RB | 4.84 [4.76, 4.92] | 5.88 [5.39, 6.55] | 5.805 |

- The cubic fit's apparent precision is bought entirely with its truncation assumption.
- Removing the bias with the theorem costs a factor ~5–8 in standard deviation (validated also on synthetic measures).
- At equal cost, zero-source RB at L ≤ 16 is the better estimator. At L ≥ 24, where RB fades, finite sources plus Lee–Yang are competitive but not superior.
- Finite-source ensembles remain useful as **consistency checks**: s < q, G_s < G_q, and LP feasibility t* = O(1).

---

## 5. Approach II — exact uniform-mode conditioning, sufficiency and pinning

### 5.1 Derivation [DERIVED, exact identities]

Split φ_x = u + ψ_x with Σψ = 0 (unit Jacobian; in latent variables u is one coordinate, θ̂(0) = √(mref²)·√V·u). Then

  S = S_ψ[ψ] + a₄u⁴ + a₂u² + (a₁ − ηL)u,  with a₄ = λV, a₂ = ½m₀²V + 6λΣψ², a₁ = 4λΣψ³.

The mandate's formula is confirmed, including the source term and normalisations. Negative m₀² is allowed while a₂ > 0. That held for every measured configuration at L ≥ 6; transient a₂ ≤ 0 occurs only during thermalisation.

- **Exact Gibbs step for u|ψ.** The conditional is log-concave when a₂ > 0, and exact rejection sampling from the Gaussian envelope exp[−a₂(u−u₀)²] dominates it. When a₂ ≤ 0 the step is skipped; this remains exact because the decision depends on ψ alone, which the step leaves invariant.
- **Sufficiency.** For every uniform source and every zero-mode observable,
  ⟨F⟩_η = E₀[w_η E_η(F|ψ)]/E₀[w_η],  w_η = Z_u(Σψ²,Σψ³;η)/Z_u(Σψ²,Σψ³;0).
  Two scalars per configuration therefore carry everything; block means do not.
- **Pinning.** Var X = E[Var(X|ψ)] + Var(E[X|ψ]). In the Gaussian-conditional approximation f := E[Var(X|ψ)]/Var X ≈ m_χ²/μ_c², with μ_c² = m₀² + 12λ⟨ψ²⟩ (Hartree) and m_χ² = 1/G(0).
  - Σ := μ_c² − m_χ² is a UV self-energy shift (sunset-type, ∝ λ²), nearly constant in L.
  - At fixed z, m_χ² ∝ L⁻², so f → 0: the uniform mode becomes a deterministic functional of Σψ³.
  - For the κ₄ influence function, Var_RB/Var_raw → (1−f)⁴.
- **Raw noise floor.** σ_g ≈ z⁴√24/√N_ind ≈ 397/√N_ind at z=3. RB improves it only by (1−f)².

### 5.2 Measurements [MEASURED-LOCAL]

| L | m₀² | N | z | g_R (RB) | f | f_pred | Σ | RB/raw (κ₄) | (1−f)⁴ | τ_int(κ₄ infl.) | s/sweep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -0.4 | 6.00e+05 | 3.0385(16) | 6.130 ± 0.015 | 0.876 | 0.912 | 0.0254 | 0.0007 | 0.0002 | 0.51 | 0.00022 |
| 8 | -0.53 | 4.00e+05 | 3.0295(20) | 5.992 ± 0.038 | 0.783 | 0.810 | 0.0342 | 0.0033 | 0.0022 | 0.50 | 0.0005 |
| 10 | -0.589 | 4.00e+05 | 3.0385(25) | 5.989 ± 0.059 | 0.689 | 0.709 | 0.0383 | 0.0098 | 0.0094 | 0.51 | 0.001 |
| 12 | -0.6223 | 2.40e+06 | 3.0245(11) | 5.805 ± 0.041 | 0.595 | 0.612 | 0.0405 | 0.0267 | 0.0268 | 0.51 | 0.002 |
| 16 | -0.657 | 2.50e+05 | 2.9360(43) | 5.889 ± 0.194 | 0.429 | 0.439 | 0.0431 | 0.1032 | 0.1065 | 0.55 | 0.0059 |
| 24 | -0.6775 | 4.00e+04 | 3.0151(163) | 8.266 ± 1.021 | 0.254 | 0.259 | 0.0452 | 0.2709 | 0.3096 | 0.78 | 0.031 |
| 24 | -0.676 | 4.00e+04 | 3.1451(131) | 6.951 ± 1.313 | 0.276 | 0.279 | 0.0449 | 0.2712 | 0.2742 | 0.70 | 0.048 |
| 32 | -0.686 | 2.00e+04 | 2.9990(288) | 7.866 ± 2.131 | 0.161 | 0.163 | 0.0456 | 0.4418 | 0.4948 | 1.03 | 0.15 |

![Pinning law](figures/fig3_pinning.png)

**Judgment.** Conditioning is the right estimator below L ≈ 16 at z=3; there it buys factors of 10²–10³ in variance. It cannot rescue a direct cumulant at L ≥ 24, and it does not remove the need for many independent configurations there.

---

## 6. Route C — RG designs compared

### 6.1 Identification and the cost law [DERIVED]

- Over L ∈ {12,16,24,32}, corr(log L, L⁻²) = −0.968. Profiling a free L⁻² term inflates σ_B ≈4× at equal precision (independently reproduced, including the October 5 table).
- Because σ_g per configuration is essentially independent of g while the target is a slope in 1/g, the required statistics scale as N ∝ z⁸/(g⁴σ(1/g)²).
- Allocation matters as much as the total. Cheap volumes can be made very precise with RB, so the free-L⁻² term is constrained there.

| Design (free L⁻², 80% two-sided power for B = b₁ + b₂g) | Required precision | Forecast cost (λ = 0.4, exact local sampler) |
|---|---|---|
| L = 12–32, equal precision | 0.65% everywhere | ≈3.9×10³ core-hours |
| L = 12–32, optimal allocation | 0.2%, 0.3%, 1.35%, 1.35% | **≈1.0×10³ core-hours** (L32 ≈ 725, L24 ≈ 175, L16 ≈ 55, L12 ≈ 16) |
| L = 12–48, equal precision | 1.2% | ≈10⁴ core-hours (L48 dominates) |
| Endpoint-first Stage 1 (L24, L32 at 2%, + in-session L ≤ 12) | — | ≈410 core-hours (≈510 with z-bracketing); theory-assisted 12→32 contrast at ≈4.1–4.3σ if PT holds |

**Assumptions behind these forecasts.**
- Per-configuration σ_g and τ_int are extrapolated from 20–40k-sweep runs at L = 24 and 32 (uncertainty ≈±30%).
- Timings are single-core with two concurrent processes.
- No z-transport overhead is included: L32 is already at z = 3.00 ± 0.03, and L24 needs m₀² ≈ −0.6775.
- O(a⁴) artefacts at L=12 are comparable to 0.2% precision, so MCUT must be reported alongside the exact one-loop lattice kernel (MPT) and an L ≥ 16 subset.

### 6.2 Strengthened bare coupling [MEASURED-LOCAL pilot + FORECAST]

- At λ = 2, L = 12, z = 3.045(13): g_R = 12.9 ± 1.1. Per-configuration σ ≈ 270 and τ ≈ 1.6 (RB gives no help: f ≈ 0.1).
- The net statistics gain over λ = 0.4 is ≈5–8×, smaller than naive g⁻⁴ because RB is lost and b₂g reduces the target slope by ≈15%.
- This is a valid cheaper alternative, at the price of a new trajectory and weaker perturbative control.

### 6.3 Other options assessed

- **Step scaling at fixed renormalised u.** Requires tuning (λ, m₀²) per resolution. It is cleaner for cutoff separation, but needs more tuning work than the fixed-z vertical line at this stage. It is not recommended first.
- **Nonzero-momentum symmetric coupling.** Gets no help from uniform-mode conditioning and is noisier. Systematics check only.
- **L48 or larger.** Unnecessary for a first cutoff-robust result once L12/L16 are precise.
- **Analytic continuum statement.** Because the pullback is exact and linear, the Aizenman–Duminil-Copin theorem (Gaussian scaling limit of critical nearest-neighbour lattice φ⁴ in the Griffiths–Simon class) applies verbatim to the critical NNFT measure at every λ > 0 [THEOREM]. This is an analytic complement, not a numerical demonstration. It says nothing about finite-width neural architectures.
- **Literature context.** Weisz–Wolff (2011) measured exactly this finite-size coupling (second-moment mass, fixed z = 1, 2, 4) at per-mille precision in the Ising limit with a random-current algorithm, and found PT-consistent running. A positive NNFT running result is therefore a *confirmation*. Its distinctive content is the exact-architecture estimator, the Lee–Yang control, and the sampler-cost diagnosis.

### 6.4 In-session small-volume running analysis [MEASURED-LOCAL; new prospective analysis]

Per-ensemble comparison with exact one-loop finite-lattice PT (scheme-matched), evaluated at each run's measured z and anchored at the measured L=12 value. Errors: block jackknife; for L ≥ 16 the larger of jackknife and 8-batch means. PT remainder not bounded.

| L | m₀² | z | g_R measured | g_R one-loop PT (± anchor) | pull |
|---|---|---|---|---|---|
| 6 | -0.4 | 3.0385 | 6.130 ± 0.015 | 6.192 ± 0.047 | -1.3 |
| 8 | -0.53 | 3.0295 | 5.992 ± 0.038 | 6.041 ± 0.045 | -0.8 |
| 10 | -0.589 | 3.0385 | 5.989 ± 0.059 | 5.924 ± 0.043 | +0.9 |
| 12 | -0.6223 | 3.0245 | 5.805 ± 0.041 | 5.805 ± 0.041 | +0.0 |
| 16 | -0.657 | 2.9360 | 5.889 ± 0.194 | 5.560 ± 0.038 | +1.7 |
| 24 | -0.6775 | 3.0151 | 8.266 ± 1.267 | 5.403 ± 0.036 | +2.3 |
| 24 | -0.676 | 3.1451 | 6.951 ± 1.313 | 5.498 ± 0.037 | +1.1 |
| 32 | -0.686 | 2.9990 | 7.866 ± 2.514 | 5.239 ± 0.033 | +1.0 |

Fits in g-space (independent ensembles):

| Ensembles | M1: B (B/b₁) | MCUT: B | MPT amplitude α | χ²/dof | Δχ²(α=0) | Δχ²(α=1) |
|---|---|---|---|---|---|---|
| L = 6-32 (8) | +0.0115 ± 0.0015 (0.61 ± 0.08) | +0.0064 ± 0.0096 | 0.843 ± 0.064 | 10.8/6 | 187.5 | 5.8 |
| L = 6-12 (4) | +0.0121 ± 0.0016 (0.64 ± 0.08) | +0.0161 ± 0.0109 | 0.875 ± 0.067 | 1.9/2 | 189.4 | 3.5 |
| L = 12-32 (5) | -0.0257 ± 0.0156 (-1.35 ± 0.82) | -0.1325 ± 0.0668 | -0.647 ± 0.604 | 4.0/3 | 1.0 | 6.1 |
| L = 8-32 (7) | +0.0104 ± 0.0037 (0.55 ± 0.20) | -0.0147 ± 0.0252 | 0.703 ± 0.164 | 9.9/5 | 18.3 | 3.2 |

**Reading.**
- The resolved decrease and its agreement with finite-lattice one-loop PT (pulls −1.3, −0.8, +0.9 at L = 6, 8, 10; α = 0.875 ± 0.067 with a free intercept) come from L/a = 6–12, where lattice artefacts are large but are included exactly in the one-loop kernel.
- The short L ≥ 16 runs all lie above the PT expectation (pulls +1.0 to +2.3; χ² ≈ 10.4/4). On the frozen ladder volumes alone (12–32) there is no resolved running, and α is 2.5σ below 1. This is a ≈2–2.5σ tension (p ≈ 0.01–0.03), not a measurement.
- MCUT resolves no slope anywhere. Nothing here is a cutoff-robust running coefficient.

![g_R ladder](figures/fig1_gR_ladder.png)

---

## 7. Smallest decisive test and its evidential value

**Test RG-1/Stage 1 (endpoint-first, frozen trajectory).**
- Zero-source g_R at L = 24 (m₀² = −0.6775, plus one bracketing mass) and L = 32 (m₀² = −0.686, plus one bracketing mass), each to 2%.
- Exact local sampler and exact uniform-mode estimator; per-configuration (u, Σψ², Σψ³, G(p_min)) stored.
- Combined with the in-session L = 6–12 data.

**Evidential value (decisive in either direction).**
- Under α = 1 (PT), the 12→32 contrast B₁₂,₃₂ is expected at ≈4.1–4.3σ.
- On the frozen-ladder volumes the amplitude α tightens from ±0.60 to ≈±0.17.
- If the present L ≥ 16 central values persist, Stage 1 would instead establish a ≳3σ departure from one-loop finite-size PT at z ≈ 3. That is itself a reportable result, and it would redirect Stage 2 toward z = 4, where PT is known to converge.
- Under α = 0, the one-sided false-positive rate is 2.5%.
- It also measures τ_int and per-configuration variance at L = 24/32 precisely enough to fix the Stage 2 budget to ±10%.
- It decides magnitudes on the frozen ladder: any frozen-vs-zero-source disagreement beyond the ≈0.83 rescaling would falsify the Lee–Yang calibration at large L.

**Test RG-1/Stage 2 (cutoff-robust).**
- Optimal allocation (§6.1). MCUT, M1 and MPT reported together.
- Designed for 80% power for a positive B under free L⁻² profiling.

**Resource ceilings.** Stage 1 ≤ 500 core-hours. Stage 1 + Stage 2 ≤ 2,000 core-hours total. Both fit a modest allocation, or one 8-core workstation for 1–2 weeks.

---

## 8. Decision matrix

| Window | Action | Compute | Prerequisites | Output / gate |
|---|---|---|---|---|
| **31 Oct** | Adopt the two-part claim (§1). Retire "positive interaction" as evidence. Add the Lee–Yang reinterpretation of frozen magnitudes. Record the L12 and L24/32 resolutions as a separately labelled new analysis. | 0 (done here; reproducible locally in ≈1 core-hour for a reduced L12 rerun) | Your review of this package | Claim sentence fixed; G2 untouched |
| **6 Nov** | (a) Lee–Yang re-analysis of archived per-draw (a₁, q(η), s(η), z) for L12–32 with `reanalyze_archive_ly.py`, labelled NEW_ANALYSIS. (b) Methods section: exact uniform-mode conditioning, Lee–Yang structure, pinning law, small-volume ladder. | minutes | Per-radius bootstrap draws must exist on the Mac; if only fitted (a₁, a₃) exist → report NOT_TESTABLE | Honest wide intervals (L12 ≈ [4.6, 9.6]); LY-consistency diagnostics s<q, G_s<G_q, t* |
| **Later (authorization)** | RG-1 Stage 1, then Stage 2 under the contract (§9) | ≤ 500, then ≤ 1,500 core-hours | Your authorization; sampler and analysis frozen and hashed before production; capture-before-validation | Stage 1 gate: z within 0.3% of 3 after interpolation; reweighting identity and LY inequalities pass; projected MCUT σ_B ≤ b₁/2.5, otherwise stop |

---

## 9. Measurement contract (RG-1) and the single escalation

**Claim.** At fixed z = 3 on the frozen trajectory (λ = 0.4, canonical action, exact Gaussian-latent pullback), g_R(L) decreases with L/a over L/a = 12, 16, 24, 32. The inverse-coupling slope B is resolved as positive under M1 **and** under free-L⁻² MCUT, and compared with the scheme-matched one-loop lattice kernel plus the universal two-loop term.

**Estimand.** g_R(L) = −z⁴κ₄(X)/κ₂(X)² at h = 0, with the project mass convention z = L·2asinh(μ/2), μ² = p̂²_min/(G(0)/G(p_min) − 1). Interpolated to exactly z = 3 from two bracketing m₀² values, with interpolation error propagated.

**Observable and storage.**
- Per sweep-set: (u, Σψ², Σψ³, G(p_min) per axis).
- Exact conditional (RB) and raw moments as paired estimators.
- Per-configuration binaries with SHA-256 recorded at write time (capture before validation).
- Restart identity in the manifest.

**Design.** L12 at 0.2%, L16 at 0.3%, L24/L32 at 1.35% (relative σ_g after interpolation). ≥8 independent chains per (L, m₀²). Blocks ≥50 τ_int. Thermalisation ≥5×10³ sweeps, verified by split-chain stationarity.

**Negative and positive controls (predeclared).**
1. λ = 0: g = 0 and analytic G(p).
2. Reproduce the in-session L12 values: z = 3.0245(11), g = 5.805(41).
3. Lebowitz sign and the Lee–Yang inequalities via same-ensemble reweighting at η = 4.5 and 6.
4. One direct sourced run per volume must match the reweighting identity.
5. Cross-sampler check at L = 12 and 16 against kernel-matched latent HMC (mref² = m_R², trajectory length ≈ π/2).

**Nuisances.**
- z: interpolation plus the local derivative.
- Autocorrelation: Γ-method plus replica scatter.
- Cutoff: M1, MCUT, MPT (exact one-loop lattice kernel at measured z), and the L ≥ 16 subset.
- There is no source nuisance, because the estimand is at h = 0.

**Calibration and gates.**
- 2,000 synthetic campaigns with the measured covariance under α = 0 and α = 1.
- Stop after Stage 1 if the projected MCUT σ_B exceeds b₁/2.5.
- No change to estimator, models or allocation once the first L32 production block is opened.

**Independent validation.** A second sampler at L ≤ 16 must agree within 2σ on z and g. Analysis is rerun by a reviewer who has not seen the B values.

> **Single decision requested:** authorize RG-1 with a hard ceiling of **2,000 core-hours**, staged (Stage 1 ≤ 500 with the predeclared stop gate), on any machine you control, using the packaged exact sampler or a reviewed equivalent.
>
> Declining it closes the running question at the in-session result (theory-assisted, cutoff-confounded) plus the frozen NOT_RESOLVED. No other work can improve the running evidence without new sampling.

---

## 10. Reproducibility

Everything in this verdict was produced in this session from the files listed below. Re-run commands are in `README_PACKAGE.md`.

- `src/phi4_local.cpp`, `src/phi4_local2.cpp`: independent exact sampler (C++17).
- `analysis/`:
  - `zm.py`: exact conditioning, reweighting, jackknife, characteristic function
  - `ly_bracket.py`: Lee–Yang LP and estimators
  - `analyze_ensemble.py`, `analyze_source.py`, `l12_followup.py`
  - `scaling_table.py`, `pt_compare.py`, `running_fits.py`, `running_fits_g.py`, `batch_check.py`
  - `reanalyze_archive_ly.py` (archive tool), `make_pseudo_draws.py`
  - `design_info.py`, `hmc_mismatch.py`, `forecast.py`, `figures.py`, `fill_verdict.py` (tables are generated, not hand-copied)
  - validation and pilot helpers: `validate_gauss.py`, `validate_L4.py`, `test_ly*.py`, `ly_zeros.py`, `pilot.py`, `quick.py`, `halves.py`
- `results/`: every JSON quoted here.
- `SHA256SUMS.txt`: checksums for the package and for every per-configuration data file. The data are shipped separately as `phi4_rescue_configdata.zip`.

**Validation performed.**
- Gaussian limit: G(0), G(p_min) and ⟨Σψ²⟩ match exact values within 1σ.
- Zero-mode Gibbs vs local-only chains at L=4, λ=0.4: U₄ agrees within 1.4σ.
- Reweighting identity vs direct sourced runs: ≤1.2σ.
- Lee–Yang LP on exactly solvable measures: bracket contains the truth.
- One-loop kernel reproduces 0.0192746.
- Project z at L = 12 and 32 reproduced.

**Not established here.**
- Live HIPSTER state.
- The October 27 ZIP contents.
- Measured τ_int of the configured latent HMC (only derived).
- Two-loop finite-lattice coefficients (only the continuum two-loop size is used).
- Any continuum or cutoff-robust running claim.

## Sources

- [P. Weisz and U. Wolff, Triviality of φ⁴₄ theory: small volume expansion and new data, Nucl. Phys. B 846 (2011) 316](https://arxiv.org/abs/1012.0404)
- U. Wolff, Precision check on triviality of φ⁴ theory by a new simulation method, Phys. Rev. D 79 (2009) 105002
- [M. Aizenman and H. Duminil-Copin, Marginal triviality of the scaling limits of critical 4D Ising and φ⁴₄ models, Ann. Math. 194 (2021) 163](https://annals.math.princeton.edu/2021/194-1/p03)
- B. Simon and R. B. Griffiths, The (φ⁴)₂ field theory as a classical Ising model, Commun. Math. Phys. 33 (1973) 145
- C. M. Newman, Zeros of the partition function for generalized Ising systems, Commun. Pure Appl. Math. 27 (1974) 143
- J. L. Lebowitz, GHS and other inequalities, Commun. Math. Phys. 35 (1974) 87
