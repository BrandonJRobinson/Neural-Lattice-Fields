// Independent 4D lattice phi^4 sampler for the NNFT scientific rescue review.
//
// Action (identical convention to the project's actions/phi4.py):
//   S = sum_x [ 1/2 sum_mu (phi_{x+mu}-phi_x)^2 + 1/2 m0^2 phi_x^2 + lambda phi_x^4 - h phi_x ]
// Periodic L^4 lattice.  Local single-site form: A phi^2 - B phi + lambda phi^4 with
//   A = d + m0^2/2,  B = sum_{y~x} phi_y + h.
//
// Updates (each leaves the target exactly invariant):
//   (1) local Gaussian-proposal Metropolis ("heat-bath + quartic accept") at every site,
//   (2) n_or local reflections phi -> B/A - phi accepted with exp(-lambda(phi'^4-phi^4)),
//   (3) exact Gibbs draw of the uniform mode u given psi = phi - u (rejection sampling
//       from a Gaussian envelope that dominates the log-concave conditional).
//
// Per measurement we write 4 doubles:  u = mean(phi),  S2 = sum psi^2,  S3 = sum psi^3,
//   Gp = (1/4) sum_mu |phi_hat(p_min e_mu)|^2   (unitary Fourier normalisation, as the project).
// These are sufficient statistics for every zero-mode observable at every uniform source.
//
// This code shares no source with the project; it is a cross-check, not production software.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

struct Rng {  // xoshiro256**
  uint64_t s[4];
  static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  explicit Rng(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; ++i) {  // splitmix64
      z += 0x9e3779b97f4a7c15ULL;
      uint64_t t = z;
      t = (t ^ (t >> 30)) * 0xbf58476d1ce4e5b9ULL;
      t = (t ^ (t >> 27)) * 0x94d049bb133111ebULL;
      s[i] = t ^ (t >> 31);
    }
  }
  inline uint64_t next() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
  }
  inline double uniform() { return (next() >> 11) * 0x1.0p-53; }
  inline double uniform_pos() { double u; do { u = uniform(); } while (u <= 0.0); return u; }
  bool has_spare = false; double spare = 0.0;
  inline double normal() {
    if (has_spare) { has_spare = false; return spare; }
    double u1 = uniform_pos(), u2 = uniform();
    double r = std::sqrt(-2.0 * std::log(u1));
    spare = r * std::sin(2.0 * M_PI * u2); has_spare = true;
    return r * std::cos(2.0 * M_PI * u2);
  }
};

int main(int argc, char** argv) {
  if (argc < 11) {
    std::fprintf(stderr,
      "usage: %s L m0sq lambda h n_therm n_meas n_or meas_every seed outfile [zero_mode_gibbs=1]\n", argv[0]);
    return 2;
  }
  const int L = std::atoi(argv[1]);
  const double m0sq = std::atof(argv[2]);
  const double lam = std::atof(argv[3]);
  const double h = std::atof(argv[4]);
  const long n_therm = std::atol(argv[5]);
  const long n_meas = std::atol(argv[6]);
  const int n_or = std::atoi(argv[7]);
  const int meas_every = std::atoi(argv[8]);
  const uint64_t seed = std::strtoull(argv[9], nullptr, 10);
  const std::string outfile = argv[10];
  const int use_zm = (argc > 11) ? std::atoi(argv[11]) : 1;

  const int d = 4;
  const long V = (long)L * L * L * L;
  const double A = d + 0.5 * m0sq;
  if (A <= 0) { std::fprintf(stderr, "A<=0\n"); return 2; }
  const double sigma_prop = std::sqrt(0.5 / A);

  // neighbour table
  std::vector<int> nb(8 * V);
  std::vector<int> coord(4 * V);
  for (long x = 0; x < V; ++x) {
    long r = x; int c[4];
    for (int mu = 0; mu < 4; ++mu) { c[mu] = r % L; r /= L; }
    for (int mu = 0; mu < 4; ++mu) coord[4 * x + mu] = c[mu];
    long stride = 1;
    for (int mu = 0; mu < 4; ++mu) {
      int cp = (c[mu] + 1) % L, cm = (c[mu] - 1 + L) % L;
      nb[8 * x + 2 * mu] = (int)(x + (cp - c[mu]) * stride);
      nb[8 * x + 2 * mu + 1] = (int)(x + (cm - c[mu]) * stride);
      stride *= L;
    }
  }
  std::vector<double> cosv(L), sinv(L);
  for (int t = 0; t < L; ++t) { cosv[t] = std::cos(2 * M_PI * t / L); sinv[t] = std::sin(2 * M_PI * t / L); }

  Rng rng(seed);
  std::vector<double> phi(V, 0.0);
  for (long x = 0; x < V; ++x) phi[x] = 0.1 * rng.normal();

  long acc_hb = 0, try_hb = 0, acc_or = 0, try_or = 0, zm_rej = 0, zm_draws = 0, zm_skipped = 0;

  auto local_B = [&](long x) {
    const int* n = &nb[8 * x];
    return phi[n[0]] + phi[n[1]] + phi[n[2]] + phi[n[3]] + phi[n[4]] + phi[n[5]] + phi[n[6]] + phi[n[7]] + h;
  };

  auto sweep = [&]() {
    for (long x = 0; x < V; ++x) {
      double B = local_B(x);
      double c = B / (2.0 * A);
      double old = phi[x];
      double prop = c + sigma_prop * rng.normal();
      double o2 = old * old, p2 = prop * prop;
      double dS = lam * (p2 * p2 - o2 * o2);
      ++try_hb;
      if (dS <= 0.0 || rng.uniform() < std::exp(-dS)) { phi[x] = prop; ++acc_hb; }
    }
    for (int k = 0; k < n_or; ++k) {
      for (long x = 0; x < V; ++x) {
        double B = local_B(x);
        double old = phi[x];
        double prop = B / A - old;
        double o2 = old * old, p2 = prop * prop;
        double dS = lam * (p2 * p2 - o2 * o2);
        ++try_or;
        if (dS <= 0.0 || rng.uniform() < std::exp(-dS)) { phi[x] = prop; ++acc_or; }
      }
    }
  };

  // exact zero-mode Gibbs: conditional density exp(-(a4 u^4 + a2 u^2 + a1 u)), u = mean(phi)
  auto zero_mode = [&](double& u_out, double& S2_out, double& S3_out) {
    double S1 = 0.0;
    for (long x = 0; x < V; ++x) S1 += phi[x];
    double u = S1 / V;
    double S2 = 0.0, S3 = 0.0;
    for (long x = 0; x < V; ++x) { double p = phi[x] - u; double p2 = p * p; S2 += p2; S3 += p2 * p; }
    if (use_zm) {
      const double a4 = lam * V;
      const double a2 = 0.5 * m0sq * V + 6.0 * lam * S2;
      const double a1 = 4.0 * lam * S3 - h * V;
      // Exact: the decision depends on psi only, which this step leaves invariant; skipping preserves p(u|psi).
      if (a2 <= 0) { ++zm_skipped; u_out = u; S2_out = S2; S3_out = S3; return; }
      // unique stationary point of f(u)=a4 u^4 + a2 u^2 + a1 u (f convex), Newton from -a1/(2a2)
      double u0 = -a1 / (2.0 * a2);
      for (int it = 0; it < 60; ++it) {
        double f1 = 4 * a4 * u0 * u0 * u0 + 2 * a2 * u0 + a1;
        double f2 = 12 * a4 * u0 * u0 + 2 * a2;
        double step = f1 / f2; u0 -= step;
        if (std::fabs(step) < 1e-15 * (1 + std::fabs(u0))) break;
      }
      auto f = [&](double v) { return a4 * v * v * v * v + a2 * v * v + a1 * v; };
      // envelope: f(v) >= f(u0) + a2 (v-u0)^2  (since f'' >= 2 a2 and f'(u0)=0)
      const double f0 = f(u0);
      const double sd = std::sqrt(0.5 / a2);
      double unew;
      while (true) {
        double v = u0 + sd * rng.normal();
        double logacc = -(f(v) - f0 - a2 * (v - u0) * (v - u0));
        ++zm_draws;
        if (logacc >= 0.0 || std::log(rng.uniform_pos()) < logacc) { unew = v; break; }
        ++zm_rej;
      }
      double delta = unew - u;
      for (long x = 0; x < V; ++x) phi[x] += delta;
      u = unew;
    }
    u_out = u; S2_out = S2; S3_out = S3;
  };

  auto gpmin = [&]() {
    double re[4] = {0, 0, 0, 0}, im[4] = {0, 0, 0, 0};
    for (long x = 0; x < V; ++x) {
      const int* c = &coord[4 * x];
      double p = phi[x];
      for (int mu = 0; mu < 4; ++mu) { re[mu] += p * cosv[c[mu]]; im[mu] += p * sinv[c[mu]]; }
    }
    double g = 0.0;
    for (int mu = 0; mu < 4; ++mu) g += re[mu] * re[mu] + im[mu] * im[mu];
    return g / (4.0 * V);
  };

  auto t0 = std::chrono::steady_clock::now();
  double u, S2, S3;
  for (long i = 0; i < n_therm; ++i) { sweep(); zero_mode(u, S2, S3); }
  auto t1 = std::chrono::steady_clock::now();

  FILE* fo = std::fopen(outfile.c_str(), "wb");
  if (!fo) { std::fprintf(stderr, "cannot open output\n"); return 2; }
  std::vector<double> buf; buf.reserve(4 * 4096);
  long acc_hb0 = acc_hb, try_hb0 = try_hb, acc_or0 = acc_or, try_or0 = try_or;
  const long zm_skipped_therm = zm_skipped;
  for (long i = 0; i < n_meas; ++i) {
    for (int k = 0; k < meas_every; ++k) { sweep(); zero_mode(u, S2, S3); }
    // after zero_mode, phi has the freshly drawn u; S2,S3 are of psi (unchanged by the shift)
    double Gp = gpmin();
    buf.push_back(u); buf.push_back(S2); buf.push_back(S3); buf.push_back(Gp);
    if (buf.size() >= 4 * 4096) { std::fwrite(buf.data(), sizeof(double), buf.size(), fo); buf.clear(); }
  }
  if (!buf.empty()) std::fwrite(buf.data(), sizeof(double), buf.size(), fo);
  std::fclose(fo);
  auto t2 = std::chrono::steady_clock::now();
  double th = std::chrono::duration<double>(t1 - t0).count();
  double tm = std::chrono::duration<double>(t2 - t1).count();
  std::printf("{\"L\":%d,\"m0sq\":%.10g,\"lambda\":%.10g,\"h\":%.12g,\"n_therm\":%ld,\"n_meas\":%ld,"
              "\"n_or\":%d,\"meas_every\":%d,\"seed\":%llu,\"zero_mode_gibbs\":%d,"
              "\"acc_hb\":%.6f,\"acc_or\":%.6f,\"zm_rejection_rate\":%.6f,"
              "\"zm_skipped_therm\":%ld,\"zm_skipped_meas\":%ld,\"therm_seconds\":%.3f,\"meas_seconds\":%.3f,\"seconds_per_sweepset\":%.6g}\n",
              L, m0sq, lam, h, n_therm, n_meas, n_or, meas_every, (unsigned long long)seed, use_zm,
              double(acc_hb - acc_hb0) / std::max(1L, try_hb - try_hb0),
              double(acc_or - acc_or0) / std::max(1L, try_or - try_or0),
              zm_draws ? double(zm_rej) / zm_draws : 0.0, zm_skipped_therm, zm_skipped - zm_skipped_therm, th, tm,
              tm / std::max(1L, n_meas * (long)meas_every));
  return 0;
}
