#!/bin/sh
while kill -0 1840 2>/dev/null; do sleep 10; done
./bin/phi4_local2 24 -0.676 0.4 0.0 3000 40000 2 1 2401 data/L24_h0_s2401.bin 1 > data/L24_h0_s2401.json
./bin/phi4_local2 6 -0.40 0.4 0.0 3000 600000 2 1 601 data/L6_h0_s601.bin 1 > data/L6_h0_s601.json
./bin/phi4_local2 10 -0.589 0.4 0.0 3000 400000 2 1 1001 data/L10_h0_s1001.bin 1 > data/L10_h0_s1001.json
./bin/phi4_local2 24 -3.00 2.0 0.0 3000 12000 2 1 2493 data/lam2_L24_m-3.00.bin 1 > data/lam2_L24_m-3.00.json
./bin/phi4_local2 24 -2.96 2.0 0.0 3000 12000 2 1 2494 data/lam2_L24_m-2.96.bin 1 > data/lam2_L24_m-2.96.json
