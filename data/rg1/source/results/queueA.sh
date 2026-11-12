#!/bin/sh
while kill -0 484 2>/dev/null; do sleep 10; done
./bin/phi4_local 24 -0.676 0.4 0.0 3000 40000 2 1 2401 data/L24_h0_s2401.bin 1 > data/L24_h0_s2401.json
./bin/phi4_local 12 -0.6223 0.4 0.0026041666666666665 3000 300000 2 1 1245 data/L12_eta4p5_s1245.bin 1 > data/L12_eta4p5_s1245.json
./bin/phi4_local 8 -0.53 0.4 0.0 3000 400000 2 1 801 data/L8_h0_s801.bin 1 > data/L8_h0_s801.json
./bin/phi4_local 12 -2.93 2.0 0.0 3000 200000 2 1 1293 data/lam2_L12_m-2.93.bin 1 > data/lam2_L12_m-2.93.json
./bin/phi4_local 16 -0.657 0.4 0.0 3000 250000 2 1 1601 data/L16_h0_s1601.bin 1 > data/L16_h0_s1601.json
