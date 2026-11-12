#!/bin/sh
while kill -0 1839 2>/dev/null; do sleep 10; done
./bin/phi4_local2 32 -0.686 0.4 0.0 3000 20000 2 1 3201 data/L32_h0_s3201.bin 1 > data/L32_h0_s3201.json
./bin/phi4_local2 8 -0.53 0.4 0.0 3000 400000 2 1 801 data/L8_h0_s801.bin 1 > data/L8_h0_s801.json
./bin/phi4_local2 12 -2.93 2.0 0.0 3000 200000 2 1 1293 data/lam2_L12_m-2.93.bin 1 > data/lam2_L12_m-2.93.json
./bin/phi4_local2 16 -0.657 0.4 0.0 3000 250000 2 1 1601 data/L16_h0_s1601.bin 1 > data/L16_h0_s1601.json
