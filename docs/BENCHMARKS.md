# Benchmarks

Appended by `python dev/bench.py` (see its --help). Compare runs on the same machine; the cloud build box is not Chris's PC.

## 2026-11-23 09:48 UTC - Linux (cloud build box) x86_64, Python 3.13.18, numpy 2.5.3, --gb 0.6

| step | detail | time (s) |
|---|---|---|
| write synthetic SEG-Y | 0.60 GB, 336 x 420 x 1000 | 11.566 |
| import 3D (two float16 stores) | 166 MB/s | 3.595 |
| inline section (random, x10) | 1.2 ms each | 0.012 |
| crossline section (random, x10) | 1.3 ms each | 0.013 |
| time slice (random, x10) | 0.4 ms each | 0.004 |
| arbitrary line corner to corner |  | 0.012 |
| well index build (10,000 wells) |  | 0.034 |
| well index box query (view) | 102 wells | 0.000 |
| build 1,000 synthetic laterals |  | 0.084 |
| colour-batch laterals for the map (LOD 25 m) | 16 curves | 0.070 |
| peak RAM of the benchmark process | 1339 MB |  |

Notes (run 1): slices come from memory-mapped float16 stores, so section latency stays in milliseconds whatever
the volume size (one inline / crossline / time slice is read, never the volume). "Peak RAM" includes the pages
of the memory-mapped SEG-Y and stores that the import touched (file-backed, given back by the OS under memory
pressure); arrays held by the import itself are about 0.1-0.4 GB (trace chunk of 20,000 traces + header arrays).
On Chris's PC: run `python dev/bench.py --gb 5 --dir D:\bench` (needs about 3x the size free) and compare.
