# dev_tools (development panels and artificial data only; refuse lockbox roles)

| Path | Purpose |
| --- | --- |
| `capture_records.py` | run named targets (`name:k`) and store per-fold SER records (`--set key=value` for config overrides) |
| `replay_ser.py` | replay SER gate variants on stored records without refitting |
| `synthetic/` | artificial stress-test bank: `synth_bank.py` (generator), `run_synth.py` (resumable runner), `analyze_synth.py`, spec, first-slice results |
| `diagnostics/` | one-off diagnostics for the weak panels and the three-inhibitor crash (container paths inside; adapt before running) |
| `results/` | printed replay results quoted in `protocol/DEVELOPMENT_LOG.md` |

The generated bank itself (about 115 MB) is not committed; it is rebuilt by `synth_bank.py` (seed 20261114).
