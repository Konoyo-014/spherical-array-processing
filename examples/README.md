# Examples

Core examples live under `examples/core`. They demonstrate individual API areas
such as SH transforms, fixed beamforming, and DOA estimation.

Tutorial examples live under `examples/tutorials`. They are short end-to-end
pipelines that raise an error if the numerical checks fail, so they are useful
both for learning and for sanity-checking a local checkout.

Notebook tutorials live under `examples/notebooks`. They contain the same style
of workflow, split into smaller cells for readers who prefer Jupyter.

Run scripts from the repository root:

```bash
python examples/core/basic_usage.py
python examples/tutorials/01_sht_and_beamforming.py
python examples/tutorials/02_simulated_doa_pipeline.py
python examples/tutorials/03_modal_equalization_pipeline.py
```
