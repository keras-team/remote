## Summary

- Add autoscaling support to accelerator node pools — pools scale to zero when idle by default (min=0, max=current fixed count), saving costs on expensive GPU/TPU nodes
- Add `--no-autoscale` CLI flag to `up` and `pool add` for opting out
- Add `pool autoscale <pool_name> --enable/--disable` command to toggle autoscaling on existing pools
- Add interactive autoscale prompt in the `up` flow when selecting an accelerator interactively
- Fix `up` command to preserve existing pools on re-run (previously could delete pools added via `pool add`)
- Display autoscaling status ("Enabled"/"Disabled") in `status` and `pool list` output
