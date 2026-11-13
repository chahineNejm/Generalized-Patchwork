# Result Archive Inbox

Put downloaded experiment ZIPs in this folder. Archive contents stay outside Git;
this guide and INDEX.csv are tracked. Keep the original ZIP unchanged.

INDEX.csv records each archive's location, experiment, datasets, factors, seeds
and verification status. Add a row when an archive arrives; do not infer completed
runs from its filename. Inspect saved plans, configurations, metrics and failure
records before marking it verified. Record unavailable information as UNVERIFIED.
Overlapping seeds are not independent repeats. Pool only matching protocols,
budgets, hyperparameters and source/numerical implementations.

The two archives indexed initially are still in the project root. Their paths
are relative to the project root; update the paths if you move them here.
The FLIGHT notebook is planned, not a completed result archive.
