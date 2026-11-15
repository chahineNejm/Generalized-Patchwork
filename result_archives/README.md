# Result Archive Inbox

Put downloaded experiment ZIPs in this folder. Archive contents stay outside Git;
this guide and INDEX.csv are tracked. Keep the original ZIP unchanged.

INDEX.csv records each archive's location, experiment, datasets, factors, seeds,
SHA256 and verification status. Paths are relative to the project root. Add a row
when an archive arrives; do not infer completed
runs from its filename. Inspect saved plans, configurations, metrics and failure
records before marking it verified. Record unavailable information as UNVERIFIED.
Overlapping seeds are not independent repeats. Pool only matching protocols,
budgets, hyperparameters and source/numerical implementations.

All four received result ZIPs now reside in this folder. SHA256 hashes of the
first three were checked before/after moving;the rank64 archive was received here
and indexed with its SHA256. Contents and filenames are unchanged. Old audit
records retain their historical input paths, not current archive locations.
Shared-kernel completeness/outer ZIP integrity are checked, but its full metric
audit and matched local-kernel comparison remain pending. Do not double-count
its embedded collection/per-run ZIPs as independent runs.
The FLIGHT/TCO.L2/ICETHICK rank64 overnight archive contains9 completed seed31
runs. Saved predictions,metrics,matched initialization and FLIGHT source rows were
checked;see results/stitch_rank64_audit_20261114/report.json. Single-seed results
do not establish repeatability or significance.
