# UKHousing Source Snapshots

This reconstructs the 2018-flats task, not the verified VNNGP dataset snapshot.
The source URLs are mutable. A pin identifies exact bytes, not a permanent
upstream release. Future changes must fail until explicitly verified/approved.

## Verified 2026-11-12

- Official listing: https://www.gov.uk/government/statistical-data-sets/price-paid-data-yearly-file
- Prices: https://price-paid-data.publicdata.landregistry.gov.uk/pp-2018.csv
- Observed SHA-256: `dbf8e95e719630fa8b8d0c90bea52e9e282e7bb6c938951737d4ed65c7655eda`
- Size: 180,179,408 bytes; 1,037,647 transactions.
- Two complete HTTPS downloads have identical hashes. Observed server
  Last-Modified: 2026-10-28.
- Postcodes: https://data.freemaptools.com/download/full-uk-postcodes/ukpostcodes.zip
- Unchanged SHA-256: `d75818f49dc7de39c61fcc0577d27eb548caaa4a28509adc9f602da5cbc48906`.
- Checks: exact 16-column price schema, unique transaction IDs, positive prices,
  2018 sale dates; postcode ZIP integrity/schema and unique postcode keys.
- Unchanged reconstruction retains 185,257 flats, versus 185,254 previously.
  It records 185,943 price-eligible flats and 612 unmatched/nonfinite coordinates.
- Compared with the previous CSV: 30 new IDs, 49 removed IDs, 63 changed records.

These are locally observed hashes from HTTPS downloads, not publisher-signed
checksums. Audit artifacts and raw files remain in ignored
`data/ukhousing_verified_20261112/`; no dataset is committed.

The earlier price pin was
`2e9fb3872f0800c86458ea62a2c999195a7b55b5da17b4a8a96705c122590b27`,
with 1,037,666 raw rows. Its local files and historical results remain intact.

Preparation now records `source_snapshot="2026-11-12"` and both source hashes.
Filters, geographic region, scaling, seed, split proportions and model settings
are unchanged. A revised set/order of rows can change split membership despite
using the same seed; do not pool results across snapshots as identical splits.
Cached files with an old checksum still raise; they are never replaced silently.
