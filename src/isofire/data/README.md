# Auxiliary data

The retrieval needs two files at runtime:

- `fire_library.csv`  - fire endmember library (default for `--fire_lib_csv`)
- `kurudz_0.1nm.dat`  - Kurudz solar irradiance (default for `--solar_irr_file`)

These ship inside the installed `isofire` package as package data. The
resolver in `airborne.py` looks for them in this order:

1. `--fire_lib_csv` / `--solar_irr_file` flag values (always wins)
2. `$ISOFIRE_DATA_DIR` (if the environment variable is set)
3. `<package_root>/data/`  (this directory; bundled in every install)
4. `<repo_root>/data/`     (legacy; only present if you run from a checkout
                            that still has a top-level `data/`)

Drop your own copies here to override.
