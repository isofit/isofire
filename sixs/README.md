# 6sMultiComponent

Unofficial 6S for easy isofit integration and argument updates

This is an unofficial repository for the Second Simulation of a Satellite Signal in the Solar Spectrum vector (6S) code.  Please see the [official site](https://salsa.umd.edu/6spage.html) for documentation, additional aerosol models, and the official code.  The initial upload here used 6SV2.1 (2014), downloaded and included in 2025.  Additional updates that may be implemented in this repository are not necessary sanctioned by the core development team.  That rationalization for this asynchronous upload is to support additional capabilities (mainly inputs and outputs) from 6S that support additional sRTMnet emulator capabilities, which many ISOFIT workflows utilize.

The ISOFIT team expresses sincere gratitude towards the 6S team for all of their efforts.


This `sixs/` directory is a copy of the
[6sMultiComponent](https://github.com/isofit/6sMultiComponent) repo and is not kept up to date. The top-level `airborne` retrieval calls 6S as a subprocess at runtime; the location of the compiled binary is passed through `--sixs_dir`.

After cloning ISOFire-science:

```bash
cd sixs/Sixs
make                    # builds the sixsV2.1 binary (needs gfortran)
cd ../..
```

Then run the retrieval with `--sixs_dir ./sixs/Sixs/`. The
`apptainer/airborne.def` recipe builds 6S automatically during image
creation and bakes the default path into the container, so apptainer users
do not need to do anything special.

The compiled binary itself (`Sixs/sixsV2.1`) is intentionally **not**
checked in: it is platform-specific, and a binary built on one host will
not run on another. See `sixs/.gitignore`.
