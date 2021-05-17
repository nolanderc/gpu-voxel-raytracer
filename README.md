
# Interactive Voxel Path Tracer

## Building from Soruce

Requires Rust 1.50 or later (may work with earlier version, not tested).

In order to build and run the executable, run the following command:

```bash
cargo run --release
```

If you only want to build the executable, simply run:

```bash
cargo build --release
```

## Prebuilt Executables

Some prebuilt executables exist for MacOS Darwin and Ubuntu Linux. You will find
these in the `bin/` directory. Note that these executables must be run from this
root directory, therefore run them with the following command:

```bash
./bin/voxel-OS
```

where `OS` corresponds to your operating system.


## Building the Shaders

Shaders are compiled from GLSL to SPIR-V if they are out of date. This is done
using the [`glslValidator` compiler](https://github.com/KhronosGroup/glslang).
The shaders should be up-to-date when you first open this repository, so
installing this executable should not be required. However, if you want to edit
the shaders, make sure that the compiler is installed.

