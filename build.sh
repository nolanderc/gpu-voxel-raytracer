#!/bin/bash

cargo build --release
mkdir -p bin
cp ./target/release/voxel "./bin/voxel-$OSTYPE"

