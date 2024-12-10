#!/bin/bash 

set -o nounset
set -o pipefail

# Script for testing what risc0 v2 circuit files compile properly.
# The path is bounded to a location in my dev machine.
# If you have access to the circuit and want to run it change the path 
# to the location where you have it.

CIRCUIT_PATH=$HOME/code/veridise/risczero-wip/zirgen/circuit/rv32im/v2/dsl/
DEST=../risc0_v2_circuit_build_results/$(date +%Y-%m-%d-%H-%M)
# BAZELFLAGS=--local_resources=cpu='HOST_CPUS*0.5' 
BAZELFLAGS= 
ZKLANG_FLAGS=

mkdir -p $DEST

function die {
  echo $@
  exit 1
}

function zir_files {
  find $CIRCUIT_PATH -name '*.zir'
}

function build_zklang {
  bazel build $BAZELFLAGS //zklang:zklang  
}

function build_zir {
  zir=$(realpath $1)
  name=$(basename $zir)
  mlir_out=$(realpath $DEST/$name.mlir)
  stdout=$DEST/$name.stdout
  stderr=$DEST/$name.stderr
  errcode=$DEST/$name.errcode
  echo "[=] Building $name..."
  bazel run $BAZELFLAGS //zklang:zklang -- -o $mlir_out $zir $ZKLANG_FLAGS > $stdout 2> $stderr &
  tail -F $stderr --pid=$!
  echo $? > $errcode
  echo " ============= $name =============="
  echo -n "Return code: "
  cat $errcode
  echo " ========== $name stdout =========="
  tail $stdout
  echo " ========== $name stderr =========="
  tail $stderr
  
}

# Build first to avoid filling run output with build logs
build_zklang || die "Failed to build zklang!"
for zir in $(zir_files); do 
  build_zir $zir
done
