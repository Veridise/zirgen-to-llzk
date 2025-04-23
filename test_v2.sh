#!/bin/bash

set -o nounset
set -o pipefail

# Script for testing what risc0 v2 circuit files compile properly.
# The path is bounded to a location in my dev machine.
# If you have access to the circuit and want to run it change the path
# to the location where you have it.

CIRCUIT_PATH=$HOME/code/veridise/zir-benchmarks/rv32im-v2/
KECCAK_PATH=$HOME/code/veridise/zir-benchmarks/keccak2/
DEST=../risc0_v2_circuit_build_results/$(date +%Y-%m-%d-%H-%M)
ZKLANG_FLAGS=
DERIVATION='.?submodules=1#withGCC'

mode=${1:-""}

if [ $mode == 'nix' ]; then

function build_zklang {
  nix build "$DERIVATION"
}

function run_zklang {
  nix run "$DERIVATION" -- $@
}

else

function build_zklang {
  cmake --build build/Debug --target zklang
}

function run_zklang {
  build/Debug/bin/zklang $@
}

fi


mkdir -p $DEST

function die {
  echo $@
  exit 1
}

function zir_files {
  workdir=$(realpath $1)
  find $workdir -name '*.zir'
}


function build_zir {
  zir=$(realpath $1)
  workdir=$(realpath $2)
  dst=$(realpath $3)
  name=$(basename $zir)
  mlir_out=$(realpath $dst/$name.mlir)
  stdout=$dst/$name.stdout
  stderr=$dst/$name.stderr
  errcode=$dst/$name.errcode
  echo "[=] Building $name..."
  run_zklang -o $mlir_out -I $workdir $zir $ZKLANG_FLAGS > $stdout 2> $stderr
  # tail -F $stderr --pid=$!
  echo $? > $errcode
  echo " ============= $name =============="
  echo -n "Return code: "
  cat $errcode
  echo " ========== $name stdout =========="
  tail $stdout
  echo " ========== $name stderr =========="
  tail $stderr

}

function build_project {
  name=$1
  workdir=$(realpath $2)
  dst=$(realpath $3)/$name
  mkdir -p $dst
  for zir in $(zir_files $workdir); do
    build_zir $zir $workdir $dst
  done
}

# Build first to avoid filling run output with build logs
build_zklang || die "Failed to build zklang!"
build_project "rv32im-v2" $CIRCUIT_PATH $DEST
build_project keccak2 $KECCAK_PATH $DEST
