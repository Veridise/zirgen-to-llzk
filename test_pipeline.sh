#!/usr/bin/bash 

zkc_opt=bazel-bin/ZirToZkir/tools/zkc-opt

read -r -d '' pipeline <<'EOF'
builtin.module(
  inject-builtins
)
EOF

$zkc_opt --pass-pipeline="$pipeline" $1
