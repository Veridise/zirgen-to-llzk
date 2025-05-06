# Tool Guides {#tools}

# zklang {#zklang}

`zklang` is the primary tool provided by this project.
`zklang` is the frontend compiler that converts Zirgen source code into different
MLIR dialects. The end-to-end workflow of `zklang` is to perform the following translations:
1. zirgen source code to zirgen AST
2. zirgen AST to ZHL, an MLIR dialect designed by Risc0 to represent zirgen programs (see \ref ZHLDialect "the ZHL dialect documentation").
3. ZHL to ZML, an MLIR dialect designed by Veridise as an intermediary dialect between ZHL and LLZK (see \ref ZMLDialect "the ZML dialect documentation").
4. ZML to LLZK, Veridise's ZK circuit IR (see [the LLZK project site][llzk-site] for more information about the LLZK dialect).

Generally, users will use `zklang` to translate zirgen circuits to LLZK bytecode (which is more efficient for downstream tooling to operate on),
but `zklang` can optionally emit any of the above intermediate representations for inspection or debugging purproses.

## zklang Options

`zklang` inherits some options from `mlir` utilites, but provides the following unique settings:

```
-I <path>           - Add include path
--emit=<value>      - The kind of output desired
  =ast              -   Output the AST
  =zhl              -   Output untyped high level ZIR IR
  =zml              -   Output typed medium level ZIR IR
  =zmlopt           -   Output typed medium level ZIR IR with separate compute and constrain functions
  =llzk             -   Output LLZK IR
--emit-bytecode     - Emit IR in bytecode format
--print-debug-info  - Toggle printing debug information when emitting IR
--strip-debug-info  - Toggle stripping debug information when writing the output
```

Run `zklang --help` for more details.

# zklang-opt {#zklang-opt}

`zklang-opt` is a version of the [`mlir-opt` tool][mlir-opt-docs] that supports
passes on ZHL, ZML, and LLZK IR files. You can refer to the `mlir-opt` documentation for a general
overview of the operation of `*-opt` tooling, but note that many options and passes
available in `mlir-opt` are not available in `zklang-opt`.
`zklang-opt -h` will show a list of all available flags and options. This includes the standard `mlir-opt`
options along with the passes used for lowering to LLZK.

## Zklang Pass Documentation {#passes}

The primary addition of `zklang-opt` over `mlir-opt` is the set of passes available
that perform dialect conversion between ZHL, ZML, and LLZK.
We document the added passes below.

### Lowering Passes

\include{doc,raise=1} build/doc/mlir/passes/Passes.md

### ZHL Typing Passes

\include{doc,raise=1} build/doc/mlir/passes/ZhlTypingPasses.md

### ZML Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/ZmlTransformationPasses.md

\tableofcontents{HTML:3}


<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref setup | \ref contribution-guide |
</div>

<!-- TODO: Change this link to the github pages site -->
[llzk-site]: https://github.com/Veridise/llzk-lib
[mlir-opt-docs]: https://mlir.llvm.org/docs/Tutorials/MlirOpt/
