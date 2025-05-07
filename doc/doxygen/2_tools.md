# Tool Guides {#tools}

# zklang {#zklang}

`zklang` is the primary tool provided by this project.
`zklang` is the frontend compiler that converts Zirgen source code into different
MLIR dialects. The end-to-end workflow of `zklang` is to perform the following translations:
1. zirgen source code to zirgen AST
2. zirgen AST to ZHL, an MLIR dialect designed by Risc0 to represent zirgen programs (see [the ZHL dialect documentation](build/doc/doxygen/mlir/ZHLDialect.md)).
3. ZHL to ZML, an MLIR dialect designed by Veridise as an intermediary dialect between ZHL and LLZK (see [the ZML dialect documentation](build/doc/doxygen/mlir/ZMLDialect.md)).
4. ZML to LLZK, Veridise's ZK circuit IR (see [the LLZK project site][llzk-site] for more information about the LLZK dialect).

Generally, users will use `zklang` to translate zirgen circuits to LLZK bytecode (which is more efficient for downstream tooling to operate on),
but `zklang` can optionally emit any of the above intermediate representations for inspection or debugging purproses.


## Usage 

```
zklang [options] <input zirgen file>
```

## General options

```
--help             - Display available options
--version          - Display the version of this program
-I <path>          - Add include path
-o <output>        - Where to write the result
--emit-bytecode    - Emit IR in bytecode format
--strip-debug-info - Toggle stripping debug information when writing the output
```

## Emission options 

The `--emit=<value>` flag is used to control the output `zklang` produces. Only one kind of output can be selected among the following options. If none is selected the tool by default produces LLZK IR.

```
--emit=ast    -   Output the AST
--emit=zhl    -   Output untyped high level ZIR IR
--emit=zml    -   Output typed medium level ZIR IR
--emit=zmlopt -   Output typed medium level ZIR IR with separate compute and constrain functions
--emit=llzk   -   Output LLZK IR (default)
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

\include{doc,raise=1} build/doc/doxygen/mlir/passes/Passes.md

### ZHL Typing Passes

\include{doc,raise=1} build/doc/doxygen/mlir/passes/ZhlTypingPasses.md

### ZML Transformation Passes

\include{doc,raise=1} build/doc/doxygen/mlir/passes/ZmlTransformationPasses.md

\tableofcontents{HTML:3}


<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref setup | \ref contribution-guide |
</div>

[llzk-site]: https://veridise.github.io/llzk-lib/
[mlir-opt-docs]: https://mlir.llvm.org/docs/Tutorials/MlirOpt/
