# Tool Guides {#tools}

\tableofcontents

# zklang {#zklang}

`zklang` is the primary tool provided by this project.
`zklang` is the frontend compiler that converts


# zklang-opt {#zklang-opt}

`llzk-opt` is a version of the [`mlir-opt` tool](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) that supports
passes on LLZK IR files. You can refer to the `mlir-opt` documentation for a general
overview of the operation of `*-opt` tooling, but note that many options and passes
available in `mlir-opt` are not available in `llzk-opt`.
`llzk-opt -h` will show a list of all available flags and options.

##### LLZK-Specific Options
```
-I <directory> : Directory of include files
```

## Zklang Pass Documentation {#passes}

### Lowering Passes

\include{doc,raise=1} build/doc/mlir/passes/Passes.md

### ZHL Typing Passes

\include{doc,raise=1} build/doc/mlir/passes/ZhlTypingPasses.md

### ZML Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/ZmlTransformationPasses.md


<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref setup | \ref contribution-guide |
</div>

<!-- links -->