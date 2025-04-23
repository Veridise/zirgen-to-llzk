# What is Zklang? {#overview}

Zklang is a frontend compiler that converts Zirgen circuits into LLZK.
- [Zirgen][zirgen-repo] is a compiler for a domain-specific language, also called "zirgen",
which creates Zero Knowledge (ZK) circuits for the RISC Zero proof system.
- [LLZK][llzk-site] is an open-source Intermediate Representation (IR) for ZK
circuit languages.

## Project Overview

The Zklang repo consists of two main components:

1. **`zklang`**, the frontend compiler which translates Zirgen circuit source code into MLIR dialects.
2. **`zklang-opt`**, and `mlir-opt`-style tool which can be used to manipulate the MLIR dialects produced by `zklang`.

The general workflow of using `zklang` is as follows:
1. Translate the source language into LLZK IR using `zklang`.
2. Pass the generated LLZK IR to another tool, either for continued transformation, analysis, or conversion to downstream formats.
See the [LLZK site][llzk-site] for more information on using LLZK IR.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref mainpage | \ref setup |
</div>

<!-- TODO: Change this link to the github pages site -->
[llzk-site]: https://github.com/Veridise/llzk-lib
[zirgen-repo]: https://github.com/risc0/zirgen