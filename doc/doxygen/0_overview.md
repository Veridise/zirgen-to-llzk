# Repository Organization {#overview}

The Zklang repo consists of two main components:

1. **`zklang`**, the frontend compiler which translates Zirgen circuit source code into MLIR dialects.
2. **`zklang-opt`**, an `mlir-opt`-style tool which can be used to manipulate the MLIR dialects produced by `zklang`.

The general workflow of using `zklang` is as follows:
1. Translate the source language into LLZK IR using `zklang`.
2. Pass the generated LLZK IR to another tool, either for continued transformation, analysis, or conversion to downstream formats.
See the [LLZK site][llzk-site] for more information on using LLZK IR.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref mainpage | \ref setup |
</div>

[llzk-site]: https://veridise.github.io/llzk-lib/
[zirgen-repo]: https://github.com/risc0/zirgen
