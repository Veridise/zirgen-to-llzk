# zir-to-zkir

MLIR modules for transforming ZIR to ZKIR

## Build instructions

Install [bazel](https://bazel.build/install) 7.3.2 (you can check with `bazel version`) and then run the following command to build everything.

```
bazel build //...
```

**The first time you do a build it will build LLVM from source**, which
can be pretty intense. If the build machine is not very powerful
adjust the alloted resources for the build. For example, to use 
only half of the CPUs use the command below. That is what I do in 
a ThinkPad Intel i5-8265U with 16GB of ram and it takes forever but 
completes. On something like a M2 mac it should be faster.

```
bazel build --local_resources=cpu='HOST_CPUS*0.5' //...
```

To build a specific component pass the path of the module
where it is defined and the name of the component. For example to 
build the `zkc-opt` tool run the following command.

```
bazel build //ZirToZkir/tools:zkc-opt
```

## Usage 

The project has two tools; the frontend, `zklang`, and the optimizer, `zkc-opt`. The frontend is a Python 
script that generates IR from `zirgen` and sends it to `zkc-opt` with a preconfigured lowering pipeline to LLZK.

The general syntax is as follows:

```
bazel run //zklang:zklang -- <frontend arguments>
bazel run //ZirToZkir/tools:zkc-opt -- <optimizer arguments>
```

Bazel will recompile the project before running if it detects that a dependency has changed. **You can pass the 
resource limiting flags showed above here as well, should you need them.** Since it's running 
the tool inside the Bazel sandbox, as a precaution, pass any path as an absolute path. Relative paths will not
use the directory you are running from which makes them more annoying to work with.

To compile a file from source use the following command.

```
bazel run //zklang:zklang -- /absolute/path/to/the/source/file.zir
```

The frontend at this stage is intended as a e2e testing tool so it doesn't have many features beyond compiling the source code and showing the 
final IR after the last pass defined in the pipeline. You can do `bazel run //zklang:zklang -- --help` to get a list of options. 
To get a list of options for the optimizer you can do `bazel run //ZirToZkir/tools:zkc-opt -- --help`.
