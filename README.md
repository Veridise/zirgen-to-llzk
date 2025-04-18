# zirgen-to-llzk

MLIR modules for transforming Zirgen to LLZK

## Build instructions

::warning:: This section is out-of-date and will be updated in LLZK-228.

Install [bazel](https://bazel.build/install) and then run the following command to build everything.

```
bazel build //...
```

The first time you do a build it will build LLVM from source, which
can be pretty intense. If the build machine is not very powerful
adjust the alloted resources for the build. For example, to use
only half of the CPUs use the following command.

```
bazel build --local_resources=cpu='HOST_CPUS*0.5' //...
```

To build a specific component pass the path of the module
where it is defined and the name of the component. For example to
build the `zkc-opt` tool run the following command.

```
bazel build //ZirToZkir/tools:zkc-opt
```
