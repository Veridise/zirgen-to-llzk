import argparse
import os
from subprocess import DEVNULL, PIPE, Popen

WS = os.path.dirname(__file__)
ZIRGEN = os.path.join(WS, "../external/zirgen/zirgen/dsl/zirgen")
ZKC_OPT = os.path.join(WS, "../ZirToZkir/tools/zkc-opt")


class Scope:
    def __init__(self, name, *passes):
        self.name = name
        self.passes = passes

    def __str__(self) -> str:
        return f"{self.name}({','.join(str(p) for p in self.passes)})"


class Pass:
    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return self.name


ZKC_OPT_PIPELINE = Scope(
    "builtin.module",
    Pass("strip-tests"),
    Pass("strip-directives"),
    Pass("inject-builtins"),
    Pass("transform-component-decls"),
    Scope("zmir.component", Pass("zhl-to-zmir")),
    Pass("cse"),
    Pass("canonicalize"),
    Scope("zmir.component", Scope("func.func", Pass("lower-builtins"))),
    Pass("remove-builtins"),
    Pass("split-component-body"),
    Scope(
        "zmir.split_component",
        Scope(
            "func.func",
            Pass("remove-illegal-compute-ops"),
            Pass("remove-illegal-constrain-ops"),
        ),
    ),
    Pass("cse"),
    Pass("canonicalize"),
    Pass("zmir-components-to-zkir"),
    Scope("zkir.struct", Pass("zmir-to-zkir")),
    # Pass("reconcile-unrealized-casts"),
)


def main():
    parser = argparse.ArgumentParser(
        prog="zklang",
        description="A simple script to translate from source ZIR to ZKIR",
    )

    print(os.getcwd())
    parser.add_argument("filename")
    args = parser.parse_args()
    basedir = os.path.dirname(args.filename)

    with Popen(
        [ZIRGEN, "--emit=zhl", "-I", basedir, args.filename], stdout=PIPE, stderr=PIPE
    ) as zirgen:
        zirgen.wait()
        if zirgen.returncode != 0:
            print(
                f"zirgen failed ({zirgen.returncode}):\n",
                zirgen.stderr.read().decode("utf8"),
            )

            print("=========== Stdout (last 10 lines) ===========")
            print("\n".join(zirgen.stdout.read().decode("utf8").split("\n")[-10:]))
            return
        with Popen(
            [
                ZKC_OPT,
                f"--pass-pipeline={ZKC_OPT_PIPELINE}",
                "-debug-only=dialect-conversion",
                "--mlir-print-ir-after-failure",
                "--mlir-disable-threading",
            ],
            stdin=zirgen.stdout,
            # stdout=DEVNULL,
        ) as opt:
            opt.wait()
            if opt.returncode != 0:
                print("Something went wrong with zkc-opt")
            else:
                print("Success!")


if __name__ == "__main__":
    main()
