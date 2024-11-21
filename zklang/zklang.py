import argparse
import os
from subprocess import DEVNULL, PIPE, Popen

WS = os.path.dirname(__file__)
ZIRGEN = os.path.join(WS, "../external/zirgen/zirgen/dsl/zirgen")
ZKC_OPT = os.path.join(WS, "../ZirToZkir/tools/zkc-opt")


class Scope:
    def __init__(self, name, passes):
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
    [
        Pass("strip-tests"),
        Pass("strip-directives"),
        Pass("inject-builtins"),
        Pass("transform-component-decls"),
        Scope("zmir.component", [Pass("convert-zhl-to-zmir")]),
        Pass("cse"),
    ],
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
        ) as _:
            pass


if __name__ == "__main__":
    main()
