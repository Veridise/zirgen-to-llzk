import argparse
import os
import sys
from subprocess import PIPE, Popen

WS = os.path.dirname(__file__)
ZIRGEN = os.path.join(WS, "../external/zirgen/zirgen/dsl/zirgen")
ZKC_OPT = os.path.join(WS, "../ZirToZkir/tools/zkc-opt")

ZKC_OPT_PIPELINE = """
builtin.module(
  inject-builtins
)
"""


def main():
    parser = argparse.ArgumentParser(
        prog="zklang",
        description="A simple script to translate from source ZIR to ZKIR",
    )

    print(os.getcwd())
    parser.add_argument("filename")
    args = parser.parse_args()

    with Popen([ZIRGEN, "--emit=zhl", args.filename], stdout=PIPE) as zirgen:
        with Popen(
            [ZKC_OPT, f"--pass-pipeline={ZKC_OPT_PIPELINE}"], stdin=zirgen.stdout
        ) as _:
            pass


if __name__ == "__main__":
    main()
