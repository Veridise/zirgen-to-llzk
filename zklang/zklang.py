import argparse
import os
import shlex
import sys
from subprocess import PIPE, Popen
from typing import Optional

WS = os.path.dirname(__file__)
ZIRGEN = os.path.join(WS, "../external/zirgen/zirgen/dsl/zirgen")
ZKC_OPT = os.path.join(WS, "../ZirToZkir/tools/zkc-opt")


class Pipeline: ...


class Scope(Pipeline):
    def __init__(self, name, *passes):
        self.name = name
        self.passes = passes

    def __str__(self) -> str:
        return f"{self.name}({','.join(str(p) for p in self.passes)})"


class Pass(Pipeline):
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
    Scope(
        "zmir.component",
        Pass("zhl-to-zmir"),
        Pass("insert-temporaries"),
        Scope("func.func", Pass("lower-builtins")),
        Pass("legalize-types"),
        Pass("super-coercion"),
    ),
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
    Pass("canonicalize"),
)


class ZkcOpt:
    pipeline: Optional[Pipeline]
    debug_only: Optional[str]

    def __init__(self) -> None:
        self.pipeline = None
        self.dump_ir_after_failure = False
        self.dump_ir_as_tree = False
        self.disable_threading = False
        self.debug_only = None
        self.additional_args = None
        self.output_filename = None

    def set_pipeline(self, pipeline: Pipeline):
        self.pipeline = pipeline
        return self

    def dump_ir(self, dump, as_tree=False):
        self.dump_ir_after_failure = dump
        self.dump_ir_as_tree = as_tree
        return self

    def dump_as_tree(self, val):
        self.dump_ir_as_tree = val
        return self

    def set_disable_threading(self, val: bool):
        self.disable_threading = val
        return self

    def set_debug_only(self, debug_type: str):
        self.debug_only = debug_type
        return self

    def debug_dialect_conversion(self):
        return self.set_debug_only("dialect-conversion")

    def set_additional_args(self, args):
        self.additional_args = args
        return self

    def set_output(self, filename):
        self.output_filename = filename
        return self

    def run(self, input):
        assert self.pipeline is not None
        cmd = [ZKC_OPT, f"--pass-pipeline={self.pipeline}"]
        if self.dump_ir_after_failure:
            cmd.append("--mlir-print-ir-after-failure")
            if self.dump_ir_as_tree:
                cmd.append("--mlir-print-ir-tree-dir")
        if self.debug_only is not None:
            cmd.append(f"--debug-only={self.debug_only}")
        if self.disable_threading:
            cmd.append("--mlir-disable-threading")
        if self.output_filename is not None:
            cmd.append("-o")
            cmd.append(self.output_filename)
        if self.additional_args:
            extra_args = shlex.split(self.additional_args)
            cmd += extra_args
        return Popen(cmd, stdin=input)


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def color_msg(color, *args, **kwargs):
    in_tty = (kwargs["file"] if "file" in kwargs else sys.stdout).isatty()
    msg = []
    if in_tty:
        msg.append(color)
    msg += args
    if in_tty:
        msg.append(RESET)
    print(*msg, **kwargs)


def success(*args, **kwargs):
    color_msg(GREEN, *args, **kwargs)


def failure(*args, **kwargs):
    color_msg(RED, *args, **kwargs)


def print_zirgen_failure(zirgen):
    print_kwargs = {"file": sys.stderr}
    failure(f"zirgen failed ({zirgen.returncode}):", **print_kwargs)
    print(zirgen.stderr.read().decode("utf8"), **print_kwargs)

    print("=========== Stdout (last 10 lines) ===========", **print_kwargs)
    print(
        "\n".join(zirgen.stdout.read().decode("utf8").split("\n")[-10:]), **print_kwargs
    )


def main():
    parser = argparse.ArgumentParser(
        prog="zklang",
        description="A simple script to translate from source ZIR to ZKIR",
    )

    parser.add_argument("filename")
    parser.add_argument("-o", "--output")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--dump-as-tree", action="store_true")
    parser.add_argument("--enable-threading", action="store_true")
    parser.add_argument("--debug-dialect-conversion", action="store_true")
    parser.add_argument("--zirgen-args")
    parser.add_argument("--zkc-opt-args")
    parser.add_argument("--zkc-opt-help", action="store_true")
    args = parser.parse_args()
    basedir = os.path.dirname(args.filename)

    if args.zkc_opt_help:
        with Popen([ZKC_OPT, "--help"]) as opt:
            opt.wait()
            return

    extra_zirgen_args = shlex.split(args.zirgen_args) if args.zirgen_args else []

    with Popen(
        [ZIRGEN, "--emit=zhl", "-I", basedir, *extra_zirgen_args, args.filename],
        stdout=PIPE,
        stderr=PIPE,
    ) as zirgen:
        zirgen.wait()
        if zirgen.returncode != 0:
            print_zirgen_failure(zirgen)
            return
        tool = (
            ZkcOpt()
            .set_pipeline(ZKC_OPT_PIPELINE)
            .dump_ir(args.dump)
            .dump_as_tree(args.dump_as_tree)
            .set_disable_threading(not args.enable_threading)
            .set_additional_args(args.zkc_opt_args)
        )
        if args.debug_dialect_conversion:
            tool.debug_dialect_conversion()
        if args.output:
            tool.set_output(args.output)
        with tool.run(zirgen.stdout) as opt:
            opt.wait()
            if opt.returncode != 0:
                failure("Something went wrong with zkc-opt", file=sys.stderr)
            else:
                success("Success!", file=sys.stderr)
                if args.output:
                    print(f"Result writen to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
