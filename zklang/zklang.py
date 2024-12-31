import argparse
import os
import shlex
import sys
from subprocess import DEVNULL, PIPE, Popen
from typing import Optional, Protocol

WS = os.path.dirname(__file__)
ZIRGEN = os.path.join(WS, "../external/zirgen/zirgen/dsl/zirgen")
ZKC_OPT = os.path.join(WS, "../ZirToZkir/tools/zkc-opt")


class Pipeline(Protocol):
    def stop_at(self, name: str): ...

    def is_stop_point(self) -> bool: ...


class Scope(Pipeline):
    def __init__(self, name: str, *passes):
        self.name = name
        self.all_passes = passes
        self.stop_point = None

    @property
    def passes(self):
        if self.stop_point is None:
            yield from self.all_passes
        for p in self.all_passes:
            yield p
            if p.is_stop_point():
                break

    def __str__(self) -> str:
        return f"{self.name}({','.join(str(p) for p in self.passes)})"

    def stop_at(self, name: str):
        self.stop_point = name
        for p in self.all_passes:
            p.stop_at(name)

    def is_stop_point(self) -> bool:
        return any(p.is_stop_point() for p in self.all_passes)


class Pass(Pipeline):
    def __init__(self, name: str):
        self.name = name
        self.stop_point = None

    def __str__(self) -> str:
        return self.name

    def stop_at(self, name: str):
        self.stop_point = name

    def is_stop_point(self) -> bool:
        return self.name == self.stop_point


ZKC_OPT_PIPELINE = Scope(
    "builtin.module",
    Pass("strip-tests"),
    Pass("strip-directives"),
    Pass("inject-builtins"),
    Pass("lower-zhl"),
    Scope(
        "zmir.component",
        Pass("insert-temporaries"),
        Scope("func.func", Pass("lower-builtins")),
        Pass("legalize-types"),
        # Pass("super-coercion"),
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
        self.dump_pipeline = False

    def set_pipeline(self, pipeline: Pipeline):
        self.pipeline = pipeline
        return self

    def set_dump_pipeline(self, dump: bool):
        self.dump_pipeline = dump
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
        if self.dump_pipeline:
            cmd.append("--dump-pass-pipeline")
            return Popen(cmd, stdin=DEVNULL)
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
OK = "\033[32m[+]\033[0m"


def in_tty(kwargs):
    return (kwargs["file"] if "file" in kwargs else sys.stdout).isatty()


def color_msg(color, *args, **kwargs):
    msg = []
    if in_tty(kwargs):
        msg.append(color)
    msg += args
    if in_tty(kwargs):
        msg.append(RESET)
    print(*msg, **kwargs)


def success(*args, **kwargs):
    color_msg(GREEN, *args, **kwargs)


def failure(*args, **kwargs):
    color_msg(RED, *args, **kwargs)


def ok(*args, **kwargs):
    if in_tty(kwargs):
        hdr = OK
    else:
        hdr = "[+]"
    print(hdr, *args, **kwargs)


def dbg(*args, **kwargs):
    print("[D]", *args, **kwargs)


def print_zirgen_failure(zirgen):
    print_kwargs = {"file": sys.stderr}
    failure(f"zirgen failed with error code {zirgen.returncode}:", **print_kwargs)
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
    parser.add_argument("--dump-pipeline", action="store_true")
    parser.add_argument("--dump-as-tree", action="store_true")
    parser.add_argument("--enable-threading", action="store_true")
    parser.add_argument("--debug-dialect-conversion", action="store_true")
    parser.add_argument("--zirgen-args")
    parser.add_argument("--zkc-opt-args")
    parser.add_argument("--zkc-opt-help", action="store_true")
    parser.add_argument("--stop-at")
    parser.add_argument("--pipeline")
    args = parser.parse_args()
    basedir = os.path.dirname(args.filename)

    if args.zkc_opt_help:
        with Popen([ZKC_OPT, "--help"]) as opt:
            opt.wait()
            return

    if args.stop_at:
        ZKC_OPT_PIPELINE.stop_at(args.stop_at)

    selected_pipeline = ZKC_OPT_PIPELINE
    if args.pipeline:
        selected_pipeline = args.pipeline

    extra_zirgen_args = shlex.split(args.zirgen_args) if args.zirgen_args else []

    ok(f"Generating ZHL with zirgen for {args.filename}...", file=sys.stderr)
    zirgen_cmd = [
        ZIRGEN,
        "--emit=zhl",
        "-I",
        basedir,
        *extra_zirgen_args,
        args.filename,
    ]
    dbg("zirgen command:", *zirgen_cmd, file=sys.stderr)
    with Popen(
        zirgen_cmd,
        stdout=PIPE,
        stderr=PIPE,
    ) as zirgen:
        dbg("Waiting for zirgen to finish...", file=sys.stderr)
        zirgen.poll()
        if zirgen.returncode is not None and zirgen.returncode != 0:
            print_zirgen_failure(zirgen)
            sys.exit(zirgen.returncode)
        tool = (
            ZkcOpt()
            .set_pipeline(selected_pipeline)
            .dump_ir(args.dump)
            .set_dump_pipeline(args.dump_pipeline)
            .dump_as_tree(args.dump_as_tree)
            .set_disable_threading(not args.enable_threading)
            .set_additional_args(args.zkc_opt_args)
        )
        if args.debug_dialect_conversion:
            tool.debug_dialect_conversion()
        if args.output:
            tool.set_output(args.output)
        ok(f"Generating LLZK with zkc-opt for {args.filename}...", file=sys.stderr)
        with tool.run(zirgen.stdout) as opt:
            opt.wait()
            if opt.returncode != 0:
                failure("Something went wrong with zkc-opt", file=sys.stderr)
                sys.exit(opt.returncode)
            else:
                success("Success!", file=sys.stderr)
                if args.output:
                    print(f"Result writen to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
