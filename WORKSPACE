workspace(name = "com_veridise_zir-to-zkir")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ZIRGEN_COMMIT = "91b3abdb1778f0089f5f07119c5d57b5477d8bcf"
ZIRGEN_SHA256 = "f7bc4f3dd192247905c7be86bae64fa66b4fcb7812e0a247a79aad46a69c5693"

http_archive(
  name = "zirgen",
  # build_file_content = "# empty",
  sha256 = ZIRGEN_SHA256,
  strip_prefix = "zirgen-" + ZIRGEN_COMMIT,
  urls = ["https://github.com/risc0/zirgen/archive/{commit}.tar.gz".format(commit = ZIRGEN_COMMIT)]
)

