{
  pkgs,
  stdenv, lib,

  # build dependencies
  clang, cmake, ninja,
  mlir, nlohmann_json,
  llzk, #zirgen,

  # test dependencies
  gtest, python3, lit, z3, cvc5
}:
let
  zirgen-src = pkgs.fetchgit {
    url = "https://github.com/Veridise/zirgen";
    rev = "8e4eaa65a45f4b7c38bd672729e6e1a3ba9ae2a7";
    sha256 = "vlet1olaBdofpigwBjPeIOc2gn2a7inlxcJsl4tc/ss=";
  };
in
  stdenv.mkDerivation {
    name = "zklang";
    version = "0.1.0";
    src =
      let
        src0 = lib.cleanSource (builtins.path {
          path = ./..;
          name = "zklang-source";
        });
      in
        lib.cleanSourceWith {
          # Ignore unnecessary files
          filter = path: type: !(lib.lists.any (x: x) [
            (path == toString (src0.origSrc + "/README.md"))
            (type == "directory" && path == toString (src0.origSrc + "/.github"))
            (type == "regular" && lib.strings.hasSuffix ".nix" (toString path))
            (type == "regular" && baseNameOf path == "flake.lock")
          ]);
          src = src0;
        };

    nativeBuildInputs = [ clang cmake ninja z3.lib z3 ];
    buildInputs =  [
      mlir llzk z3.lib
    ];

    cmakeFlags = [
      "-DZIRGEN_SRC=${zirgen-src}"
    ];

    # Needed for mlir-tblgen to run properly.
    preBuild = ''
      export LD_LIBRARY_PATH=${z3.lib}/lib:$LD_LIBRARY_PATH
    '';

    # This is done specifically so that the configure phase can find /usr/bin/sw_vers,
    # which is MacOS specific.
    # Note that it's important for "/usr/bin/" to be last in the list so we don't
    # accidentally use the system clang, etc.
    preConfigure = ''
      if [[ "$(uname)" == "Darwin" ]]; then
        export OLD_PATH=$PATH
        export PATH="$PATH:/usr/bin/"
      fi
    '';

    # this undoes the above configuration, as it will cause problems later.
    postConfigure = ''
      if [[ "$(uname)" == "Darwin" ]]; then
        export PATH=$OLD_PATH
        # unset OLD_PATH
      fi
    '';

    doCheck = true;
    checkTarget = "check";
    checkInputs = [ clang gtest python3 lit z3 cvc5 ];
  }
