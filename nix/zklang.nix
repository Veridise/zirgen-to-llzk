{
  stdenv, lib,

  # build dependencies
  clang, cmake, ninja,
  mlir, nlohmann_json,
  llzk, #zirgen,

  # test dependencies
  gtest, python3, lit, z3, cvc5
}:

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
          # (type == "directory" && path == toString (src0.origSrc + "/third-party"))
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

  # Needed for mlir-tblgen to run properly.
  preBuild = ''
    export LD_LIBRARY_PATH=${z3.lib}/lib:$LD_LIBRARY_PATH
  '';

  # cmakeFlags = [
  #   "-DLLZK_BUILD_DEVTOOLS=ON"
  # ];

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
