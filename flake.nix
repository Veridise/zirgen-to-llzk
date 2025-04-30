# Based on the LLZK flake
{
  inputs = {
    llzk-pkgs.url = "github:Veridise/llzk-nix-pkgs?ref=main";

    nixpkgs = {
      url = "github:NixOS/nixpkgs";
      follows = "llzk-pkgs/nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils/v1.0.0";
      follows = "llzk-pkgs/flake-utils";
    };

    llzk = {
      url = "git+ssh://git@github.com/Veridise/llzk-lib.git?ref=dani/reenable-dynamic-size-arrays";
      inputs.nixpkgs.follows = "llzk-pkgs/nixpkgs";
    };
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][LLZK]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs = { self, nixpkgs, flake-utils, llzk-pkgs, llzk }:
    {
      # First, we define the packages used in this repository/flake
      overlays.default = final: prev: {

        # Default zklang build uses the default compiler for the system (usually gcc for Linux and clang for Macos)
        zklang = final.callPackage ./nix/zklang.nix { clang = final.clang_18; llzk = final.llzk; };
        # Build in release with symbols mode with a particular compiler. Mostly useful for development and CI
        zklangClang = (final.zklang.override { stdenv = final.clangStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "RelWithDebInfo";
        });
        zklangGCC = (final.zklang.override { stdenv = final.gccStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "RelWithDebInfo";
        });

        # The default shell is used for ZKLANG development.
        # Because `nix develop` is used to set up a dev shell for a given
        # derivation, we just need to extend the  derivation with any
        # extra tools we need.
        devShellBase = { pkgs, zklangEnv ? final.zklang, ... }: {
          shell = zklangEnv.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python

            ]);

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

            '';
          });
        };
      };
    } //
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          overlays = [
            self.overlays.default
            llzk-pkgs.overlays.default
            llzk.overlays.default
          ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake
        packages = flake-utils.lib.flattenTree {
          # Copy the packages from the overlay.
          inherit (pkgs) zklang;

          # For debug purposes, expose the MLIR/LLVM packages.
          inherit (pkgs) mlir llzk clang gtest python3 lit z3 cvc5;
          # Prevent use of libllvm and llvm from nixpkgs, which will have different
          # versions than the mlir from llzk-pkgs.
          inherit (pkgs.llzk_llvmPackages) libllvm llvm;

          default = pkgs.zklang;
          withClang = pkgs.zklangClang;
          withGCC = pkgs.zklangGCC;
        };

        devShells = flake-utils.lib.flattenTree {
          default =  pkgs.zklang.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python

            ]);

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

              # For using mlir-tblgen inside the dev environment
              export LD_LIBRARY_PATH=${pkgs.z3.lib}/lib:$LD_LIBRARY_PATH

              # Disable container overflow checks because it can give false positives in
              # ConvertZmlToLlzkPass::runOnOperation() since LLVM itself is not built with ASan.
              # https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
              export ASAN_OPTIONS=detect_container_overflow=0:detect_leaks=0
            '';
          });

          llvm = pkgs.mkShell {
            buildInputs = [ pkgs.libllvm.dev ];
          };
        };
      }
    ));
}
