{ 
  stdenv, pkgs
}:
stdenv.mkDerivation {
  name = "zirgen";
  src = pkgs.fetchgit {
    url = ../third-party/zirgen;
    sha256 = "";
  };
}

