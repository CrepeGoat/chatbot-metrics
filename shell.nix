{
  pkgs ?
    import
      (fetchTarball "https://github.com/NixOS/nixpkgs/archive/11cb3517b3af6af300dd6c055aeda73c9bf52c48.tar.gz")
      { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    # python312Packages.matplotlib
    # python312Packages.numpy
    # python312Packages.scikitlearn

    # python312Packages.jupyterlab
    # python312Packages.ibm-watson
    # python312Packages.python-dotenv
  ];
}
