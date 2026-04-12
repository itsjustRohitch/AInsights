from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas          = collect_data_files("chromadb")
hiddenimports  = collect_submodules("chromadb")
hiddenimports += collect_submodules("onnxruntime")