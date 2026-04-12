from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all non-Python data files (JSON configs, tokenizer files, etc.)
datas = collect_data_files("transformers")

# Collect all submodules so PyInstaller's static analysis finds them
hiddenimports = collect_submodules("transformers")
