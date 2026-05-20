import ast, pathlib
for f in sorted(pathlib.Path('D:/eeg_background_seizure/scripts').glob('*.py')):
    ast.parse(f.read_text())
    print(f'  OK: {f.name}')
