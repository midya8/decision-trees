"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['treelab', 'treelab.__main__', 'treelab.tree', 'treelab.data',
           'treelab.experiment', 'treelab.bagging', 'treelab.sweep',
           'treelab.counterfactual', 'treelab.forest_search']
COMMANDS = [
    ['demo'], ['audit', '--output', 'refit'],
    ['predict', '--model', 'treelab/artifacts/selected-model.json', '--input', 'examples/input.json'],
    ['fit', '--input', 'examples/train.json', '--output', 'fit.json', '--min-leaf', '1'],
    ['prune', '--model', 'fit.json', '--alpha', '1', '--output', 'pruned.json'],
    ['bag-demo'], ['bag-audit', '--output', 'bag-refit'],
    ['bag-fit', '--input', 'examples/train.json', '--output', 'ensemble.json', '--trees', '4', '--min-leaf', '1'],
    ['bag-predict', '--model', 'treelab/artifacts/bagging-model.json.gz', '--input', 'examples/input.json'],
    ['bag-predict', '--model', 'ensemble.json', '--input', 'fit-input.json'],
    ['bag-oob', '--model', 'ensemble.json', '--input', 'examples/train.json'],
    ['sweep-demo'], ['sweep-audit', '--output', 'sweep-refit'],
    ['sweep', '--model', 'treelab/artifacts/bagging-model.json.gz', '--input', 'examples/sweep.json'],
    ['counterfactual-demo'], ['counterfactual-audit', '--output', 'cf-refit'],
    ['counterfactual', '--model', 'treelab/artifacts/selected-model.json', '--input', 'examples/counterfactual.json'],
    ['forest-demo'], ['forest-audit', '--output', 'forest-refit'],
    ['forest-search', '--model', 'treelab/artifacts/bagging-model.json.gz', '--input', 'examples/forest-search.json'],
]
FILES = ['fit.json', 'pruned.json', 'ensemble.json', 'refit/report.json',
         'refit/grown-model.json', 'refit/selected-model.json',
         'bag-refit/bagging-model.json.gz', 'bag-refit/bagging-report.json',
         'sweep-refit/sweep-report.json', 'cf-refit/counterfactual-report.json',
         'forest-refit/forest-search-report.json']


def load(path):
    import gzip
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'docs', 'assets']:
        shutil.copytree(ROOT / name, outside / name)
    for name in ['artifacts', 'data']:
        shutil.copytree(ROOT / 'treelab' / name, outside / 'treelab' / name)
    assert not list((outside / 'treelab').rglob('*.py'))
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        reference.mkdir()
        for name in ['examples', 'treelab/artifacts']:
            shutil.copytree(ROOT / name, reference / name)
        fixture = json.loads((ROOT / 'examples/train.json').read_text())
        for target in [reference, outside]:
            (target / 'fit-input.json').write_text(json.dumps({'features': fixture['features']}))
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)
        expected = []
        for command in COMMANDS:
            output = call([sys.executable, '-m', 'treelab'] + command, reference, source_env).strip()
            expected.append(output)
        files = {name: load(reference / name) for name in FILES}
        # This package has no console-script entrypoint. Check its module CLI.
        for command, report in zip(COMMANDS, expected):
            output = call([py, '-I', '-m', 'treelab'] + command, outside).strip()
            assert output == report, command
        for name, value in files.items():
            assert load(outside / name) == value, name
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_invocations=len(COMMANDS), command_kinds=19,
                              complete_output_files=len(FILES), entrypoints=1)))
