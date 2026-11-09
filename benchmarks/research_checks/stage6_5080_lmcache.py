"""Rebuild the exact Stage 6 LMCache release for Blackwell; no policy edits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

REVISION = 'cad4db10592e8487d1e0bbd02d1a6f92c37d0581'
REDIST = 'https://developer.download.nvidia.com/compute/cuda/redist/'
COMPONENTS = {
    'cuda_nvcc': ('12.8.93', '9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3'),
    'cuda_cudart': ('12.8.90', '8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68'),
    'cuda_cccl': ('12.8.90', '0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(source, python, toolkit, output, original_python):
    source, python, toolkit, output = map(Path, (source, python, toolkit, output))
    if output.exists():
        raise ValueError('Use a new receipt directory; preserve failed builds')
    output.mkdir(parents=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']):
        raise ValueError('LMCache must be the unchanged pinned release checkout')
    # Prove that the installed release really corresponds to this source pin.
    probe = '''import json,lmcache._version as v,torch,importlib.metadata as m
print(json.dumps(dict(version=v.__version__,commit=v.__commit_id__,torch=torch.__version__,cuda=torch.version.cuda,abi=torch._C._GLIBCXX_USE_CXX11_ABI,site=str(m.distribution('lmcache').locate_file('lmcache')))))'''
    original = json.loads(subprocess.check_output([str(original_python), '-c', probe], text=True))
    if original['version'] != '0.3.6' or original['commit'] != 'g' + REVISION[:9]:
        raise ValueError('Original wheel does not attest the requested source revision')
    if original['torch'] != '2.8.0+cu128' or original['cuda'] != '12.8':
        raise ValueError('Unexpected pinned torch/CUDA identity')
    original_site = Path(original['site'])
    compared = []
    for path in (source / 'lmcache').rglob('*.py'):
        relative = path.relative_to(source / 'lmcache')
        if str(relative) == '_version.py':
            continue
        if sha(path) != sha(original_site / relative):
            raise ValueError('Release Python differs from original wheel: ' + str(relative))
        compared.append(str(relative))
    toolkit.mkdir(parents=True, exist_ok=True)
    archives = toolkit / 'downloads'
    archives.mkdir(exist_ok=True)
    for name, (version, expected) in COMPONENTS.items():
        filename = f'{name}-linux-x86_64-{version}-archive.tar.xz'
        archive = archives / filename
        if not archive.exists():
            urllib.request.urlretrieve(REDIST + f'{name}/linux-x86_64/' + filename, archive)
        if sha(archive) != expected:
            raise ValueError('NVIDIA archive hash mismatch: ' + name)
        with tarfile.open(archive) as tar:
            tar.extractall(toolkit / 'components', filter='data')
        root = toolkit / 'components' / filename.removesuffix('.tar.xz')
        for child in root.iterdir():
            destination = toolkit / child.name
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
    # PyTorch's linker expects lib64, while redistributables use lib.
    if not (toolkit / 'lib64').exists():
        (toolkit / 'lib64').symlink_to('lib', target_is_directory=True)
    env = dict(os.environ, CUDA_HOME=str(toolkit), TORCH_CUDA_ARCH_LIST='12.0+PTX',
               ENABLE_CXX11_ABI='1' if original['abi'] else '0', MAX_JOBS='2',
               CC='/usr/bin/gcc-14', CXX='/usr/bin/g++-14',
               PATH=str(toolkit / 'bin') + ':' + os.environ['PATH'],
               TMPDIR=str(output), PYTHONDONTWRITEBYTECODE='1')
    command = [str(python), 'setup.py', 'bdist_wheel', '--dist-dir', str(output / 'wheels')]
    recipe = dict(schema='stage6-5080-lmcache-build-v1', source_revision=revision,
                  source_repository='https://github.com/LMCache/LMCache', original=original,
                  original_extension_sha256=sha(next(original_site.glob('c_ops*.so'))),
                  unchanged_release_python_files=len(compared), architectures='12.0+PTX',
                  toolkit_components=COMPONENTS, command=command,
                  host_compiler=subprocess.check_output(['/usr/bin/g++-14', '--version'], text=True),
                  compiler=subprocess.check_output([str(toolkit / 'bin/nvcc'), '--version'], text=True),
                  declared_deviation='Same LMCache source and policy, rebuilt only for RTX 5080 sm_120 and compute_120 PTX; separate environment preserves the original wheel')
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    with (output / 'build.log').open('xb') as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = (output / 'wheels').glob('*.whl')
    recipe.update(wheel=str(wheel), wheel_sha256=sha(wheel))
    (output / 'build.json').write_text(json.dumps(recipe, indent=2) + '\n')
    print(json.dumps(dict(wheel=str(wheel), sha256=sha(wheel), release_python_files=len(compared))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'python', 'toolkit', 'out', 'original-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.source.resolve(), args.python.absolute(), args.toolkit.resolve(),
          args.out.resolve(), args.original_python.absolute())
