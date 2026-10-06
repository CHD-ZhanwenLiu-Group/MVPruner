import argparse
import shutil
from pathlib import Path

MODEL_FILES = [
    '__init__.py',
    'configuration_intern_vit.py',
    'configuration_internlm2.py',
    'configuration_internvl_chat.py',
    'conversation.py',
    'modeling_intern_vit.py',
    'modeling_internvl_chat.py',
    'modeling_internlm2_mvpruner.py',
    'tokenization_internlm2.py',
]


def main():
    parser = argparse.ArgumentParser(description='Install MVPruner remote-code files into a local DriveLMM-o1 checkpoint directory.')
    parser.add_argument('--model-dir', required=True, help='Local DriveLMM-o1 checkpoint directory, e.g. ckpt/DriveLMMo1-MVPruner')
    parser.add_argument('--code-dir', default='modeling', help='Directory containing MVPruner model code files')
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    code_dir = Path(args.code_dir).expanduser().resolve()
    if not (model_dir / 'config.json').exists():
        raise FileNotFoundError(f'{model_dir}/config.json not found. Download DriveLMM-o1 checkpoint first.')
    if not code_dir.exists():
        raise FileNotFoundError(f'Model code directory not found: {code_dir}')

    for name in MODEL_FILES:
        src = code_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, model_dir / name)

    print(f'MVPruner model code installed into: {model_dir}')


if __name__ == '__main__':
    main()
