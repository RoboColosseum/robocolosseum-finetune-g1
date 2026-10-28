"""Small, idempotent inference portability patches for the pinned upstream checkout.

Keep training exports lazy, and use PyTorch attention when FlashAttention is
unavailable. No parameter names, tensor shapes or checkpoint files are changed.
"""
import ast
from pathlib import Path
import sys


def patch(root):
    for relative in ('lingbotvla/data/__init__.py', 'lingbotvla/data/vla_data/__init__.py'):
        path = root / relative
        source = path.read_text()
        if '# G01 lazy training exports' in source:
            if '__all__' not in source:
                path.write_text(source + '\n__all__ = list(_exports)\n')
            continue
        exports = {}
        for node in ast.parse(source).body:
            if not isinstance(node, ast.ImportFrom):
                raise ValueError(f'Unexpected upstream initializer: {relative}')
            for name in node.names:
                exports[name.asname or name.name] = ('.' * node.level + node.module, name.name)
        path.write_text('# G01 lazy training exports\nfrom importlib import import_module\n'
                        f'_exports = {exports!r}\n'
                        '__all__ = list(_exports)\n'
                        'def __getattr__(name):\n'
                        '    if name not in _exports: raise AttributeError(name)\n'
                        '    module, symbol = _exports[name]\n'
                        '    value = getattr(import_module(module, __name__), symbol)\n'
                        '    globals()[name] = value\n'
                        '    return value\n')
    path = root / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
    source = path.read_text()
    if '# G01 portable attention initialization' not in source:
        if source.count('= "flash_attention_2"') != 3:
            raise ValueError('Upstream attention initialization changed; review patch')
        source = source.replace('= "flash_attention_2"', '= "eager"')
        path.write_text('# G01 portable attention initialization\n' + source)


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
