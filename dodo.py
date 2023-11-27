# -*- coding: utf-8 -*-
# pydoit tast file
# see https://pydoit.org/
# run from this dir with `doit`, or `doit list`, `doit help` etc. (pip install doit 1st)

import pathlib
import pygraphviz
from import_deps import PyModule, ModuleSet


def task_prospector():
    """Run prospector static analysis"""
    return {
        "actions": [
            'prospector --profile dukit.prospector.yaml -o grouped:prospector.log',
        ]
    }

def task_mz_test():
    """mz_test"""
    return {
        "actions": [
            "python3 examples/magnetization/mz_test.py"
        ]
    }

# === START IMPORT GRAPHING
def task_imports():
    """find imports from a python module"""

    def get_imports(pkg_modules, module_path):
        module = pkg_modules.by_path[module_path]
        imports = pkg_modules.get_imports(module, return_fqn=True)
        return {'modules': list(sorted(imports))}

    base_path = pathlib.Path("src/dukit")
    pkg_modules = ModuleSet(base_path.glob('**/*.py'))
    for name, module in pkg_modules.by_name.items():
        yield {
            'name': name,
            'file_dep': [module.path],
            'actions': [(get_imports, (pkg_modules, module.path))],
        }


def task_dot():
    """generate a graphviz's dot graph from module imports"""

    def module_to_dot(imports, targets):
        graph = pygraphviz.AGraph(strict=False, directed=True)
        graph.node_attr['color'] = 'black'
        graph.node_attr['style'] = 'rounded'
        for source, sinks in imports.items():
            for sink in sinks:
                graph.add_edge(source, sink)
        graph.write(targets[0])
    return {
            'targets': ['dukit.dot'],
            'actions': [module_to_dot],
            'getargs': {'imports': ('imports', 'modules')},
            'clean': True,
        }


def task_draw():
    """generate image from a dot file"""
    return {
        'file_dep': ['dukit.dot'],
        'targets': ['dukit.png'],
        'actions': ['dot -Tpng %(dependencies)s -o %(targets)s'],
        'clean': True,
    }

# === END IMPORT GRAPHING
