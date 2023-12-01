To build documentation:

- navigate to the root directory, e.g. /dukit/
    (which should contain the directory 'dukit')
- install pdoc: `pip3 install pdoc3` or similar 
    (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc3 --output-dir docs/ --html --template-dir docs/ --force --skip-errors .src/dukit/`
- (or similar, may be different on windows)
- OR you can run `doit docs` from the root directory, with pydoit installed.

- the force option overwrites any existing docs
- skip-errors will avoid issues e.g. if you don't have pygpufit installed.
