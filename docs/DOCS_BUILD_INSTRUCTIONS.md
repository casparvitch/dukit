To build documentation:

- navigate to the root directory, e.g. /dukit/
    (which should contain the directory 'dukit')
- install pdoc: `pip3 install pdoc3` or similar 
    (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc3 --output-dir docs --html --template-dir ./docs/ --force ./dukit/`
- (or similar, may be different on windows)

- the force option overwrites any existing docs
- you may want to use skip errors option raises warnings for import errors etc. 
    (missing packages, e.g. if you haven't installed gpufit)
