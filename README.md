# Aerial Flower Intensity

## Installing the module

We recommend installing this module into a dedicated Python virtual environment to avoid dependency
conflicts or polluting the global Python distribution.

### From source

For this install to work, **you need to have the git command available in your terminal**.

Install the module from the local source with pip:

```shell
pip install .
```

### Checking the installation

Now, you should have a new command ``aerial-flower-intensity`` available in your terminal
and in any directory when the corresponding vitual environment is activated. You can test it with
the following command to display the help:

```shell
aerial-flower-intensity --help
```

## Installing for development

If you need to work on the module for maintenance or updates, always use a dedicated Python virtual
environment. Install the module in development mode with pip. As for the source installation, the git
command must be available in your terminal.

```shell
pip install -e .
```

In development mode, pip will install a reference to the source code instead of doing a full install.
This will allow to update the source code and directly see the modified behavior with the installed
``aerial-flower-intensity`` command.

You also need to install [tox](https://tox.readthedocs.io) for doing unit tests,
static code analysis and generating the documentation.

```shell
# Running all tests and static code analysis environments
tox
# Running only the static code analysis
tox run -e pylint
# Generating the documentation
tox run -e docs
```

All defined tox environments can be listed as follow:

```shell
# List all defined environments
tox list
```

## Building the Docker image

First, create a directory named ``wheels`` into the root directory. Place any needed private wheels
inside it before building the image.

This project contains a ``Makefile`` to build the Docker image on **Linux**:

```shell
make build
```

Once done, you should have a new Docker image called ``aerialflowerintensity`` that you can
directly use to run the module. For example:

```shell
docker run --rm aerialflowerintensity --help
```
