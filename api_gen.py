"""Script to generate keras_remote public API in `keras_remote/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import shutil
import namex

PACKAGE = "keras_remote"


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def copy_source_to_build_directory(root_path):
    # Copy sources (`keras_remote/` directory and setup files) to build dir
    build_dir = os.path.join(root_path, "tmp_build_dir")
    build_package_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_package_dir, "src")
    root_src_dir = os.path.join(root_path, PACKAGE, "src")
    
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_package_dir)
    shutil.copytree(root_src_dir, build_src_dir)
    return build_dir


def export_version_string(api_init_fname):
    with open(api_init_fname) as f:
        contents = f.read()
    with open(api_init_fname, "w") as f:
        # We assume version is available in src.version or similar, 
        # but for now we might skip valid version import if not present.
        # Check if src/version.py exists or similar.
        # For now, let's just leave it or import from setup?
        # Keras does: from keras.src.version import __version__ as __version__
        # We'll try to do the same if we have a version file.
        pass


def build():
    root_path = os.path.dirname(os.path.abspath(__file__))
    code_api_dir = os.path.join(root_path, PACKAGE, "api")
    # Create temp build dir
    build_dir = copy_source_to_build_directory(root_path)
    build_api_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_api_dir, "src")
    build_api_init_fname = os.path.join(build_api_dir, "__init__.py")
    
    try:
        os.chdir(build_dir)
        open(build_api_init_fname, "w").close()
        namex.generate_api_files(
            PACKAGE,
            code_directory="src",
        )
        
        # Copy back the keras_remote/api from build directory
        if os.path.exists(code_api_dir):
            shutil.rmtree(code_api_dir)
        shutil.copytree(
            build_api_dir, code_api_dir, ignore=shutil.ignore_patterns("src")
        )
    finally:
        # Clean up: remove the build directory (no longer needed)
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)


if __name__ == "__main__":
    build()
