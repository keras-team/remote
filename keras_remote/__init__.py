# This file should NEVER be packaged! This is a hack to make "import keras_remote" from
# the base of the repo just import the source files. We'll keep it for compat.

import os  # isort: skip

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

from keras_remote.api import *  # noqa: F403, E402

# Don't pollute namespace.
del os
