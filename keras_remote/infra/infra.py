import logging
import os


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("keras_remote")


def get_default_project():
  return os.environ.get("KERAS_REMOTE_PROJECT")
