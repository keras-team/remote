from keras_remote.cli.config import InfraConfig
from keras_remote.core import accelerators
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import get_stack

config = InfraConfig(
    project="keras-team-gcp",
    zone="us-central1-a",
    cluster_name="keras-remote-cluster",
    accelerator=accelerators.parse_accelerator("v5litepod-8"),
)

program = create_program(config)
stack = get_stack(program, config)

print("Refreshing stack...")
stack.refresh(on_output=print)
print("Stack refreshed.")
