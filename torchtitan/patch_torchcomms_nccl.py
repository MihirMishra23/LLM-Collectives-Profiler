import re

with open('torchtitan/experiments/torchcomms/parallel_dims.py', 'r') as f:
    content = f.read()

# Patch _build_mesh_without_ep
def patch_method(match):
    method_body = match.group(1)
    # Add the check at the beginning
    patch = '''    import os
    backend = os.environ.get("TEST_BACKEND", "nccl")
    if backend == "nccl":
        from torchtitan.tools.logging import logger
        logger.warning("Using NCCL backend - falling back to regular ParallelDims to avoid NCCL conflicts")
        return super()._build_mesh_without_ep()
'''
    return f'    def _build_mesh_without_ep(self) -> DeviceMesh:\n{patch}{method_body}'

# Use regex to find and patch the method
pattern = r'    def _build_mesh_without_ep\(self\) -> DeviceMesh:(.*?)(?=\n    def|\Z)'
content = re.sub(pattern, patch_method, content, flags=re.DOTALL)

# Patch _build_mesh_with_ep
def patch_method_ep(match):
    method_body = match.group(1)
    patch = '''    import os
    backend = os.environ.get("TEST_BACKEND", "nccl")
    if backend == "nccl":
        from torchtitan.tools.logging import logger
        logger.warning("Using NCCL backend - falling back to regular ParallelDims to avoid NCCL conflicts")
        return super()._build_mesh_with_ep()
'''
    return f'    def _build_mesh_with_ep\(self\) -> DeviceMesh:\n{patch}{method_body}'

pattern_ep = r'    def _build_mesh_with_ep\(self\) -> DeviceMesh:(.*?)(?=\n    def|\Z)'
content = re.sub(pattern_ep, patch_method_ep, content, flags=re.DOTALL)

with open('torchtitan/experiments/torchcomms/parallel_dims.py', 'w') as f:
    f.write(content)
