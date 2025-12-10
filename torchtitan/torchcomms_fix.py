import os

# Read the file
with open('torchtitan/experiments/torchcomms/parallel_dims.py', 'r') as f:
    lines = f.readlines()

# Find _build_mesh_without_ep method
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    if 'def _build_mesh_without_ep(self) -> DeviceMesh:' in line:
        # Add our patch after the method signature
        indent = ' ' * 4  # 4 spaces for class method
        patch_lines = [
            f'{indent}import os\n',
            f'{indent}backend = os.environ.get("TEST_BACKEND", "nccl")\n',
            f'{indent}if backend == "nccl":\n',
            f'{indent}    from torchtitan.tools.logging import logger\n',
            f'{indent}    logger.warning("Using NCCL backend - falling back to regular ParallelDims to avoid NCCL conflicts")\n',
            f'{indent}    return super()._build_mesh_without_ep()\n',
        ]
        new_lines.extend(patch_lines)
    
    elif 'def _build_mesh_with_ep(self) -> DeviceMesh:' in line:
        # Add our patch after the method signature
        indent = ' ' * 4
        patch_lines = [
            f'{indent}import os\n',
            f'{indent}backend = os.environ.get("TEST_BACKEND", "nccl")\n',
            f'{indent}if backend == "nccl":\n',
            f'{indent}    from torchtitan.tools.logging import logger\n',
            f'{indent}    logger.warning("Using NCCL backend - falling back to regular ParallelDims to avoid NCCL conflicts")\n',
            f'{indent}    return super()._build_mesh_with_ep()\n',
        ]
        new_lines.extend(patch_lines)
    
    i += 1

# Write back
with open('torchtitan/experiments/torchcomms/parallel_dims.py', 'w') as f:
    f.writelines(new_lines)

print("Patched successfully!")
