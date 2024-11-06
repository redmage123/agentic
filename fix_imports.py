import subprocess
from pathlib import Path


def fix_imports():
    """Fix imports in all Python files in the project."""
    # First, run isort
    subprocess.run(["isort", "."])

    # Then run autoflake
    subprocess.run(
        ["autoflake", "--in-place", "--remove-all-unused-imports", "--recursive", "."]
    )

    # Finally, run black
    subprocess.run(["black", "."])

    # Specific fix for protobuf generated files
    proto_dir = Path("services/protos")
    if proto_dir.exists():
        for file in proto_dir.glob("*_pb2_grpc.py"):
            content = file.read_text()
            # Fix the import statement in generated gRPC files
            content = content.replace(
                "import hamiltonian_agent_pb2", "from . import hamiltonian_agent_pb2"
            )
            file.write_text(content)


if __name__ == "__main__":
    fix_imports()
