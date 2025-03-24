#!/usr/bin/env python3
import pkg_resources

# 核心依赖列表——只包含 Search API 所需的包
core_packages = [
    "fastapi",
    "uvicorn",
    "faiss-cpu",
    "sentence-transformers",
    "torch",
    "transformers",
    "numpy",
    "pydantic",
    "requests",
    "sqlite3"  # Python 内建，可选
]

with open("requirements.txt", "w") as fout:
    for pkg in core_packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            fout.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            print(f"WARNING: {pkg} not installed, skipping")
