"""Build and publish the DAGL Runtime Node.js Lambda Layer.

Usage:
    python build_layer_nodejs.py [--publish] [--region eu-central-1]

This script:
1. Creates a zip file with the Layer structure: nodejs/node_modules/dagl_runtime/...
2. Optionally publishes it to AWS Lambda as a Layer version

The Node.js layer has zero external dependencies - it uses only built-in
Node.js modules (crypto) and AWS SDK v3 (pre-installed in Lambda runtime).
"""

import argparse
import os
import sys
import zipfile
import subprocess
import json
from pathlib import Path


LAYER_NAME = "dagl-runtime-nodejs"
LAYER_DIR = Path(__file__).parent / "nodejs"
COMPATIBLE_RUNTIMES = ["nodejs18.x", "nodejs20.x", "nodejs22.x"]


def build_zip(output_path: str = "dagl-runtime-nodejs.zip") -> str:
    """Package the Layer into a zip file."""

    output = Path(output_path)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(LAYER_DIR):
            dirs[:] = [d for d in dirs if d != 'node_modules' or root == str(LAYER_DIR)]
            for file in files:
                if file.endswith('.map'):
                    continue
                filepath = Path(root) / file
                arcname = filepath.relative_to(LAYER_DIR.parent)
                zf.write(filepath, arcname)

    size_kb = output.stat().st_size / 1024
    print(f"  Built: {output} ({size_kb:.1f} KB)")
    print(f"  Contents:")
    with zipfile.ZipFile(output, 'r') as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"    {name} ({info.file_size} bytes)")

    return str(output)


def publish_layer(zip_path: str, region: str) -> str:
    """Publish the Layer to AWS Lambda. Returns the Layer ARN with version."""

    print(f"\n  Publishing Layer '{LAYER_NAME}' to {region}...")

    cmd = [
        "aws", "lambda", "publish-layer-version",
        "--layer-name", LAYER_NAME,
        "--description", "DAGL Runtime (Node.js) - Direct function orchestration for serverless workflows",
        "--zip-file", f"fileb://{zip_path}",
        "--compatible-runtimes", *COMPATIBLE_RUNTIMES,
        "--region", region,
        "--output", "json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)

    response = json.loads(result.stdout)
    layer_arn = response["LayerVersionArn"]
    version = response["Version"]
    size = response["Content"]["CodeSize"]

    print(f"  Published: {layer_arn}")
    print(f"  Version: {version}")
    print(f"  Size: {size / 1024:.1f} KB")

    return layer_arn


def main():
    parser = argparse.ArgumentParser(description="Build and publish DAGL Node.js Lambda Layer")
    parser.add_argument("--publish", action="store_true", help="Publish to AWS after building")
    parser.add_argument("--region", default="eu-central-1", help="AWS region (default: eu-central-1)")
    parser.add_argument("--output", default="dagl-runtime-nodejs.zip", help="Output zip file path")
    args = parser.parse_args()

    print("Building DAGL Node.js Lambda Layer...")
    zip_path = build_zip(args.output)

    if args.publish:
        layer_arn = publish_layer(zip_path, args.region)
        print(f"\n  Layer ARN: {layer_arn}")
    else:
        print(f"\n  To publish: python {__file__} --publish --region {args.region}")


if __name__ == "__main__":
    main()
