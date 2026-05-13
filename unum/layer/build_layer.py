"""Build and publish the DAGL Runtime Lambda Layer.

Usage:
    python build_layer.py [--publish] [--region eu-central-1]

This script:
1. Creates a zip file with the Layer structure: python/dagl_runtime/...
2. Optionally publishes it to AWS Lambda as a Layer version
"""

import argparse
import os
import sys
import zipfile
import subprocess
import json
from pathlib import Path


LAYER_NAME = "dagl-runtime-python"
LAYER_DIR = Path(__file__).parent / "python"
COMPATIBLE_RUNTIMES = ["python3.10", "python3.11", "python3.12", "python3.13"]


def build_zip(output_path: str = "dagl-runtime-python.zip") -> str:
    """Package the Layer into a zip file."""
    
    output = Path(output_path)
    if output.exists():
        output.unlink()
    
    # Install Layer dependencies into the package
    deps_dir = LAYER_DIR / "dagl_runtime" / "_deps"
    deps_dir.mkdir(exist_ok=True)
    
    # boto3 is pre-installed in Lambda, so we only need minimal deps
    # The Layer should be lightweight
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(LAYER_DIR):
            # Skip __pycache__ and _deps
            dirs[:] = [d for d in dirs if d != '__pycache__' and d != '_deps']
            for file in files:
                if file.endswith('.pyc'):
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
        "--description", "DAGL Runtime - Direct function orchestration for serverless workflows",
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
    parser = argparse.ArgumentParser(description="Build and publish DAGL Python Lambda Layer")
    parser.add_argument("--publish", action="store_true", help="Publish to AWS after building")
    parser.add_argument("--region", default="eu-central-1", help="AWS region (default: eu-central-1)")
    parser.add_argument("--output", default="dagl-runtime-python.zip", help="Output zip file path")
    args = parser.parse_args()
    
    print("Building DAGL Runtime Layer (Python)...")
    zip_path = build_zip(args.output)
    
    if args.publish:
        layer_arn = publish_layer(zip_path, args.region)
        
        # Save ARN for use by dagl deploy
        arn_file = Path(__file__).parent / "layer-arn.txt"
        arn_file.write_text(layer_arn)
        print(f"\n  Layer ARN saved to: {arn_file}")
        print(f"\n  Use with: dagl deploy --layer-arn {layer_arn}")
    else:
        print(f"\n  To publish: python build_layer.py --publish --region {args.region}")


if __name__ == "__main__":
    main()
