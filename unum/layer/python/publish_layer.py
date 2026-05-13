"""Publish updated DAGL Python Lambda layer."""
import zipfile, os, io, boto3

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("dagl_runtime"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            filepath = os.path.join(root, f)
            arcname = os.path.join("python", filepath)
            zf.write(filepath, arcname)

client = boto3.client("lambda", region_name="eu-central-1")
resp = client.publish_layer_version(
    LayerName="dagl-runtime-python",
    Content={"ZipFile": buf.getvalue()},
    CompatibleRuntimes=["python3.11", "python3.12", "python3.13"],
    Description="DAGL runtime with multi-cloud support (conditional boto3 import)",
)
print(f"Published: {resp['LayerVersionArn']}")
print(f"Version: {resp['Version']}")
