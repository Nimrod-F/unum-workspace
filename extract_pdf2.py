import re, zlib
with open(r'c:\Users\foldv\Downloads\CLOSER_Camera_ready.pdf', 'rb') as f:
    data = f.read()
streams = list(re.finditer(rb'stream\r?\n(.+?)\r?\nendstream', data, re.DOTALL))
# Check all streams for longer text content
for i, m in enumerate(streams):
    try:
        decompressed = zlib.decompress(m.group(1))
        # Look for Tj or TJ operators (text showing)
        tj_matches = re.findall(rb'\(([^)]{20,})\)\s*Tj', decompressed)
        for t in tj_matches[:5]:
            print("Stream %d Tj: %s" % (i, t.decode("latin-1", errors="replace")))
        # Also look for TJ arrays
        tj2 = re.findall(rb'\[([^\]]{50,})\]\s*TJ', decompressed)
        for t in tj2[:3]:
            # Extract strings from the array
            parts = re.findall(rb'\(([^)]+)\)', t)
            combined = b''.join(parts)
            text = combined.decode("latin-1", errors="replace")
            if len(text) > 15:
                print("Stream %d TJ: %s" % (i, text))
    except:
        pass
