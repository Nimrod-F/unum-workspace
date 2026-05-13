import re, zlib
with open(r'c:\Users\foldv\Downloads\CLOSER_Camera_ready.pdf', 'rb') as f:
    data = f.read()
streams = list(re.finditer(rb'stream\r?\n(.+?)\r?\nendstream', data, re.DOTALL))
for i, m in enumerate(streams[:15]):
    try:
        decompressed = zlib.decompress(m.group(1))
        texts = re.findall(rb'\(([^)]{15,})\)', decompressed)
        for t in texts[:3]:
            print("Stream %d: %s" % (i, t.decode("latin-1", errors="replace")))
        hexts = re.findall(rb'<([0-9A-Fa-f]{30,})>', decompressed)
        for h in hexts[:2]:
            try:
                raw = bytes.fromhex(h.decode("ascii"))
                decoded = raw.decode("utf-16-be", errors="replace")
                if len(decoded) > 10:
                    print("Stream %d hex: %s" % (i, decoded))
            except:
                pass
    except:
        pass
