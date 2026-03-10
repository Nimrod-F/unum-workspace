### **Classic Mode: Non-Fusion vs. Fusion**

| Metric                          | Classic (Non-Fusion) | Classic (Fusion) | Change     |
| ------------------------------- | -------------------- | ---------------- | ---------- |
| **Mean E2E Latency**            | 5,337 ms             | 4,174 ms         | **-21.8%** |
| **Estimated Cost per Run**      | $0.000124            | $0.000107        | **-13.7%** |
| **Total Memory Used**           | 549 MB               | 369 MB           | **-32.8%** |
| **Cold Starts**                 | 6                    | 4                | **-33.3%** |
| **Aggregator Invocation Delay** | 5,160 ms             | 3,994 ms         | **-22.6%** |

### **Future-Based Mode: Non-Fusion vs. Fusion**

| Metric                          | Futures (Non-Fusion) | Futures (Fusion) | Change     |
| ------------------------------- | -------------------- | ---------------- | ---------- |
| **Mean E2E Latency**            | 4,797 ms             | 3,688 ms         | **-23.1%** |
| **Estimated Cost per Run**      | $0.000168            | $0.000160        | **-4.8%**  |
| **Total Memory Used**           | 475 MB               | 367 MB           | **-22.7%** |
| **Cold Starts**                 | 6                    | 4                | **-33.3%** |
| **Aggregator Invocation Delay** | 1,246 ms             | 1,014 ms         | **-18.7%** |

###

###

| Metric                | Classic (Non-Fusion) | Classic (Fusion) | Classic Gain/Loss | Future (Non-Fusion) | Future (Fusion) | Future Gain/Loss |
| --------------------- | -------------------- | ---------------- | ----------------- | ------------------- | --------------- | ---------------- |
| **Mean E2E Latency**  | 5,337 ms             | 4,174 ms         | **-21.8%**        | 4,797 ms            | 3,688 ms        | **-23.1%**       |
| **Estimated Cost**    | $0.000124            | $0.000107        | **-13.7%**        | $0.000168           | $0.000160       | **-4.8%**        |
| **Total Memory Used** | 549 MB               | 369 MB           | **-32.8%**        | 475 MB              | 367 MB          | **-22.7%**       |
| **Total Cold Starts** | 6                    | 4                | **-33.3%**        | 6                   | 4               | **-33.3%**       |
| **Aggregator Delay**  | 5,160 ms             | 3,994 ms         | **-22.6%**        | 1,246 ms            | 1,014 ms        | **-18.7%**       |
