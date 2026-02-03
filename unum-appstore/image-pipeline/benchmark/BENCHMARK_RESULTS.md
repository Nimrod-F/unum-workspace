
## Benchmark Results: CLASSIC vs FUTURE_BASED Execution

| Scenario | Delays (Thumb/Trans/Filt/Cont) | CLASSIC | FUTURE | Savings | Improvement |
|----------|-------------------------------|---------|--------|---------|-------------|
| Reversed (Fastest Slowest) | 4000/2000/1000/0ms | 13114ms | 8839ms | 4275ms | **32.6%** |
| Single Slow (Transform) | 0/4000/0/0ms | 12867ms | 10369ms | 2499ms | **19.4%** |
| Thumbnail Slowest | 5000/0/0/0ms | 9759ms | 8140ms | 1619ms | **16.6%** |
| Moderate Variance | 0/500/1000/1500ms | 11621ms | 9937ms | 1685ms | **14.5%** |
| Uniform (Baseline) | 0/0/0/0ms | 8058ms | 6891ms | 1167ms | **14.5%** |
| Single Slow (Filters) | 0/0/4000/0ms | 11642ms | 10014ms | 1628ms | **14.0%** |
| All Equal (2s each) | 2000/2000/2000/2000ms | 11804ms | 10245ms | 1560ms | **13.2%** |
| Exponential Growth | 0/500/1500/4000ms | 13802ms | 12171ms | 1631ms | **11.8%** |
| Bimodal (2 Fast, 2 Slow) | 0/0/3000/3000ms | 12660ms | 11234ms | 1426ms | **11.3%** |
| Staggered Delays | 0/1000/2000/3000ms | 11207ms | 9962ms | 1245ms | **11.1%** |
| Three Fast, One Slow | 0/0/0/6000ms | 15957ms | 14227ms | 1730ms | **10.8%** |
| Extreme Outlier | 0/0/0/5000ms | 13263ms | 11905ms | 1358ms | **10.2%** |

### Summary
- **Scenarios tested**: 12
- **Average improvement**: 15.0%
- **Average savings**: 1819ms
- **Best improvement**: 32.6%
- **Maximum savings**: 4275ms

### Key Findings
1. **Future-Based execution consistently outperforms Classic mode** across all scenarios
2. **Largest gains** occur when branch execution times vary significantly
3. **Even with equal delays**, Future-Based still wins due to early fan-in start
4. **The reversed scenario** (where the naturally fastest branch becomes slowest) shows the highest improvement
