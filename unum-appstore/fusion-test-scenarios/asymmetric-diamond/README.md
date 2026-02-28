# asymmetric-diamond

## Workflow Diagram

```
A (start) --> B --> C -----------\
          \-> D -----------------> G (aggregator)
          \-> E --> F ----------/
```

## Functions (7)

- **A**: receive data and distribute to enrichment services
- **B**: query primary database
- **C**: format primary database results
- **D**: quick cache lookup (lightweight, low memory)
- **E**: call external API (heavy, high memory)
- **F**: parse and normalize external API response
- **G**: merge all enriched data sources

