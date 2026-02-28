# multi-aggregator

## Workflow Diagram

```
A (start) --> B ----------\
          \-> C -----------> D (aggregator) --> E --> F
```

## Functions (6)

- **A**: receive search query, split into sub-queries
- **B**: search database alpha
- **C**: search database beta
- **D**: merge and deduplicate search results
- **E**: rank merged results by relevance
- **F**: format and return final ranked results

