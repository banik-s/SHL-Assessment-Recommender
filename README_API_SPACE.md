---
title: SHL Assessment Recommender API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SHL Assessment Recommender — API

FastAPI backend for the SHL Assessment Recommendation Engine.

## Endpoint

```
POST /recommend
```

**Request body:**
```json
{
  "query": "I am hiring Java developers who collaborate with business teams",
  "top_k": 10
}
```

**Response:**
```json
{
  "query": "...",
  "expanded_query": "...",
  "total_returned": 10,
  "recommendations": [
    {
      "name": "Core Java (Entry Level)",
      "url": "https://www.shl.com/...",
      "test_type": "K",
      "duration_mins": 15
    }
  ]
}
```

## Health check
```
GET /health
```
