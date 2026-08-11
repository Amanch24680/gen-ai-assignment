# Cost-Efficient RAG System Specification

## Overview
This document specifies a lightweight Retrieval-Augmented Generation (RAG) application architecture.

## Vector Store
The system utilizes Qdrant running in local mode for vector storage and similarity search.

## Chunking Strategy
Documents are ingested and split into deterministic text chunks using a configurable sliding window with defined chunk size and chunk overlap.