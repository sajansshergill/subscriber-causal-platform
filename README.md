# Subscriber Perks Experimentation & Causal Impact Platform

A production-style analytcis platform that simlates how a streaming company optimizes a subscriber perks program using large-scale experimentation and causal inference.

This project mirrors real-world experimentation infrastructure used by companueslike Disney+, Netflix, and Huli to evaluate subscriber perks, retention strategies, and revenue optimization.

---
## Overview

Streaming platform constantly test new perks to increase retention and engagement. But correlation is not causation - leadership needs rigourous experimentation and causal analysis to make decisions worth millions.

This platform provides:

- A/B testing framework
- Geo experiments
- Causal inference modeling
- Automated experimentation pipelines
- Executive dashboards
- ROI-driven recommendations

---

## Business Problem

A stremaing company launches a **Subscriber Perks Program** that includes:

- Partner discounts
- Loyalty rewards
- Premium feature upgrades
- Early content access
- Ad-free perks

Leadership wants to know:
- Which perks increase subscriber retention?
- What is the causal impact on engagement?
- Which variants drive long-term revenue growth?
- Should experiments be rolled out globally?

This platform answers those questions using rigorous statistical methods.

--- 

## Key Features

### 1. Synthetics Subscriber Data Generator

Simulates milllions of streaming users with realisitic behavior:

- subscription tiers
- engagement minutes
- churn events
- perk exposure
- geo rollout
- marketing channels
- revenue signals

Designed to mimic real subscriber ecosystems.

---

### 2. A/B Testing Engine

End-to-end experimentation pipeline:

- randomized experiment assignment
- hypothesis testing
- lift calculation
- confidence intervals
- statistical power checks
- Bayesian experimentation support

Ouputs experiment significance and business lift.

### 3. Causal Inference Framework

Advanced methods to estimate true impact:

- Difference-in-Differences
- Propensity Score Matching
- Uplift Modeling
- Synthetic Control
- Instrumental Variables
- Causal Impact (Bayesian structural time series)

Separates correlation from causation.

---

### 4. Automated Experiment Pipelines

Airflow-powered system:

- scheduled experiment evaluation
- metric aggregation
- statistical monitoring
- significance alerts
- experiment lifecycle tracking

---

### 5. Executive Decision Dashboard

Streamlit Dahshboard for leadership:

- active experiments
- retention lift visualization
- causal impact charts
- revenue forecasting
- rollout recommendations

Designed fotr non-technical stakeholders.

## Tech Stack

- Python
- SQL / Postgres
- Pandas / NumPy
- Statsmodels
- DoWhy / EconML / CausalML
- Apache Airflow
- Stremalit
- Plotly
- Docker
- Git

Optional extensions:

- Spark / Databricks simulations
- Snowflake-style warehouse schema

---

## Architecture

```
Synthtic Data Genrator


