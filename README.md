## Chess Outcome Prediction – Model Comparison

This repository contains the code used in our DS3000 Group 18 project, where we predict chess game outcomes (white win vs. black win) from **Elo ratings** and **opening information** using machine learning.

### 1. Project Overview

We use a large Lichess game dataset (~6M games) and train several classification models to predict whether **White wins (1)** or **loses (0)** based on:

- Players’ Elo ratings  
- The opening played (ECO code)  
- The time control category (bullet, blitz, rapid, classical)

We compare a family of **Logistic Regression** models against a family of **Random Forest** models to understand the trade-off between interpretability and predictive performance.

And we look at the best chess opening per ELO bracket.

---


