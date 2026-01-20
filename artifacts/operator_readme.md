# Retention Campaign Brief

## Objective
Select a limited number of customers for a retention campaign, maximizing expected net value under capacity constraints.

---

## Data & Setup
- Dataset: Online Retail transactional data
- Snapshot date: 2011-10-10
- Lookback window: 180 days
- Prediction horizon: 60 days
- Capacity (Top-K): 500 customers

---

## Policy Assumptions (Offline Simulation)
Customer value is segmented based on historical spend (monetary_180d):

- Low value:  €5 coupon, 5% uplift  
- Mid value:  €5 coupon, 10% uplift  
- High value: €10 coupon, 30% uplift  

Expected Value (EV) is computed as:

EV = churn_risk × uplift × customer_value − incentive_cost

---

## Key Observation
At K = 500, the EV-optimal ranking selects exclusively high-value customers receiving €10 incentives.

This outcome is driven by the combination of high customer value and strong assumed uplift in the high-value segment, which dominates lower-cost alternatives in EV ranking under the current capacity constraint.

While segmentation shows increasing benefits at larger campaign scale, execution-level optimization at K = 500 indicates that concentrating resources on high-value customers maximizes expected return given the current assumptions.

---

## Results (Offline Evaluation)
- Base churn rate: ~0.47  
- Precision@500: materially higher than baseline risk-based targeting  
- Cumulative expected value (Top-500): ~€138k  
- All selected customers have positive individual EV

---

## Recommendation
Proceed with the Top-500 EV-ranked customer list and apply €10 incentives to the selected customers.

If campaign capacity increases or uplift assumptions change, re-evaluate allocation across value segments to capture additional upside.

---

## Notes & Limitations
- Uplift rates are assumed and should be validated via A/B testing.
- Results are sensitive to policy assumptions and customer value distribution.
- This analysis focuses on short-term retention value and does not model long-term lifetime effects.