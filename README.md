# Customer Churn Prediction

Predicts which telecom customers are likely to leave, explains what is driving each prediction, and shows what would lower the risk.

**Live demo:** https://customer-churn-predictor-plum.vercel.app/

## What it does

- **Scores a customer** from 19 features (demographics, services, contract, billing) and returns a churn probability.
- **Explains the prediction** by showing each factor's push on the model's log-odds compared with the average customer.
- **Suggests retention levers** by re-scoring the customer with one change at a time (for example, moving to a two-year contract).
- **Shows model evaluation** with an interactive decision threshold, confusion matrix, precision/recall curve and ROC curve.

## Results

Evaluated on a stratified 20% hold-out (1,409 customers) at the default 0.50 threshold.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic regression** | 0.802 | 0.644 | 0.570 | 0.604 | 0.849 |
| Random forest (200 trees) | 0.791 | 0.625 | 0.527 | 0.572 | 0.833 |

Logistic regression won on every metric and its coefficients are interpretable, so it is the model used in the web app.

### Choosing the threshold

A missed churner costs more than a retention offer sent to a customer who would have stayed, so recall matters more than precision here. Lowering the threshold from 0.50 to 0.35 trades precision for recall:

| Threshold | Precision | Recall | F1 | Churners caught (of 374) | False alarms |
|---|---|---|---|---|---|
| 0.50 | 0.644 | 0.570 | 0.604 | 213 | 118 |
| 0.35 | 0.561 | 0.717 | 0.629 | 268 | 210 |

At 0.35 the model catches 55 more churners for 92 more false alarms.

## Key findings

- 26.5% of customers churned (1,869 of 7,043).
- Month-to-month customers churn at 42.7%, against 2.8% on two-year contracts.
- Churn is highest in the first six months (53%) and falls steadily with tenure.
- Fiber optic customers churn at 41.9%, against 19.0% for DSL.
- Customers paying by electronic check churn at 45.3%.

## How the web app works

The fitted scaler, one-hot encoder and logistic regression coefficients are exported from scikit-learn and evaluated in JavaScript. The page is a single static HTML file with no backend, so predictions are instant and it can be hosted anywhere.

## Project structure

```
app.py                  Streamlit version of the predictor
churn_analysis.ipynb    EDA, model training, evaluation and threshold tuning
data/                   IBM Telco Customer Churn dataset
web/index.html          Static web app (deployed on Vercel)
requirements.txt        Python dependencies
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py        # Streamlit app
open web/index.html         # static web app
```

## Tech stack

Python, pandas, scikit-learn, Streamlit, JavaScript, SVG

## Data

IBM Telco Customer Churn sample dataset (7,043 customers in California).
