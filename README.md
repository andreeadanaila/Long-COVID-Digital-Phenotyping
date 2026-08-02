# Long-COVID-Digital-Phenotyping
Personalized Self-Supervised Digital Phenotyping for Forecasting Post-Exertional Symptom Exacerbation in Long COVID

Steps we take:
- Wearable CSV cleaning: interpolation of small gaps, explicit flagging of large gaps (device not worn), inter-patient bias check (DONE)
- Building 5+ baseline models on wearable data (simple threshold rule, logistic regression, Random Forest, Isolation Forest-type anomaly detection, one more advanced model)(DONE)
- Final forecasting model (12-48h horizon) with risk score + uncertainty interval, statistically compared against the 5 baselines (multiple random seeds, confidence intervals, per-signal ablation)(DONE)
- Federated learning, level 2 (3 clients from the 9 patients, model aggregation)
- Simple dashboard/web app showing signal graphs, current risk score, trend, alerts, and an option to share with a doctor(partial done)
  
  Sper ca e bine cat de cat ce am facut claude ul zice da acuma sper ca nu a luat-o pe carare si zice da la orice...:))))))-Monica
