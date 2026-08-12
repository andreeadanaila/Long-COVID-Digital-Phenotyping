# Privacy and Ethical Analysis

## Privacy

This project uses wearable health data, which is sensitive personal information. The system therefore follows a privacy-aware approach to the handling of patient data.

Patient data is represented using a `user_id` rather than a person's name. The system does not require names, addresses, contact details, or other direct identifiers for model training. The dashboard operates on patient IDs rather than displaying direct personal identifiers.

The dataset contains wearable measurements including heart rate, heart-rate variability, SpO2, respiration rate, steps, distance, and body battery. These measurements can reveal sensitive information about a person's health and daily behaviour. Access to the underlying dataset and generated predictions should therefore be restricted to authorised users.

The federated-learning component provides an additional privacy-preserving architecture. Instead of requiring all client data to be pooled centrally, the simulated clients train locally and communicate model parameters for aggregation. The current implementation demonstrates this concept using separate groups of patients and Federated Averaging (FedAvg).

Federated learning by itself does not provide a formal guarantee that individual information cannot be inferred from model updates. A production system could therefore require additional protections such as secure aggregation, differential privacy, encryption, strict access controls, and audit logging.

## Ethical Considerations

The system is intended as a decision-support and monitoring tool, not as an autonomous medical diagnostic system. A predicted risk score should therefore not be treated as a diagnosis or as a substitute for clinical judgement.

The available wearable dataset does not contain self-reported symptom labels. Consequently, the model's risk target is based on deviations in wearable signals. A predicted risk therefore represents the likelihood of a future wearable-defined risk event rather than a confirmed Long COVID symptom exacerbation.

This distinction is important to avoid overstating what the model can determine. False positives could cause unnecessary concern, while false negatives could cause a potential deterioration to be overlooked. Predictions should therefore be considered together with uncertainty and appropriate clinical context.

The model is personalised using each patient's own historical measurements. This can reduce dependence on a single population-wide threshold, but it can also introduce limitations when a patient has insufficient historical data, unusual behaviour, or changes in their health state.

Potential bias and limited generalisability are also concerns. The available dataset represents a limited patient population, so model performance may not generalise to different demographic groups, disease severities, wearable devices, or healthcare settings. External validation on additional patient populations is therefore important before clinical deployment.

The dashboard includes a mechanism for sharing a summary with a doctor. In a real deployment, sharing should require explicit user consent and use secure authentication and encrypted communication.

## Data Governance and Responsible Deployment

Before real-world clinical use, the system should operate under appropriate institutional approval, informed-consent procedures, data-minimisation policies, retention limits, access controls, and secure storage.

The current project is a research prototype and should not be presented as clinically validated software. Further prospective and external validation would be required before using its predictions to make clinical decisions.
