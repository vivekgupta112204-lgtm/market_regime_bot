# Platform Troubleshooting FAQ

### Q1. Node Socket Drops Repeatedly
Check `multi_broker/failover.py` log traces inside the ELK mapping. If HA triggers perpetually, network configurations surrounding TCP NAT-timeout windows need extension.

### Q2. Model Continuously Retraining
Reduce the `DriftDetector` sensitivity inside `mlops/drift_detector.py`. High frequency volatility flags Gaussian assumptions heavily, triggering `Concept Drift` retrains rapidly.
