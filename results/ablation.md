| method | acc | macro_f1 | latency_ms | params |
|---|---|---|---|---|
| Rule-based (fingers up) | 89.89 | 90.04 | 0.033 |  |
| SVM (RBF), full features | 99.74 | 99.74 | 0.150 |  |
| Random Forest, full features | 99.84 | 99.84 | 10.871 |  |
| kNN (k=5), full features | 99.77 | 99.78 | 1.630 |  |
| MLP, raw features | 99.67 | 99.67 | 0.222 | 17287 |
| MLP, norm features | 99.80 | 99.80 | 0.293 | 17287 |
| MLP, norm_rot features | 99.80 | 99.80 | 0.317 | 17287 |
| MLP, full features, no augmentation | 99.77 | 99.78 | 0.776 | 20487 |
| MLP, full features | 99.74 | 99.74 | 0.720 | 20487 |
