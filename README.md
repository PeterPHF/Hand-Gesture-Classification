# Hand-Gesture-Classification

## Model Comparison

| Run Name                         | Model                | Accuracy   | F1 Score   | Precision  | Recall     |
|----------------------------------|----------------------|------------|------------|------------|------------|
| xgb_default                      | XGBoost              | 0.98033    | 0.98035    | 0.98046    | 0.98033    |
| gridsearch_svc_C30_ovo_gamma2    | SVC (GridSearch)     | 0.97644    | 0.97648    | 0.97661    | 0.97644    |
| svc_poly_ovo_d10                 | SVC (Poly, d=10)     | 0.97371    | 0.97378    | 0.97398    | 0.97371    |
| svc_poly_ovo_d15                 | SVC (Poly, d=15)     | 0.97332    | 0.97339    | 0.97355    | 0.97332    |
| gridsearch_svc_C20_ovo_gamma1    | SVC (GridSearch)     | 0.97176    | 0.97179    | 0.97206    | 0.97176    |
| gridsearch_svc_C10_ovo_gamma1    | SVC (GridSearch)     | 0.96670    | 0.96675    | 0.96715    | 0.96670    |
| random_forest_default            | Random Forest        | 0.96436    | 0.96437    | 0.96450    | 0.96436    |
| ovr_logistic_cv                  | Logistic Regression  | 0.93437    | 0.93423    | 0.93503    | 0.93437    |
| svc_poly_ovo_d5                  | SVC (Poly, d=5)      | 0.92795    | 0.92851    | 0.93215    | 0.92795    |
| svc_poly_ovo_d3                  | SVC (Poly, d=3)      | 0.83739    | 0.83668    | 0.84916    | 0.83739    |