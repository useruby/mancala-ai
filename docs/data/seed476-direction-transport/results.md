# Seed476: full-population versus minibatch direction transport

Retrospective and exploratory read-only diagnostic; descriptive attribution only, with no optimizer or sampling causality and no strength claim.

| Step | Weighting | Full negative | Minibatch negative | Recorded Adam | Category |
|---:|---|---:|---:|---:|---|
| 1 | exposure | -0.259968822 | -0.286273064 | -0.120634353 | both_descent |
| 1 | equal_input | -0.242378833 | -0.281904754 | -0.120528755 | both_descent |
| 2 | exposure | -0.138200716 | -0.128316929 | -0.0475558393 | both_descent |
| 2 | equal_input | -0.118690457 | -0.119456072 | -0.0450525248 | both_descent |
| 3 | exposure | -0.0844053495 | -0.028406125 | -0.00158106486 | both_descent |
| 3 | equal_input | -0.0653004121 | -0.0168477312 | 0.00237801465 | adam_coordinate_transport_failure |
| 4 | exposure | 0.00492619835 | -0.491246467 | -0.178699077 | mixed_or_ambiguous |
| 4 | equal_input | 0.0200693845 | -0.479986525 | -0.176626017 | mixed_or_ambiguous |
| 5 | exposure | 0.126761956 | 0.158729804 | 0.0598194596 | mixed_or_ambiguous |
| 5 | equal_input | 0.144279662 | 0.161851027 | 0.0611762481 | mixed_or_ambiguous |
| 6 | exposure | 0.115530229 | -0.048318624 | 0.0104053609 | adam_coordinate_transport_failure |
| 6 | equal_input | 0.129826636 | -0.0501058685 | 0.0111436309 | adam_coordinate_transport_failure |
| 7 | exposure | 0.117850597 | -0.14502959 | -0.0596186396 | mixed_or_ambiguous |
| 7 | equal_input | 0.131280789 | -0.157262006 | -0.0641194467 | mixed_or_ambiguous |
| 8 | exposure | 0.13552004 | 0.145677766 | 0.0733513474 | mixed_or_ambiguous |
| 8 | equal_input | 0.153046257 | 0.153545196 | 0.0753092425 | mixed_or_ambiguous |
| 9 | exposure | 0.074142807 | 0.271291545 | 0.112339352 | mixed_or_ambiguous |
| 9 | equal_input | 0.084349135 | 0.28006138 | 0.115634914 | mixed_or_ambiguous |
| 10 | exposure | -0.096755667 | 0.131940963 | 0.0452573869 | minibatch_transport_failure |
| 10 | equal_input | -0.0917731953 | 0.133224946 | 0.0451036469 | minibatch_transport_failure |
| 11 | exposure | -0.192780788 | 0.308397826 | 0.11929132 | minibatch_transport_failure |
| 11 | equal_input | -0.188961758 | 0.312355374 | 0.120545923 | minibatch_transport_failure |
| 12 | exposure | -0.524771869 | -0.128717782 | -0.0591907282 | both_descent |
| 12 | equal_input | -0.523607926 | -0.132213892 | -0.0597603712 | both_descent |
| 13 | exposure | -0.427944301 | 0.100576566 | 0.0520131718 | minibatch_transport_failure |
| 13 | equal_input | -0.427859696 | 0.105791811 | 0.0538737238 | minibatch_transport_failure |
| 14 | exposure | -0.571214096 | -0.139744536 | -0.0349757506 | both_descent |
| 14 | equal_input | -0.571561328 | -0.144610314 | -0.0357286326 | both_descent |
| 15 | exposure | -0.582324519 | -0.260092706 | -0.106765689 | both_descent |
| 15 | equal_input | -0.582179442 | -0.25443055 | -0.105874471 | both_descent |
| 16 | exposure | -0.498702933 | 0.0672429983 | 0.00678543973 | minibatch_transport_failure |
| 16 | equal_input | -0.498928033 | 0.0677083184 | 0.00746852499 | minibatch_transport_failure |

No training arm, optimizer sweep, searches, games, runtime export, promotion, or strength claim was performed.
