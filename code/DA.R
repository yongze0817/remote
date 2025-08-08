library(grf)
library(DiagrammeR)

W <- data_scenario_G_n_500_dataset_0[, c("W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10")]
Y <- data_scenario_G_n_500_dataset_0$Y
T <- data_scenario_G_n_500_dataset_0$T


tau.forest<-causal_forest(W, Y, T, num.trees = 2000)
average_treatment_effect(tau.forest,target.sample = "all", method = "AIPW")
