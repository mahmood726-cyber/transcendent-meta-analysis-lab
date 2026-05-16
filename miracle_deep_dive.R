# miracle_deep_dive.R
library(metafor)

cat("===================================================================\n")
cat("  DEEP DIVE: OPIATE REDUCTION TIME-SERIES (CD015229_pub2)\n")
cat("===================================================================\n")

# 1. Data Construction (from the identified outcomes)
timepoints <- c(0.5, 1.0, 3.0)
pain_theta <- c(-0.6195, -1.1003, -1.2928)
pain_se <- c(0.2769, 0.2758, 0.4652)

opiate_theta <- c(-33.0705, -37.7743, -50.2215)
opiate_se <- c(14.8413, 11.8556, 9.7623)

df_deep <- data.frame(
    time = timepoints,
    pain = pain_theta,
    pain_se = pain_se,
    opiate = opiate_theta,
    opiate_se = opiate_se
)

cat("Consolidated Time-Series of Treatment Effects:\n")
print(df_deep)

# 2. Model 1: Cross-Domain Correlation (Pain vs Opiates)
cat("\n--- MODEL 1: Cross-Domain Correlation ---\n")
cat("CAVEAT: n=3 data points (df=1). This test has very low power.\n")
cat("Results are descriptive only and should not be used for inference.\n")
cor_test <- cor.test(df_deep$pain, df_deep$opiate)
cat(paste("Pearson r:", round(cor_test$estimate, 4), "\n"))
cat(paste("P-value:", round(cor_test$p.value, 4), "(unreliable with n=3)\n"))

# 3. Model 2: Acceleration of Clinical Benefit
cat("\n--- MODEL 2: Second Derivative (Acceleration) ---\n")
dt <- diff(timepoints)
# Central difference: f''(t1) = [(f2-f1)/dt2 - (f1-f0)/dt1] / [(t2-t0)/2]
half_span <- (timepoints[3] - timepoints[1]) / 2
accel_pain <- (diff(pain_theta)[2]/dt[2] - diff(pain_theta)[1]/dt[1]) / half_span
accel_opiate <- (diff(opiate_theta)[2]/dt[2] - diff(opiate_theta)[1]/dt[1]) / half_span
cat(paste("Acceleration of Pain Relief (per month^2):", round(accel_pain, 4), "\n"))
cat(paste("Acceleration of Opiate Reduction (per month^2):", round(accel_opiate, 4), "\n"))
if (accel_opiate < 0) {
    cat("Observation: Opiate reduction is accelerating over time.\n")
} else {
    cat("Observation: Opiate reduction is decelerating.\n")
}

# 4. Model 3: Information Distance (Fisher-Rao analogy)
cat("\n--- MODEL 3: Fisher-Rao Information Distance (14d to 3mo) ---\n")
cat("NOTE: Pain (SMD) and opiate (mg) are on different scales;\n")
cat("cross-domain distance comparisons are illustrative only.\n")
riemannian_dist <- function(t1, s1, t2, s2) {
    delta <- ((t1 - t2)^2 + 2*(s1^2 + s2^2)) / (4 * s1 * s2)
    return(sqrt(2) * acosh(delta))
}
dist_pain <- riemannian_dist(pain_theta[1], pain_se[1], pain_theta[3], pain_se[3])
dist_opiate <- riemannian_dist(opiate_theta[1], opiate_se[1], opiate_theta[3], opiate_se[3])

cat(paste("Pain consensus distance (14d to 3mo):", round(dist_pain, 4), "\n"))
cat(paste("Opiate consensus distance (14d to 3mo):", round(dist_opiate, 4), "\n"))

# 5. Model 4: Benefit Ratio Over Time
cat("\n--- MODEL 4: Opiate/Pain Benefit Ratio ---\n")
ratio <- opiate_theta / pain_theta
cat(paste("Ratio at 14d:", round(ratio[1], 4), "\n"))
cat(paste("Ratio at 3mo:", round(ratio[3], 4), "\n"))
if (ratio[3] > ratio[1]) {
    cat("Observation: The benefit ratio is increasing, suggesting opiate reduction\n")
    cat("may involve factors beyond pain relief alone (e.g., reduced opioid-seeking\n")
    cat("behavior, resolution of hyperalgesia). This is a hypothesis, not a proof.\n")
} else {
    cat("Observation: Benefit ratio is stable or decreasing.\n")
}

cat("\n===================================================================\n")
cat("  DEEP DIVE COMPLETE\n")
cat("===================================================================\n")
