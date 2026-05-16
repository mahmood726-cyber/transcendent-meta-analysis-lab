# massive_r_synthesis.R
library(brms)
library(metafor)

cat("===================================================================\n")
cat("  BAYESIAN HIERARCHICAL META-ANALYSIS (R)\n")
cat("===================================================================\n")

# 1. Load Data (Pairwise 70 Results)
data_dir <- Sys.getenv("TMAL_R_DATA_DIR", unset = file.path(getwd(), "data"))
data_path <- file.path(data_dir, "ma4_results_pairwise70.csv")
if (!file.exists(data_path)) {
    stop(paste("Data file not found:", data_path,
               "\nSet TMAL_R_DATA_DIR to the directory containing ma4_results_pairwise70.csv"))
}
df <- read.csv(data_path)
df$theta <- as.numeric(df$theta)
df$sigma <- as.numeric(df$sigma)
df <- df[!is.na(df$theta) & !is.na(df$sigma), ]

set.seed(42)

cat(paste("Analyzing", nrow(df), "Meta-Analytical outcomes across", length(unique(df$review_id)), "Cochrane reviews.\n"))

# 2. Model 1: Robust Bayesian Hierarchical Meta-Analysis
# Student-t likelihood to handle heavy-tailed effects
cat("\n--- MODEL 1: Robust Bayesian Hierarchical Model (t-dist) ---\n")
fit_robust <- brm(
  formula = theta | se(sigma) ~ 1 + (1 | review_id),
  data = df,
  family = student(),
  prior = c(
    prior(normal(0, 1), class = "Intercept"),
    prior(cauchy(0, 0.5), class = "sd")
  ),
  iter = 2000, chains = 4, cores = min(4, parallel::detectCores() - 1),
  backend = "rstan", seed = 42,
  silent = 2, refresh = 0
)

cat("Summary of Global Bayesian Effect (Intercept):\n")
print(fixef(fit_robust))

# 3. Model 2: Intraclass Correlation
cat("\n--- MODEL 2: Intraclass Correlation ---\n")
var_comp <- VarCorr(fit_robust)
tau2 <- var_comp$review_id$sd[1]^2
avg_within <- mean(df$sigma^2)
icc <- tau2 / (tau2 + avg_within)
cat(paste("ICC:", round(icc, 4), "\n"))
cat(paste("  (tau^2 =", round(tau2, 4), ", avg within-study var =", round(avg_within, 4), ")\n"))
cat("NOTE: ICC uses mean sampling variance as approximation for within-study component.\n")

# 4. Model 3: Tail Analysis
cat("\n--- MODEL 3: Extreme Tail Analysis ---\n")
tails <- aggregate(theta ~ review_id, data = df, FUN = min)
mu_tail <- mean(tails$theta)
sd_tail <- sd(tails$theta)
cat(paste("Mean of review-level minima:", round(mu_tail, 4), "\n"))
cat(paste("SD of review-level minima:", round(sd_tail, 4), "\n"))
cat("NOTE: Block minima summary, not a formal GEV fit.\n")

# 5. Information-Theoretic Analysis
cat("\n--- MODEL 4: Shannon Entropy of Effect Directionality ---\n")
p_beneficial <- mean(df$theta < 0)
eps <- 1e-10
p_safe <- max(eps, min(1 - eps, p_beneficial))
shannon_entropy <- - (p_safe * log2(p_safe) + (1 - p_safe) * log2(1 - p_safe))
cat(paste("Shannon Entropy:", round(shannon_entropy, 4), "bits\n"))
cat(paste("Proportion beneficial (theta < 0):", round(p_beneficial, 4), "\n"))
if (shannon_entropy < 0.9) {
    cat("Observation: Literature is skewed toward beneficial effects (Low Entropy).\n")
} else {
    cat("Observation: Literature is balanced between beneficial and non-beneficial (High Entropy).\n")
}

cat("\n===================================================================\n")
cat("  BAYESIAN ANALYSIS COMPLETE\n")
cat("===================================================================\n")
