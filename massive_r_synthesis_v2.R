# massive_r_synthesis_v2.R
library(metafor)
library(Matrix)

cat("===================================================================\n")
cat("  HIERARCHICAL META-ANALYSIS V2: MULTI-LEVEL + META-REGRESSION\n")
cat("===================================================================\n")

# 1. Load Data
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

cat(paste("Analyzing", nrow(df), "Meta-Analytical outcomes across", length(unique(df$review_id)), "Cochrane reviews.\n"))

# 2. Model 1: Multi-Level Random Effects (Hierarchical)
cat("\n--- MODEL 1: Multi-Level Random Effects (Hierarchical) ---\n")
res_ml <- rma.mv(theta, sigma^2, random = ~ 1 | review_id / analysis_number, data = df)
print(summary(res_ml))

# 3. Model 2: Meta-Regression (log(k) as moderator)
cat("\n--- MODEL 2: Meta-Regression (Evidence Volume) ---\n")
# Guard against k <= 0
df_reg <- df[!is.na(df$k) & df$k > 0, ]
if (nrow(df_reg) > 10) {
    res_reg <- rma.mv(theta, sigma^2, mods = ~ log(k), random = ~ 1 | review_id, data = df_reg)
    cat("Meta-Regression Summary (Effect of log(k) on pooled estimate):\n")
    print(summary(res_reg))
} else {
    cat("Insufficient rows with valid k > 0 for meta-regression.\n")
}

# 4. Model 3: Heteroskedastic Signal Variance
cat("\n--- MODEL 3: Variance-Volume Correlation ---\n")
ma_var_analysis <- aggregate(theta ~ review_id, data = df, FUN = var)
ma_k_analysis <- aggregate(k ~ review_id, data = df, FUN = sum)
merged_analysis <- merge(ma_var_analysis, ma_k_analysis, by="review_id")
cor_val <- cor(merged_analysis$theta, merged_analysis$k, use="complete.obs")
cat(paste("Correlation between total k and effect variance:", round(cor_val, 4), "\n"))

# 5. Model 4: Extreme Tail Analysis (Top 1%)
cat("\n--- MODEL 4: Extreme Tail Analysis (Top 1%) ---\n")
sorted_theta <- sort(df$theta)
tail_1_percent <- head(sorted_theta, max(1, floor(nrow(df)*0.01)))
cat(paste("Mean of top 1% most extreme effects:", round(mean(tail_1_percent), 4), "\n"))
cat(paste("Threshold (1st percentile):", round(max(tail_1_percent), 4), "\n"))
cat("NOTE: Extreme tails expected from hundreds of MAs; no multiple testing correction applied.\n")

# 6. Spectral Analysis of Cross-Outcome Structure
cat("\n--- MODEL 5: Spectral Analysis of Outcome Covariance ---\n")
ma_ids <- as.numeric(as.factor(df$review_id))
n_mas <- length(unique(ma_ids))
min_outcomes <- 10
# Only include reviews with >= min_outcomes outcomes (avoid zero-padding bias)
eligible <- which(table(ma_ids) >= min_outcomes)
if (length(eligible) >= 5) {
    theta_mat <- matrix(NA, nrow = length(eligible), ncol = min_outcomes)
    for (idx in seq_along(eligible)) {
        vals <- df$theta[ma_ids == eligible[idx]]
        theta_mat[idx, ] <- vals[1:min_outcomes]
    }
    C <- cov(theta_mat, use = "pairwise.complete.obs")
    eigenvals <- eigen(C)$values
    cat(paste("Reviews with >=", min_outcomes, "outcomes:", length(eligible), "\n"))
    cat(paste("Spectral Gap (lambda1 - lambda2):", round(eigenvals[1] - eigenvals[2], 4), "\n"))
    cat(paste("First component variance explained:", round(100 * eigenvals[1] / sum(eigenvals[eigenvals > 0]), 2), "%\n"))
} else {
    cat(paste("Fewer than 5 reviews with >=", min_outcomes, "outcomes. Skipping spectral analysis.\n"))
}

cat("\n===================================================================\n")
cat("  HIERARCHICAL ANALYSIS V2 COMPLETE\n")
cat("===================================================================\n")
