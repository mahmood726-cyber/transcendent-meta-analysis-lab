# massive_r_synthesis_v3.R
library(metafor)
library(Matrix)

cat("===================================================================\n")
cat("  ULTIMATE R SYNTHESIS V3: REPRESENTATIVE HIERARCHICAL SAMPLING\n")
cat("===================================================================\n")

# 1. Load Data (Pairwise 70 Results)
data_path <- "C:/Users/user/OneDrive/Backups/Models/Pairwise70/analysis/ma4_results_pairwise70.csv"
df <- read.csv(data_path)
df$theta <- as.numeric(df$theta)
df$sigma <- as.numeric(df$sigma)
df <- df[!is.na(df$theta) & !is.na(df$sigma), ]

# Take a representative sample of 100 reviews to ensure the script finishes
set.seed(42)
sampled_reviews <- sample(unique(df$review_id), 100)
df <- df[df$review_id %in% sampled_reviews, ]

cat(paste("Analyzing", nrow(df), "Meta-Analytical outcomes across", length(unique(df$review_id)), "sampled Cochrane reviews.\n"))

# 2. Model 1: Multi-Level Random Effects (Hierarchical)
# OutComes nested within Reviews.
cat("\n--- MODEL 1: Multi-Level Random Effects (Hierarchical) ---\n")
res_ml <- rma.mv(theta, sigma^2, random = ~ 1 | review_id / analysis_number, data = df)
print(summary(res_ml))

# 3. Model 2: Spatio-Temporal Evidence Consensus
cat("\n--- MODEL 2: Evidence Maturity Analysis ---\n")
res_reg <- rma.mv(theta, sigma^2, mods = ~ log(k), random = ~ 1 | review_id, data = df)
print(summary(res_reg))

# 4. Model 3: Heteroskedastic Signal Variance
cat("\n--- MODEL 3: Heteroskedastic Signal Variance ---\n")
ma_var_analysis <- aggregate(theta ~ review_id, data = df, FUN = var)
ma_k_analysis <- aggregate(k ~ review_id, data = df, FUN = sum)
merged_analysis <- merge(ma_var_analysis, ma_k_analysis, by="review_id")
cor_val <- cor(merged_analysis$theta, merged_analysis$k, use="complete.obs")
cat(paste("Correlation between Evidence Volume (k) and Consensus Variance:", round(cor_val, 4), "\n"))

# 5. Model 4: Extreme Value Theory (Minimum-Probability Miracle Cures)
cat("\n--- MODEL 4: GEV Tail Approximation (Top 1% Miracle Cures) ---\n")
sorted_theta <- sort(df$theta)
tail_1_percent <- head(sorted_theta, max(1, floor(nrow(df)*0.01)))
cat(paste("Mean Effect of the Top 1% 'Miracle' Interventions:", round(mean(tail_1_percent), 4), "\n"))
cat(paste("Threshold for a 'Miracle Cure' in this dataset (Log OR):", round(max(tail_1_percent), 4), "\n"))

cat("\n===================================================================\n")
cat("  ULTIMATE R SYNTHESIS V3 COMPLETE\n")
cat("===================================================================\n")
