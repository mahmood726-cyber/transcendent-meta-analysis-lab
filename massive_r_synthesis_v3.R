# massive_r_synthesis_v3.R
library(metafor)
library(Matrix)

cat("===================================================================\n")
cat("  HIERARCHICAL META-ANALYSIS V3: REPRESENTATIVE SAMPLING\n")
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

# Take a representative sample of 100 reviews
set.seed(42)
sampled_reviews <- sample(unique(df$review_id), min(100, length(unique(df$review_id))))
df <- df[df$review_id %in% sampled_reviews, ]

cat(paste("Analyzing", nrow(df), "outcomes across", length(unique(df$review_id)), "sampled reviews.\n"))

# 2. Model 1: Multi-Level Random Effects
cat("\n--- MODEL 1: Multi-Level Random Effects ---\n")
res_ml <- rma.mv(theta, sigma^2, random = ~ 1 | review_id / analysis_number, data = df)
print(summary(res_ml))

# 3. Model 2: Meta-Regression (log(k))
cat("\n--- MODEL 2: Meta-Regression (Evidence Volume) ---\n")
df_reg <- df[!is.na(df$k) & df$k > 0, ]
if (nrow(df_reg) > 10) {
    res_reg <- rma.mv(theta, sigma^2, mods = ~ log(k), random = ~ 1 | review_id, data = df_reg)
    print(summary(res_reg))
} else {
    cat("Insufficient rows with valid k > 0 for meta-regression.\n")
}

# 4. Model 3: Variance-Volume Correlation
cat("\n--- MODEL 3: Variance-Volume Correlation ---\n")
ma_var_analysis <- aggregate(theta ~ review_id, data = df, FUN = var)
ma_k_analysis <- aggregate(k ~ review_id, data = df, FUN = sum)
merged_analysis <- merge(ma_var_analysis, ma_k_analysis, by="review_id")
cor_val <- cor(merged_analysis$theta, merged_analysis$k, use="complete.obs")
cat(paste("Correlation between total k and effect variance:", round(cor_val, 4), "\n"))

# 5. Model 4: Extreme Tail Analysis
cat("\n--- MODEL 4: Extreme Tail Analysis (Top 1%) ---\n")
sorted_theta <- sort(df$theta)
tail_1_percent <- head(sorted_theta, max(1, floor(nrow(df)*0.01)))
cat(paste("Mean of top 1% most extreme effects:", round(mean(tail_1_percent), 4), "\n"))
cat(paste("Threshold (1st percentile):", round(max(tail_1_percent), 4), "\n"))
cat("NOTE: Extreme tails expected from many MAs; no multiple testing correction applied.\n")

cat("\n===================================================================\n")
cat("  HIERARCHICAL ANALYSIS V3 COMPLETE\n")
cat("===================================================================\n")
