
## PARTE 1

# Set seed for reproducibility
set.seed(123)

# Parameters
true_mean <- 0       # Population mean (mu)
true_sd <- 1         # Population standard deviation (sigma)
max_n <- 1000        # Maximum sample size
num_sims <- 500      # Number of parallel sample paths (simulations)

# Create a matrix to store sample means for each path up to size n
# Rows represent sample sizes (1 to max_n), columns represent simulation runs
sample_means <- matrix(NA, nrow = max_n, ncol = num_sims)

for (j in 1:num_sims) {
  # Generate a single stream of random normal values
  x <- rnorm(max_n, mean = true_mean, sd = true_sd)
  # Compute cumulative sum divided by index n to get running means
  sample_means[, j] <- cumsum(x) / (1:max_n)
}

# Plotting the convergence in probability (WLLN illustration)
plot(1:max_n, sample_means[, 1], type = "l", col = rgb(0, 0, 1, 0.1),
     ylim = c(-1.5, 1.5), xlab = "Sample Size (n)", ylab = "Sample Mean",
     main = "Weak Law of Large Numbers Convergence")

# Add lines for multiple simulations to show variance shrinking
for (j in 2:num_sims) {
  lines(1:max_n, sample_means[, j], col = rgb(0, 0, 1, 0.1))
}

# Add bounds for a chosen epsilon (e.g., epsilon = 0.1)
epsilon <- 0.1
abline(h = true_mean, col = "red", lwd = 2)
abline(h = true_mean + epsilon, col = "darkgreen", lty = 2, lwd = 1.5)
abline(h = true_mean - epsilon, col = "darkgreen", lty = 2, lwd = 1.5)




## PARTE 2

library(dplyr)
library(tidyr)
library(ggplot2)

# 1. Setup Parameters
set.seed(42)
n_values <- seq(10, 2000, by = 50)  # Sequence of increasing sample sizes
n_simulations <- 1000               # Number of simulated universes per n
true_mean <- 0                      # Standard Normal distribution mean (μ = 0)
epsilon <- 0.05                     # Error margin (tolerance)

# 2. Simulate Sample Means and Calculate Empirical Probability
prob_results <- tibble(n = n_values) %>%
  rowwise() %>%
  mutate(
    # Generate a vector of 'n_simulations' sample means for the current 'n'
    sample_means = list(replicate(n_simulations, mean(rnorm(n, mean = true_mean, sd = 1)))),
    
    # Calculate the fraction of sample means where |Sample Mean - μ| > epsilon
    prob_outside = mean(abs(unlist(sample_means) - true_mean) > epsilon)
  ) %>%
  ungroup()

# View the explicit probability values
print(head(prob_results, 10))

# 3. Plot the Explicit Convergence
ggplot(prob_results, aes(x = n, y = prob_outside)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "red", size = 2) +
  labs(
    title = "Empirical Proof of the Weak Law of Large Numbers",
    subtitle = paste0("Probability that |Sample Mean - μ| > ", epsilon, " as 'n' increases"),
    x = "Sample Size (n)",
    y = "P(|Sample Mean - μ| > ε)"
  ) +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA)) +
  theme_minimal()




## PARTE 3
## -- refazer com a t-Student (com nu = 2) [!]

library(dplyr)
library(tidyr)
library(ggplot2)

# 1. Setup Parameters
set.seed(7)
n_values <- seq(20, 3000, by = 40)   # Sample sizes to test
n_simulations <- 1000                # Simulations per sample size
epsilon <- 0.05                      # Error tolerance
true_mean <- 0                       # Standard distribution mean

# Variances to compare
sd_low  <- 1                         # Variance = 1
sd_high <- 2                         # Variance = 4

# 2. Simulation Function
run_wlln_sim <- function(n, sd_val) {
  # Generate sample means
  means <- replicate(n_simulations, mean(rnorm(n, mean = true_mean, sd = sd_val)))
  # Calculate empirical probability of exceeding epsilon
  mean(abs(means - true_mean) > epsilon)
}

# 3. Build Data Frame with Empirical and Theoretical Values
df_sim <- expand.grid(n = n_values, Variance = c("Low (σ²=1)", "High (σ²=4)")) %>%
  rowwise() %>%
  mutate(
    # Assign correct standard deviation based on the group
    sd_current = if_else(Variance == "Low (σ²=1)", sd_low, sd_high),
    
    # Compute Empirical Probability
    Empirical = run_wlln_sim(n, sd_current),
    
    # Compute Chebyshev Theoretical Upper Bound: σ² / (n * ε²)
    # We cap the bound at 1.0 since probabilities cannot exceed 1
    Chebyshev_Bound = min(1, (sd_current^2) / (n * (epsilon^2)))
  ) %>%
  ungroup()

# 4. Reshape for ggplot (combining Empirical and Theoretical lines)
df_plot <- df_sim %>%
  pivot_longer(cols = c(Empirical, Chebyshev_Bound), 
               names_to = "Metric", 
               values_to = "Probability")

# 5. Plot the Comparison
ggplot(df_plot, aes(x = n, y = Probability, color = Variance, linetype = Metric)) +
  geom_line(linewidth = 1) +
  scale_linetype_manual(values = c("Empirical" = "solid", "Chebyshev_Bound" = "dashed"),
                        labels = c("Empirical Simulation", "Chebyshev Upper Bound")) +
  scale_color_manual(values = c("Low (σ²=1)" = "#2c7bb6", "High (σ²=4)" = "#d7191c")) +
  labs(
    title = "WLLN Convergence Rate: Variance vs. Chebyshev Bound",
    subtitle = paste0("Error margin (ε) = ", epsilon, " | 1,000 simulations per step"),
    x = "Sample Size (n)",
    y = "Probability of Deviation P(|X̄ - μ| > ε)",
    color = "Distribution Variance",
    linetype = "Metric Type"
  ) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.05)) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.box = "vertical")



library(dplyr)
library(tidyr)
library(ggplot2)

# 1. Setup Parameters
set.seed(42)
n_values <- seq(20, 2000, by = 40)   # Sample sizes to test
n_simulations <- 1000                # Simulations per sample size
epsilon <- 0.05                      # Error tolerance

# 2. Simulation Function for Cauchy
# Note: rcauchy(n, location=0, scale=1) has a theoretical median/location of 0
run_cauchy_sim <- function(n) {
  means <- replicate(n_simulations, mean(rcauchy(n, location = 0, scale = 1)))
  mean(abs(means - 0) > epsilon)
}

# 3. Build Data Frame 
# We calculate the empirical failure rate and try to map a "fake" Chebyshev bound
df_cauchy <- tibble(n = n_values) %>%
  rowwise() %>%
  mutate(
    Empirical_Cauchy = run_cauchy_sim(n),
    # Chebyshev requires a finite variance (sigma^2). 
    # If we plugged in infinity, the bound would just be infinity (or capped at 100%)
    Chebyshev_Bound = 1.0 
  ) %>%
  ungroup()

# 4. Plot the Result
ggplot(df_cauchy, aes(x = n)) +
  geom_line(aes(y = Empirical_Cauchy, color = "Cauchy Empirical"), linewidth = 1.2) +
  geom_line(aes(y = Chebyshev_Bound, color = "Chebyshev Upper Bound"), linetype = "dashed", linewidth = 1) +
  scale_color_manual(values = c("Cauchy Empirical" = "#e41a1c", "Chebyshev Upper Bound" = "black")) +
  labs(
    title = "WLLN Failure: The Cauchy Distribution",
    subtitle = paste0("Error margin (ε) = ", epsilon, " | Heavy tails prevent convergence"),
    x = "Sample Size (n)",
    y = "Probability of Deviation P(|X̄ - 0| > ε)",
    color = "Legend"
  ) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.05)) +
  theme_minimal() +
  theme(legend.position = "bottom")


