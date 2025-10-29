library(ggplot2)
library(patchwork)  # install if missing: install.packages("patchwork")

# --- Load R output ---
r_data <- read.table("r_psd.txt", header = TRUE)
cat("Loaded", nrow(r_data), "rows from r_psd.txt\n")

# --- Load Python output ---
py_data <- read.table(
  "python_psd.txt",
  header = FALSE,
  comment.char = "#",
  skip = 1,
  col.names = c("freq_Hz", "psd_X2", "psd_model")
)
cat("Loaded", nrow(py_data), "rows from python_psd.txt\n")

# Ensure numeric and align by overlapping frequencies
r_data[] <- lapply(r_data, as.numeric)
py_data[] <- lapply(py_data, as.numeric)

# Trim to common frequency range (in case one file is shorter)
n <- min(nrow(r_data), nrow(py_data))
r_data <- r_data[1:n, ]
py_data <- py_data[1:n, ]

# Compute errors
err_periodogram_abs <- abs(r_data$psd_X2 - py_data$psd_X2)
err_model_abs       <- abs(r_data$psd_model - py_data$psd_model)
err_periodogram_rel <- err_periodogram_abs / py_data$psd_X2
err_model_rel       <- err_model_abs / py_data$psd_model

# --- Plot 1: PSD overlay ---
p1 <- ggplot() +
  geom_line(data = r_data, aes(x = freq_Hz, y = psd_X2, color = "Data (R)"), linewidth = 0.9) +
  geom_line(data = r_data, aes(x = freq_Hz, y = psd_model, color = "Model (R)"), linewidth = 0.9) +
  geom_line(data = py_data, aes(x = freq_Hz, y = psd_X2, color = "Data (Python)"), linewidth = 0.9, linetype = "dashed") +
  geom_line(data = py_data, aes(x = freq_Hz, y = psd_model, color = "Model (Python)"), linewidth = 0.9, linetype = "dashed") +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_manual(values = c(
    "Data (R)" = "#1f78b4",
    "Data (Python)" = "#33a02c",
    "Model (R)" = "#e31a1c",
    "Model (Python)" = "#ff7f00"
  )) +
  labs(x = NULL, y = "PSD [1/Hz]", title = "R vs Python PSD Comparison") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom", legend.title = element_blank())

# --- Plot 2: Error subplot ---
error_df <- data.frame(
  freq_Hz = r_data$freq_Hz,
  abs_error_data = err_periodogram_abs,
  abs_error_model = err_model_abs,
  rel_error_data = err_periodogram_rel,
  rel_error_model = err_model_rel
)

p2 <- ggplot(error_df) +
  geom_line(aes(x = freq_Hz, y = rel_error_data, color = "Data (rel error)"), linewidth = 0.8) +
  geom_line(aes(x = freq_Hz, y = rel_error_model, color = "Model (rel error)"), linewidth = 0.8, linetype = "dashed") +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_manual(values = c(
    "Data (rel error)" = "#377eb8",
    "Model (rel error)" = "#e41a1c"
  )) +
  labs(x = "Frequency [Hz]", y = "Relative Error", title = "R vs Python Relative Errors") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom", legend.title = element_blank())

# Combine vertically
combined_plot <- p1 / p2 + plot_layout(heights = c(2, 1))

ggsave("comparison_psd_with_errors.png", plot = combined_plot, width = 8, height = 7, dpi = 300)
cat("Done. Saved comparison_psd_with_errors.png\n")
