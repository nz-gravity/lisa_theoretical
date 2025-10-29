# Required packages
library(rhdf5)
library(ggplot2)

cat("Loading constants...\n")

c_light <- 299792458
fname <- "data/tdi.h5"
TEN_DAYS <- "https://raw.githubusercontent.com/nz-gravity/test_data/main/lisa_noise/noise_4a_truncated/data/tdi.h5"

start_total <- Sys.time()  # start total runtime

if (!file.exists(fname)) {
  dir.create("data", showWarnings = FALSE)
  cat("Downloading data...\n")
  start_download <- Sys.time()
  download.file(TEN_DAYS, destfile = fname, mode = "wb", quiet = TRUE)
  download_time <- as.numeric(difftime(Sys.time(), start_download, units = "secs"))
  cat(sprintf("Download complete. Download time: %.3f s\n", download_time))
} else {
  download_time <- 0.0
  cat("Data file exists; download skipped.\n")
  cat(sprintf("Download time: %.3f s\n", download_time))
}

start_processing <- Sys.time()
cat("Reading HDF5 file...\n")
t <- h5read(fname, "t")
x2 <- h5read(fname, "X2")


dt <- t[2] - t[1]
fs <- 1 / dt
n <- length(t)

Tdur <- n * dt
fmin <- 1 / Tdur
freq <- seq(0, fs / 2, length.out = floor(n / 2) + 1)[-1]  # drop DC

cat(sprintf("Samples: %d, fs = %.3f Hz, duration = %.2f days\n", n, fs, Tdur / 86400))

L <- 2.5e9 / c_light
exp_term <- exp(-2 * pi * fmin / fs) * exp(-2i * pi * freq / fs)
denom_mag2 <- abs(1 - exp_term)^2
common <- 16 * sin(2 * pi * freq * L)^2 * sin(4 * pi * freq * L)^2
tf_tm <- sqrt(common * (3 + cos(4 * pi * freq * L)))
tf_oms <- sqrt(4 * common)

tm_asd <- 2.4e-15
tm_fknee <- 4e-4
oms_asd <- 7.9e-12
oms_fknee <- 2e-3
fc <- 2.81e14

cat("Computing theoretical model...\n")

# Test-mass ASD
psd_tm_high <- ((2 * tm_asd * fc / (2 * pi * c_light))^2 *
  (2 * pi * fmin)^2 / denom_mag2 / (fs * fmin)^2)

psd_tm_low <- ((2 * tm_asd * fc * tm_fknee / (2 * pi * c_light))^2 *
  (2 * pi * fmin)^2 / denom_mag2 / (fs * fmin)^2 *
  abs(1 / (1 - exp(-2i * pi * freq / fs)))^2 *
  (2 * pi / fs)^2)

tm_asd_val <- sqrt(psd_tm_high + psd_tm_low)

# OMS ASD
psd_oms_high <- (oms_asd * fs * fc / c_light)^2 * sin(2 * pi * freq / fs)^2
psd_oms_low <- ((2 * pi * oms_asd * fc * oms_fknee^2 / c_light)^2 *
  (2 * pi * fmin)^2 / denom_mag2 / (fs * fmin)^2)
oms_asd_val <- sqrt(psd_oms_high + psd_oms_low)

# Total model
theoretical_psd <- (tf_tm * tm_asd_val)^2 + (tf_oms * oms_asd_val)^2

cat("Computing periodogram...\n")

fft_x2 <- fft(x2)
psd_x <- (Mod(fft_x2[1:length(freq)])^2) * dt / n
psd_x[2:(length(psd_x) - 1)] <- 2 * psd_x[2:(length(psd_x) - 1)]  # match Python's rfft scaling

cat("Saving results...\n")

data_out <- data.frame(freq_Hz = freq,
                       psd_X2 = Re(psd_x),
                       psd_model = Re(theoretical_psd))
write.table(data_out, "r_psd.txt", row.names = FALSE, col.names = TRUE)

cat("Plotting...\n")

df <- data.frame(freq = freq,
                 psd_data = Re(psd_x),
                 psd_model = Re(theoretical_psd))

p <- ggplot(df, aes(x = freq)) +
  geom_line(aes(y = psd_data), color = "gray50", linewidth = 0.8) +
  geom_line(aes(y = psd_model), color = "black", linewidth = 0.7) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Frequency [Hz]", y = "PSD [1/Hz]") +
  theme_minimal(base_size = 12)

ggsave("r_psd.png", p, width = 6, height = 4, dpi = 200)
cat("Done. Saved r_psd.txt and r_psd.png\n")

processing_time <- as.numeric(difftime(Sys.time(), start_processing, units = "secs"))
total_time <- as.numeric(difftime(Sys.time(), start_total, units = "secs"))
cat(sprintf("Processing time: %.3f s\n", processing_time))
cat(sprintf("Total runtime:    %.3f s\n", total_time))
