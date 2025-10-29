library(ggplot2)
library(scales)

load("data_10day.RData") # original 10-day data
# load("out_mcmc_1day_vnp.RData") # VNP results
mpg_X_py <- read.csv("periodogram_X_py.csv") # periodogram for X saved from the python results
model_total <- read.csv("1day_quarterHZ_total_model.csv") # total model fit from the python results



# Extract 1-day data -> truncate it to be sampling frequency = .25HZ
data_1day <- data_10day[seq(1,43201,by=2),]


# Here is how I got the results from Python
lisa_data = LISAData.from_hdf5("tdi.h5")[1:43201:2]


# frequencies
index <- seq(0, .25/2, len=10801)


# # store vnp results for ggplot
# vnp_1day <- data.frame(
#   f11 = out_mcmc_vnp$psd.median[1,1,-1]*var(data_1day[,1])*pi/.25, # VNP was fitted to the standardized data,
#                                                                    # so here rescales the results to fit the original data
#   f22 = out_mcmc_vnp$psd.median[2,2,-1],
#   f33 = out_mcmc_vnp$psd.median[3,3,-1],
#   f12 = out_mcmc_vnp$psd.median[1,2,-1],
#   f13 = out_mcmc_vnp$psd.median[1,3,-1],
#   f23 = out_mcmc_vnp$psd.median[2,3,-1],
#   f21 = out_mcmc_vnp$psd.median[2,1,-1],
#   f31 = out_mcmc_vnp$psd.median[3,1,-1],
#   f32 = out_mcmc_vnp$psd.median[3,2,-1],
#   freq = index[-1]
# )


# store periodogram for ggplot
mpg_1day_X <- data.frame(
  f = mpg_X_py[-1,1],
  freq = index[-1]
)

# store total model results for ggplot
total_modal_1day <- data.frame(
  f = model_total[,1],
  freq = index[-1]
)


# plot for the single X channel
plot_1day_11 <- ggplot(mpg_1day_X) +
  geom_line(aes(x = freq, y = f)) +
  geom_line(data = vnp_1day, aes(x = freq, y = Re(f11), color = 'vnp'), linewidth = 1) +
  geom_line(data = total_modal_1day, aes(x = freq, y = f, color = 'total model'), linewidth = 1) +
  scale_x_log10(labels = trans_format("log10", math_format(10^.x))) +
  scale_y_log10(labels = trans_format("log10", math_format(10^.x))) +
  scale_color_manual(values = c("vnp" = "red3", "total model" = "cyan3")) +
  labs(x = NULL, y = "PSD", title = expression(S[X2])) +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 20),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 20),
        legend.position = 'none')










