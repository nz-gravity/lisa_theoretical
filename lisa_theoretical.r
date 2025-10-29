library(Rcpp)
library(rhdf5)
library(ggplot2)

cppFunction('
NumericVector total_model_cpp(NumericVector freq,
                              int n, double dt, double fs,
                              double tm_asd=2.4e-15,
                              double tm_fknee=4e-4,
                              double oms_asd=7.9e-12,
                              double oms_fknee=2e-3,
                              double central_freq=2.81e14,
                              double armlength=2.5e9) {

  const double c = 299792458.0;
  double fmin = 1.0 / (n * dt);
  double L = armlength / c;

  int N = freq.size();
  NumericVector out(N);

  for(int i=0; i<N; i++){
    double f = freq[i];
    double theta_f = 2*M_PI*f/fs;
    double exp_real = cos(theta_f);
    double exp_imag = -sin(theta_f);
    double exp_const = exp(-2*M_PI*fmin/fs);

    double denom_r = 1 - exp_const*exp_real;
    double denom_i = -exp_const*exp_imag;
    double denom_mag2 = denom_r*denom_r + denom_i*denom_i;

    double tf_common = 16*pow(sin(2*M_PI*f*L),2)*pow(sin(4*M_PI*f*L),2);
    double tf_tm = sqrt(tf_common*(3 + cos(4*M_PI*f*L)));
    double tf_oms = sqrt(4*tf_common);

    double psd_tm_high = pow((2*tm_asd*central_freq/(2*M_PI*c)),2) *
                         pow((2*M_PI*fmin),2)/denom_mag2 /
                         pow(fs*fmin,2);

    double psd_tm_low  = pow((2*tm_asd*central_freq*tm_fknee/(2*M_PI*c)),2) *
                         pow((2*M_PI*fmin),2)/denom_mag2 /
                         pow(fs*fmin,2) *
                         pow(2*M_PI/fs,2);

    double tm_asd_val = sqrt(psd_tm_high + psd_tm_low);

    double psd_oms_high = pow(oms_asd*fs*central_freq/c,2) *
                          pow(sin(2*M_PI*f/fs),2);

    double psd_oms_low  = pow((2*M_PI*oms_asd*central_freq*pow(oms_fknee,2)/c),2) *
                          pow((2*M_PI*fmin),2)/denom_mag2 /
                          pow(fs*fmin,2);

    double oms_asd_val = sqrt(psd_oms_high + psd_oms_low);

    out[i] = sqrt(pow(tf_tm*tm_asd_val,2) + pow(tf_oms*oms_asd_val,2));
  }

  return out;
}
')

fftfreq <- function(n, d = 1) {
  val <- 1.0 / (n * d)
  c(0:floor(n/2)) * val
}

rfft <- function(x) fft(x)[1:(floor(length(x)/2)+1)]

fname <- "data/tdi.h5"
t  <- h5read(fname, "t")
x2 <- h5read(fname, "X2")

dt <- t[2] - t[1]
fs <- 1 / dt
n  <- length(t)
freq <- fftfreq(n, dt)

# compute PSDs
theoretical_asd <- total_model_cpp(freq, n, dt, fs)
psd_x <- (Mod(rfft(x2))^2) * dt / n
psd_x <- psd_x[-1]
freq  <- freq[-1]
theoretical_psd <- theoretical_asd[-1]^2

# save + plot
out <- data.frame(freq_Hz=freq, psd_X2=psd_x, psd_model=theoretical_psd)
write.table(out, "r_psd.txt", row.names=FALSE, col.names=TRUE)

ggplot(out, aes(freq_Hz)) +
  geom_line(aes(y=psd_X2), colour="grey50") +
  geom_line(aes(y=psd_model), colour="black") +
  scale_x_log10() + scale_y_log10() +
  labs(x="Frequency [Hz]", y="PSD [1/Hz]") +
  theme_minimal(base_size=12)
ggsave("r_psd.png", width=6, height=4, dpi=200)
