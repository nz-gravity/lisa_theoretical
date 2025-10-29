import numpy as np
import matplotlib.pyplot as plt
import h5py, os, requests
import time  # added to track runtimes

C = 299792458
TEN_DAYS = "https://raw.githubusercontent.com/nz-gravity/test_data/main/lisa_noise/noise_4a_truncated/data/tdi.h5"
fname = "data/tdi.h5"

start_total = time.perf_counter()  # start total runtime

# --- Download if missing ---
start_download = time.perf_counter()
if not os.path.exists(fname):
    os.makedirs("data", exist_ok=True)
    print("Downloading data...")
    r = requests.get(TEN_DAYS, stream=True)
    if not r.ok: raise RuntimeError(f"Download failed ({r.status_code})")
    with open(fname, "wb") as f:
        for chunk in r.iter_content(8192): f.write(chunk)
    print("Download complete.")
    download_time = time.perf_counter() - start_download
else:
    download_time = 0.0
    print("Data file exists; download skipped.")
print(f"Download time: {download_time:.3f} s")

# --- Read data ---
start_processing = time.perf_counter()
with h5py.File(fname, "r") as f:
    t  = np.array(f["t"])
    x2 = np.array(f["X2"])

dt = t[1] - t[0]
fs = 1 / dt
n  = len(t)

T  = n * dt
fmin = 1 / T
freq = np.fft.rfftfreq(n, dt)

print(f"Samples: {n}, fs: {fs:.3f} Hz, duration: {T/86400:.2f} days")

# --- Noise + transfer models ---
L = 2.5e9 / C
exp_term = np.exp(-2*np.pi*fmin/fs) * np.exp(-2j*np.pi*freq/fs)
denom_mag2 = np.abs(1 - exp_term)**2
common = 16 * np.sin(2*np.pi*freq*L)**2 * np.sin(4*np.pi*freq*L)**2
tf_tm  = np.sqrt(common * (3 + np.cos(4*np.pi*freq*L)))
tf_oms = np.sqrt(4 * common)

tm_asd, tm_fknee = 2.4e-15, 4e-4
oms_asd, oms_fknee = 7.9e-12, 2e-3
fc = 2.81e14

# --- Test mass ASD ---
psd_tm_high = ((2*tm_asd*fc/(2*np.pi*C))**2 *
               (2*np.pi*fmin)**2 / denom_mag2 / (fs*fmin)**2)
psd_tm_low  = ((2*tm_asd*fc*tm_fknee/(2*np.pi*C))**2 *
               (2*np.pi*fmin)**2 / denom_mag2 / (fs*fmin)**2 *
               np.abs(1 / (1 - np.exp(-2j*np.pi*freq/fs)))**2 *
               (2*np.pi/fs)**2)
tm_asd_val = np.sqrt(psd_tm_high + psd_tm_low)

# --- OMS ASD ---
psd_oms_high = (oms_asd*fs*fc/C)**2 * np.sin(2*np.pi*freq/fs)**2
psd_oms_low  = ((2*np.pi*oms_asd*fc*oms_fknee**2/C)**2 *
                (2*np.pi*fmin)**2 / denom_mag2 / (fs*fmin)**2)
oms_asd_val = np.sqrt(psd_oms_high + psd_oms_low)

# --- Total model ---
theoretical_psd = ((tf_tm*tm_asd_val)**2 + (tf_oms*oms_asd_val)**2)

# --- Periodogram ---
psd_x = np.abs(np.fft.rfft(x2))**2 * dt / n
psd_x[1:-1] *= 2  # match numpy rfft normalization
freq, psd_x, theoretical_psd = freq[1:], psd_x[1:], theoretical_psd[1:]

# --- Save + plot ---
np.savetxt("python_psd.txt", np.column_stack([freq, psd_x, theoretical_psd]),
           header="freq_Hz psd_X2 psd_model", fmt="%.6e")

plt.figure(figsize=(6,4))
plt.loglog(freq, psd_x, color="gray", lw=1.5, label="X2 periodogram")
plt.loglog(freq, theoretical_psd, "k-", lw=1.2, label="Total model")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [1/Hz]")
plt.legend()
plt.tight_layout()
plt.savefig("python_psd.png", dpi=200)
print("Done. Saved python_psd.txt and python_psd.png")

processing_time = time.perf_counter() - start_processing
total_time = time.perf_counter() - start_total
print(f"Processing time: {processing_time:.3f} s")
print(f"Total runtime:    {total_time:.3f} s")
