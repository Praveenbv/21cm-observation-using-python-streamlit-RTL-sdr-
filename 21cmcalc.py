import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from numpy.fft import fft, fftshift
from datetime import datetime
import time
import csv
import os

# Constants from paper (exact values)
R0 = 8.5            # kpc (Sun to Galactic Center)
V0 = 220            # km/s (Solar orbital velocity)
F0 = 1420.405751    # MHz (21cm rest frequency)
C = 299792.458      # km/s (speed of light)
A = 14.8            # km/s/kpc (Oort A)
B = A - (V0 / R0)   # Oort B

def capture_rtl_data(center_freq, num_samples):
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.center_freq = center_freq
    sdr.gain = 'auto'
    samples = sdr.read_samples(num_samples)
    sdr.close()
    return samples

def return_averaged_spectras(ipdata, chNo, nAvgerages, nSets, npt):
    ipdata = np.asarray(ipdata)
    row = len(ipdata)
    totalSpectrasAsked = nAvgerages * nSets
    NoOfAvailableSpectras = (row // npt) - 1

    if totalSpectrasAsked <= NoOfAvailableSpectras:
        aspecA = np.zeros((npt, nSets))
        for set_idx in range(nSets):
            for I in range(nAvgerages):
                startNo = (I * npt) + (set_idx * nAvgerages * npt) + 1
                endNo = startNo + npt
                if endNo > len(ipdata): break
                segment = ipdata[startNo:endNo] if ipdata.ndim == 1 else ipdata[startNo:endNo, chNo]
                spc = np.abs(fft(segment)) ** 2
                aspecA[:, set_idx] += spc[:npt]
            aspecA[:, set_idx] = fftshift(aspecA[:, set_idx] / nAvgerages)
        return aspecA
    else:
        st.error("erorr")
        return np.zeros((2, 1))

def process_data(aa, bb, colNo=1):
    nfft = 128
    navg = len(aa) // nfft - 3
    nsets = 1
    avgps = return_averaged_spectras(aa, colNo, navg, nsets, nfft)
    avgps2 = return_averaged_spectras(bb, colNo, navg, nsets, nfft)
    return avgps, avgps2

def calculate_velocity_and_distance(rest_freq_mhz, L):
    #radians
    L_rad = np.radians(L)
    
    # Doppler formula from paper (MHz units)
    Vr = C * (F0 - rest_freq_mhz) / F0
    
    # Distance calculation
    denominator = A * np.sin(2 * L_rad)
    d = Vr / denominator if abs(denominator) > 1e-6 else np.nan
    
    # Velocity components
    Vt = d * (A * np.cos(2 * L_rad) + B)
    Ur = Vr + V0 * np.sin(L_rad)
    Ut = Vt + V0 * np.cos(L_rad)
    V = np.sqrt(Ur**2 + Ut**2)
    
    # Galactic center distance
    R = np.sqrt(R0**2 + d**2 - 2*R0*d*np.cos(L_rad))
    
    return {
        'Vr': round(Vr, 1),
        'd': round(d, 1),
        'R': round(R, 1),
        'Ur': round(Ur, 1),
        'Ut': round(Ut, 1),
        'V': round(V, 1)
    }

def create_plots(avgps, avgps2, num, L):
    nfft = 128
    Fcenter = 1420.405751  # MHz
    freq_hz = np.linspace(Fcenter*1e6 - 1e6, Fcenter*1e6 + 1e6, nfft)
    
    # Find peak frequency (convert to MHz)
    diff_spec = avgps2 - avgps
    peak = np.argmax(diff_spec)
    rest_freq_mhz = freq_hz[peak] / 1e6  # Convert Hz to MHz

    results = calculate_velocity_and_distance(rest_freq_mhz, L)

    # Create plot with grids
    plt.figure(figsize=(10, 10))
    
    # First subplot - Spectra comparison
    plt.subplot(3, 1, 1)
    plt.plot(freq_hz/1e6, avgps, 'b-', label='1423 MHz')
    plt.plot(freq_hz/1e6, avgps2, 'r--', label='1420 MHz')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power in counts')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Second subplot - Difference spectrum
    plt.subplot(3, 1, 2)
    plt.plot(freq_hz/1e6, diff_spec, 'g-')
    plt.axvline(Fcenter, color='k', linestyle='--')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Signal Power in counts')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Third subplot - Results text
    plt.subplot(3, 1, 3)
    plt.axis('off')
    result_text = (
        f"Galactic Longitude (l): {L}°\n"
        f"Observed Frequency: {rest_freq_mhz:.6f} MHz\n"
        f"Vr: {results['Vr']} km/s\n"
        f"Ur: {results['Ur']} km/s\n"
        f"Ut: {results['Ut']} km/s\n"
        f"Total V: {results['V']} km/s\n"
        f"Distance (d): {results['d']} kpc\n"
        f"Galactic center Distance R: {results['R']} kpc"
    )
    plt.text(0.1, 0.5, result_text, fontfamily='monospace')
    
    plt.tight_layout()
    filename = f'plot_trial_{num}.png'
    plt.savefig(filename, dpi=100)
    plt.close()
    
    return filename, freq_hz, results, rest_freq_mhz

def save_data(avgps, avgps2, freq, results, trial_num, L, rest_freq):
    """Save full observation data to CSV including spectra"""
    os.makedirs("data", exist_ok=True)
    filename = f"data/trial_{trial_num}_results.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata header
        writer.writerow(["Parameter", "Value", "Unit"])
        writer.writerow(["Trial Number", trial_num, ""])
        writer.writerow(["Galactic Longitude (L)", L, "degrees"])
        writer.writerow(["Rest Frequency", rest_freq, "MHz"])
        writer.writerow(["Observation Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""])
        
        # Write calculated results
        writer.writerow([])
        writer.writerow(["Calculated Parameters"])
        writer.writerow(["Vr", results['Vr'], "km/s"])
        writer.writerow(["Ur", results['Ur'], "km/s"])
        writer.writerow(["Ut", results['Ut'], "km/s"])
        writer.writerow(["Total Velocity", results['V'], "km/s"])
        writer.writerow(["Distance from Sun", results['d'], "kpc"])
        writer.writerow(["Galactic center Distance", results['R'], "kpc"])
        
        # Write spectral data
        writer.writerow([])
        writer.writerow(["Spectral Data"])
        writer.writerow(["Frequency (MHz)", "avgps", "avgps2"])
        for f, on, off in zip(freq/1e6, avgps.flatten(), avgps2.flatten()):
            writer.writerow([round(f, 4), round(on, 2), round(off, 2)])

# Streamlit UI
st.title('RTL-SDR 21cm Hydrogen Line Observation')
plot_spot = st.empty()
num_samples = st.sidebar.number_input('Samples per Capture', 102400, 20480000, 2048000)
num_trials = st.sidebar.number_input('Number of Trials', 1, 2000, 1)
L = st.sidebar.number_input('Galactic Longitude (L)', 0, 90, 35)

if st.button('Start Observation Session'):
    status_text = st.empty()
    for trial in range(1, num_trials + 1):
        try:
            status_text.text(f"Running Trial {trial}/{num_trials}...")
            # Persistent display areas
            #plot_spot = st.empty()
            # Capture data
            aa = capture_rtl_data(1422000000, num_samples)  # 1422 MHz (ON)
            bb = capture_rtl_data(1420405751, num_samples)  # 1420 MHz (OFF)
            
            # Process data
            avgps, avgps2 = process_data(aa, bb)
            
            # Create plots and get analysis results
            plot_filename, freq_hz, results, rest_freq = create_plots(avgps, avgps2, trial, L)
            
            # Save data to CSV
            save_data(avgps, avgps2, freq_hz, results, trial, L, rest_freq)
            
            # Generate and display plot
            plot_file, freq_hz, results, rest_freq = create_plots(avgps, avgps2, trial, L)
            save_data(avgps, avgps2, freq_hz, results, trial, L, rest_freq)
            
            # Update display with current plot
            with plot_spot:
                st.image(plot_file, caption=f'Trial {trial} Results')
            
            st.session_state.trial_num = trial

        except Exception as e:
            st.error(f"Error in trial {trial}: {str(e)}")
            break

    st.success("Observation session completed successfully!")
