# =========================================================
# COMPARATIVE ANALYSIS OF FIXED POWER CONTROL IN 5G NOMA-OFDM SYSTEMS
#
# Author: Gemini (AI Research Engineer)
# Date: October 11, 2025
#
# Description:
# This project simulates a 2-user NOMA-OFDM system with user movement.
# It compares the performance (Throughput, BER, Fairness) of three
# different fixed power allocation strategies on a dashboard that updates
# step-by-step via a button click.
#
# Version: 5.1 (Layout and Button Fix)
# =========================================================

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import deque
from matplotlib.widgets import Button # Import the Button widget

# For reproducibility
np.random.seed(42)

# =========================================================
# MODULE 1: SYSTEM & PHYSICS SETUP
# =========================================================
print("Initializing System Parameters...")

PARAMS = {
    'FFT_SIZE': 64, 'CP_LEN': 16, 'NUM_SYMBOLS': 10,
    'MOD_ORDER': 16, 'BITS_PER_SYMBOL': 4,
    'NUM_USERS': 2, 'TOTAL_POWER': 1.0,
    'NUM_BITS': 64 * 10 * 4,
}

PHYSICS_PARAMS = {
    'BASE_STATION_POS': np.array([0, 0]),
    'FREQ_GHz': 2.4,
    'NOISE_POWER_dBm': -90,
}
NOISE_POWER_W = 10**((PHYSICS_PARAMS['NOISE_POWER_dBm'] - 30) / 10)

QAM_MAP = {
    '0000': (-3+3j), '0001': (-3+1j), '0010': (-3-3j), '0011': (-3-1j),
    '0100': (-1+3j), '0101': (-1+1j), '0110': (-1-3j), '0111': (-1-1j),
    '1000': (3+3j),  '1001': (3+1j),  '1010': (3-3j),  '1011': (3-1j),
    '1100': (1+3j),  '1101': (1+1j),  '1110': (1-3j),  '1111': (1-1j)
}
UNNORMALIZED_CONSTELLATION = np.array(list(QAM_MAP.values()))
DEMAP_TABLE = {v: k for k, v in QAM_MAP.items()}
CONSTELLATION = UNNORMALIZED_CONSTELLATION / np.sqrt(10)

# --- Helper Functions (unchanged) ---
def generate_bits(num_bits): return np.random.randint(0, 2, num_bits)
def bits_to_qam(bits):
    symbols = []
    for i in range(0, len(bits), PARAMS['BITS_PER_SYMBOL']):
        b_str = ''.join(map(str, bits[i:i+PARAMS['BITS_PER_SYMBOL']]))
        symbols.append(QAM_MAP[b_str])
    return np.array(symbols)
def ofdm_modulate(qam_symbols):
    symbols_matrix = qam_symbols.reshape((PARAMS['NUM_SYMBOLS'], PARAMS['FFT_SIZE']))
    time_domain_signal = np.fft.ifft(symbols_matrix, axis=1, norm='ortho')
    cp = time_domain_signal[:, -PARAMS['CP_LEN']:]
    return np.concatenate((cp, time_domain_signal), axis=1)
def ofdm_demodulate(received_signal):
    num_symbols = received_signal.shape[0] // (PARAMS['FFT_SIZE'] + PARAMS['CP_LEN'])
    signal_matrix = received_signal.reshape((num_symbols, PARAMS['FFT_SIZE'] + PARAMS['CP_LEN']))
    signal_no_cp = signal_matrix[:, PARAMS['CP_LEN']:]
    return np.fft.fft(signal_no_cp, axis=1, norm='ortho')
def qam_to_bits(received_qam):
    bits = []
    for sym in received_qam:
        idx = np.argmin(np.abs(CONSTELLATION - sym))
        key_sym = UNNORMALIZED_CONSTELLATION[idx]
        bit_str = DEMAP_TABLE[key_sym]
        bits.extend([int(b) for b in bit_str])
    return np.array(bits)

# =========================================================
# MODULE 2: CHANNEL & RECEIVER (unchanged)
# =========================================================
def get_channel_gain(distance):
    if distance < 1: distance = 1
    lambda_ = 3e8 / (PHYSICS_PARAMS['FREQ_GHz'] * 1e9)
    path_loss_db = 20 * np.log10(4 * math.pi * distance / lambda_) + (10 * 2.8 * np.log10(distance/1))
    path_loss_linear = 10**(-path_loss_db / 10)
    rayleigh_fading = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return np.sqrt(path_loss_linear) * rayleigh_fading

def qam_to_bits_symbols(received_qam):
    symbols = []
    for sym in received_qam:
        idx = np.argmin(np.abs(CONSTELLATION - sym))
        symbols.append(CONSTELLATION[idx])
    return np.array(symbols)

def perform_sic(y_freq, h_strong, h_weak, alpha):
    s1_hat_noisy = y_freq / (h_strong * np.sqrt(alpha * PARAMS['TOTAL_POWER']))
    s1_hat_symbols = qam_to_bits_symbols(s1_hat_noisy.flatten())
    bits1_decoded = qam_to_bits(s1_hat_noisy.flatten())
    interference = h_weak * np.sqrt(alpha * PARAMS['TOTAL_POWER']) * s1_hat_symbols.reshape(y_freq.shape)
    y_sic = y_freq - interference
    s2_hat_noisy = y_sic / (h_weak * np.sqrt((1 - alpha) * PARAMS['TOTAL_POWER']))
    bits2_decoded = qam_to_bits(s2_hat_noisy.flatten())
    return bits1_decoded, bits2_decoded

# =========================================================
# MODULE 3: PERFORMANCE METRICS (unchanged)
# =========================================================
def calculate_ber(bits_tx, bits_rx): return np.sum(bits_tx != bits_rx) / len(bits_tx)
def calculate_throughput(ber, num_bits): return num_bits * (1 - ber)
def calculate_jains_fairness(throughputs):
    sum_sq = np.sum(throughputs)**2
    sq_sum = np.sum(np.square(throughputs))
    return sum_sq / (len(throughputs) * sq_sum) if sq_sum > 0 else 0.0

# =========================================================
# MODULE 4: SIMULATION ENVIRONMENT & STRATEGIES (unchanged)
# =========================================================
class NomaEnv:
    def __init__(self):
        self.bits1_tx = generate_bits(PARAMS['NUM_BITS'])
        self.bits2_tx = generate_bits(PARAMS['NUM_BITS'])
        self.qam1 = bits_to_qam(self.bits1_tx)
        self.qam2 = bits_to_qam(self.bits2_tx)
        self.ofdm1 = ofdm_modulate(self.qam1)
        self.ofdm2 = ofdm_modulate(self.qam2)
        self.user1_pos = np.array([50.0, 50.0])
        self.user2_pos = np.array([-80.0, 20.0])

    def run_single_step_for_strategy(self, alpha, current_u1_pos, current_u2_pos):
        dist1 = np.linalg.norm(current_u1_pos - PHYSICS_PARAMS['BASE_STATION_POS'])
        dist2 = np.linalg.norm(current_u2_pos - PHYSICS_PARAMS['BASE_STATION_POS'])
        h1 = get_channel_gain(dist1)
        h2 = get_channel_gain(dist2)
        if np.abs(h1)**2 < np.abs(h2)**2:
            h_strong, h_weak, ofdm_strong, ofdm_weak, bits_tx_strong, bits_tx_weak = h2, h1, self.ofdm2, self.ofdm1, self.bits2_tx, self.bits1_tx
        else:
            h_strong, h_weak, ofdm_strong, ofdm_weak, bits_tx_strong, bits_tx_weak = h1, h2, self.ofdm1, self.ofdm2, self.bits1_tx, self.bits2_tx
        
        rx_signal_base = h_strong * np.sqrt(alpha * PARAMS['TOTAL_POWER']) * ofdm_strong + h_weak * np.sqrt((1 - alpha) * PARAMS['TOTAL_POWER']) * ofdm_weak
        noise = np.sqrt(NOISE_POWER_W/2) * (np.random.randn(*rx_signal_base.shape) + 1j*np.random.randn(*rx_signal_base.shape))
        y_freq = ofdm_demodulate((rx_signal_base + noise).flatten())
        bits_strong_rx, bits_weak_rx = perform_sic(y_freq, h_strong, h_weak, alpha)
        
        if np.abs(h1)**2 < np.abs(h2)**2:
            bits1_rx, bits2_rx = bits_weak_rx, bits_strong_rx
        else:
            bits1_rx, bits2_rx = bits_strong_rx, bits_weak_rx
            
        ber1, ber2 = calculate_ber(self.bits1_tx, bits1_rx), calculate_ber(self.bits2_tx, bits2_rx)
        tp1, tp2 = calculate_throughput(ber1, PARAMS['NUM_BITS']), calculate_throughput(ber2, PARAMS['NUM_BITS'])
        fairness, total_throughput, average_ber = calculate_jains_fairness([tp1, tp2]), tp1 + tp2, (ber1 + ber2) / 2
        signal_power = np.mean(np.abs(rx_signal_base)**2)
        snr_db = 10 * np.log10(signal_power / NOISE_POWER_W) if NOISE_POWER_W > 0 else 100
        return total_throughput, average_ber, fairness, snr_db

# =========================================================
# MODULE 5 & 6: STATIC DASHBOARD & BUTTON-DRIVEN LOOP
# =========================================================
print("Setting up simulation dashboard...")
start_time = time.time()

env = NomaEnv()
strategies = {
    'Weak User Priority (α=0.2)': 0.2,
    'Equal Power (α=0.5)': 0.5,
    'Strong User Priority (α=0.8)': 0.8
}

SMOOTHING_WINDOW = 50
raw_history = {name: {'tp': deque(maxlen=SMOOTHING_WINDOW), 'ber': deque(maxlen=SMOOTHING_WINDOW), 'fairness': deque(maxlen=SMOOTHING_WINDOW)} for name in strategies}
smoothed_history = {name: {'tp': [], 'ber': [], 'fairness': []} for name in strategies}
global_snr_history = deque(maxlen=SMOOTHING_WINDOW)
step = 0

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("NOMA System Performance: Comparing Fixed Power Strategies", fontsize=20)
(ax_map, ax_tp), (ax_ber, ax_fair) = axes
colors = {'Weak User Priority (α=0.2)': 'tab:blue', 'Equal Power (α=0.5)': 'tab:green', 'Strong User Priority (α=0.8)': 'tab:red'}

# --- Setup Plots (as before) ---
ax_map.set_title("User Movement in 2D Space", fontsize=14)
ax_map.set_xlim(-250, 250); ax_map.set_ylim(-250, 250)
ax_map.set_xlabel("X-coordinate (meters)"); ax_map.set_ylabel("Y-coordinate (meters)")
bs_plot = ax_map.plot(0, 0, 'o', markersize=12, color='gray', label='Base Station', alpha=0.7)[0]
user1_plot = ax_map.plot([], [], 'X', markersize=12, color='darkblue', label='User 1')[0]
user2_plot = ax_map.plot([], [], 'o', markersize=10, color='darkgreen', label='User 2')[0]
map_info_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, fontsize=10, verticalalignment='top')
ax_map.legend(); ax_map.grid(True, linestyle=':', alpha=0.6)
ax_tp.set_title("Total Throughput (Smoothed)", fontsize=14); ax_tp.set_xlabel("Time Step"); ax_tp.set_ylabel("Bits/Symbol")
ax_ber.set_title("Average BER (Smoothed)", fontsize=14); ax_ber.set_xlabel("Time Step"); ax_ber.set_ylabel("Bit Error Rate (log scale)"); ax_ber.set_yscale('log')
ax_fair.set_title("Jain's Fairness Index (Smoothed)", fontsize=14); ax_fair.set_xlabel("Time Step"); ax_fair.set_ylabel("Fairness Index (0-1)")
smoothed_lines = {name: {
    'tp': ax_tp.plot([], [], color=colors[name], label=name)[0],
    'ber': ax_ber.plot([], [], color=colors[name], label=name)[0],
    'fairness': ax_fair.plot([], [], color=colors[name], label=name)[0]
} for name in strategies}
for ax in [ax_tp, ax_ber, ax_fair]: ax.legend(fontsize=10); ax.grid(True, linestyle='--', alpha=0.7)

#
# *** LAYOUT FIX: Use subplots_adjust instead of tight_layout ***
#
fig.subplots_adjust(left=0.07, bottom=0.12, right=0.95, top=0.92, wspace=0.25, hspace=0.35)


# --- Button Logic ---
# Define the function to be called on button click
def next_step(event):
    global step
    step += 1
    
    # Move users
    env.user1_pos += np.random.randn(2) * 5
    env.user2_pos += np.random.randn(2) * 5
    
    current_avg_snr = 0
    # Run strategies and update history
    for name, alpha_val in strategies.items():
        tp, ber, fairness, snr_db = env.run_single_step_for_strategy(alpha_val, env.user1_pos, env.user2_pos)
        raw_history[name]['tp'].append(tp)
        raw_history[name]['ber'].append(ber if ber > 1e-10 else 1e-10)
        raw_history[name]['fairness'].append(fairness)
        smoothed_history[name]['tp'].append(np.mean(raw_history[name]['tp']))
        smoothed_history[name]['ber'].append(np.mean(raw_history[name]['ber']))
        smoothed_history[name]['fairness'].append(np.mean(raw_history[name]['fairness']))
        current_avg_snr += snr_db / len(strategies)
    
    global_snr_history.append(current_avg_snr)

    # Update all plot data
    time_steps = range(1, step + 1) # Start x-axis from 1
    user1_plot.set_data([env.user1_pos[0]], [env.user1_pos[1]])
    user2_plot.set_data([env.user2_pos[0]], [env.user2_pos[1]])
    map_info_text.set_text(f'Current Step: {step}\nAvg Network SNR (last {min(step, SMOOTHING_WINDOW)} steps): {np.mean(global_snr_history):.1f} dB')
    for name in strategies:
        smoothed_lines[name]['tp'].set_data(time_steps, smoothed_history[name]['tp'])
        smoothed_lines[name]['ber'].set_data(time_steps, smoothed_history[name]['ber'])
        smoothed_lines[name]['fairness'].set_data(time_steps, smoothed_history[name]['fairness'])
    
    for ax in [ax_tp, ax_ber, ax_fair]:
        ax.relim()
        ax.autoscale_view(True,True,True)
    
    # Redraw the figure
    fig.canvas.draw_idle()

# --- Draw Initial State (Step 0) ---
def draw_initial_state():
    user1_plot.set_data([env.user1_pos[0]], [env.user1_pos[1]])
    user2_plot.set_data([env.user2_pos[0]], [env.user2_pos[1]])
    map_info_text.set_text(f'Current Step: 0\nClick "Next Step" to begin.')
    for name in strategies:
        smoothed_lines[name]['tp'].set_data([], [])
        smoothed_lines[name]['ber'].set_data([], [])
        smoothed_lines[name]['fairness'].set_data([], [])
    fig.canvas.draw_idle()

draw_initial_state()
    
# Create the button
ax_button = plt.axes([0.45, 0.02, 0.1, 0.04]) # x, y, width, height
button = Button(ax_button, 'Next Step', color='lightgoldenrodyellow', hovercolor='0.975')

#
# *** BUTTON FIX: Store a persistent reference to the button object ***
#
button.on_clicked(next_step)
fig.button = button # This prevents the button from being garbage collected

print("Dashboard ready. Click 'Next Step' to advance the simulation.")
plt.show()

end_time = time.time()
print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")