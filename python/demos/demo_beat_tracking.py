#! /usr/bin/env python

# Use pyaudio to record audio from microphone and run aubio's tempo detection
# on the incoming audio stream to detect beats in real-time.
# If a filename is given as the first argument, it will record 10 seconds of 
# audio to this location. Otherwise, the script will run until Ctrl+C is pressed.

# Examples:
#    $ ./python/demos/demo_beat_tracking.py
#    $ ./python/demos/demo_beat_tracking.py /tmp/recording.wav

import pyaudio
import sys
import numpy as np
import aubio
import time
import os
import math
import fcntl
import termios
import tty
import atexit
from dataclasses import dataclass

# Try to import select module - needed for keyboard input
try:
    import select
    select_available = True
except ImportError:
    select_available = False
    print("*** Note: select module not available - keyboard controls disabled")

# Define a dataclass to encapsulate terminal state and functions
@dataclass
class TerminalState:
    # Terminal capabilities
    enabled: bool = True
    select_available: bool = select_available  # Use the module-level variable
    is_restored: bool = False
    old_settings = None
    
    # Escape sequence state
    escape_seq: list = None
    escape_pending: bool = False
    escape_timeout: float = 0
    
    def __post_init__(self):
        if self.escape_seq is None:
            self.escape_seq = []
        
        # Check if we're running in a TTY
        try:
            is_tty = sys.stdin.isatty()
            if not is_tty:
                self.enabled = False
                print("*** Note: Not running in an interactive terminal - keyboard controls disabled")
        except (AttributeError, OSError):
            self.enabled = False
            print("*** Note: Could not determine terminal type - keyboard controls disabled")
            
        # Update enabled status based on select availability
        if not self.select_available:
            self.enabled = False
    
    def setup(self):
        """Set up terminal for raw input mode"""
        if not self.enabled:
            return
        
        try:
            # Save old terminal settings
            fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(fd)
            
            # Set terminal to raw mode (but allow Ctrl+C to work)
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO & ~termios.ICANON
            new_settings[6][termios.VMIN] = 0  # Non-blocking
            new_settings[6][termios.VTIME] = 0  # No timeout
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            
            # Set stdin to non-blocking
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            # Register cleanup function
            atexit.register(self.restore)
            
            # Clear any pending input
            termios.tcflush(fd, termios.TCIOFLUSH)
            
        except Exception as e:
            print(f"Terminal setup error: {e}")
            self.enabled = False

    def restore(self):
        """Restore terminal to original settings"""
        if self.enabled and self.old_settings and not self.is_restored:
            try:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSAFLUSH, self.old_settings)
                
                # Restore blocking mode
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                
                # Mark as restored to avoid multiple restorations
                self.is_restored = True
                
                # Ensure terminal is left in a good state
                sys.stdout.write("\n\r")
                sys.stdout.flush()
            except Exception:
                # Ignore errors during terminal restoration
                pass

    def read_char(self):
        """Read a single character from stdin if available"""
        if not self.enabled or not self.select_available:
            return None
        
        try:
            # Check if data is available on stdin (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except (IOError, OSError):
            # Handle input errors
            pass
        except Exception as e:
            # Catch any other select-related errors
            print(f"\nWarning: Input read error: {e}")
            self.enabled = False  # Disable input if there are errors
        return None

    def read_escape_sequence(self):
        """Read a complete escape sequence"""
        # New key - start of potential escape sequence
        if not self.escape_pending and self.escape_seq == []:
            ch = self.read_char()
            if ch == '\x1b':  # ESC
                self.escape_seq = [ch]
                self.escape_pending = True
                self.escape_timeout = time.time() + 0.05  # 50ms timeout
                return None
            return ch
            
        # Continue reading escape sequence
        if self.escape_pending:
            # Check for timeout
            if time.time() > self.escape_timeout:
                seq, self.escape_seq = self.escape_seq, []
                self.escape_pending = False
                return ''.join(seq)  # Return partial sequence
                
            # Try to read next character
            ch = self.read_char()
            if ch:
                self.escape_seq.append(ch)
                
                # Check if we have a complete sequence
                if len(self.escape_seq) >= 3 and self.escape_seq[0] == '\x1b' and self.escape_seq[1] == '[':
                    seq, self.escape_seq = self.escape_seq, []
                    self.escape_pending = False
                    return ''.join(seq)
                    
        return None

    def check_keyboard_input(self):
        """Check for keyboard input and return detected key"""
        # If we have a pending escape sequence, continue reading it
        if self.escape_pending:
            return self.read_escape_sequence()
            
        # Get the next character (may be regular key or start of escape sequence)
        key = self.read_escape_sequence()
        
        # Still support escape sequences for future expansion
        if key == '\x1b[A':
            return 'UP'
        elif key == '\x1b[B':
            return 'DOWN'
        elif key == '\x1b[C':
            return 'RIGHT'
        elif key == '\x1b[D':
            return 'LEFT'
        
        # Otherwise return the key itself (for '[' and ']')
        return key

# Create a terminal state object
term = TerminalState()

# Parameters for audio processing
buffer_size = 1024
hop_size = buffer_size // 2
samplerate = 44100

# Initialize pyaudio
p = pyaudio.PyAudio()

# Open input stream
pyaudio_format = pyaudio.paFloat32
n_channels = 1
stream = p.open(format=pyaudio_format,
                channels=n_channels,
                rate=samplerate,
                input=True,
                frames_per_buffer=buffer_size)

print("*** Starting real-time beat tracking")
print("*** Press Ctrl+C to stop")
if term.enabled:
    print("*** Press '[' to decrease or ']' to increase energy threshold")
    # Set up terminal for raw input mode
    term.setup()

# Check if output file was provided
if len(sys.argv) > 1:
    # Record for 10 seconds
    output_filename = sys.argv[1]
    record_duration = 10
    total_frames = 0
    outputsink = aubio.sink(output_filename, samplerate)
else:
    # Run indefinitely until Ctrl+C
    outputsink = None
    record_duration = None

# Initialize tempo detection
# We'll use a larger window size to better capture low frequency content
win_s = 1024                # FFT window size
tempo_o = aubio.tempo("default", win_s, hop_size, samplerate)

# Parameters to avoid spurious detections
silence_threshold = -40     # in dB, avoid detections in silence (increased from -70)
minimum_inter_onset = 0.05  # in seconds, minimum time between two consecutive onsets
energy_threshold = 0.05     # minimum RMS energy level required for beat detection
peak_threshold = 0.2        # peak threshold for onset detection
tempo_o.set_threshold(peak_threshold)

# Function to synchronize thresholds
def update_thresholds():
    global energy_threshold, peak_threshold
    # Keep peak_threshold proportional to energy_threshold (maintain the same relative scale)
    peak_threshold = energy_threshold * 4  # Scale factor of 4 from initial values (0.05 -> 0.2)
    peak_threshold = min(0.99, peak_threshold)  # Ensure it doesn't exceed 0.99
    tempo_o.set_threshold(peak_threshold)  # Update the aubio detector

# For beat visualization and timing
last_beat_time = time.time()
beat_count = 0
beat_locations = []  # in seconds

# For energy level visualization
energy_history = [0] * 20   # Keep a short history for smoother display
max_energy_seen = 0.1       # Start with a reasonable default

# For threshold adjustment feedback
threshold_changed = False
threshold_change_time = 0
threshold_message = ""
threshold_change_count = 0  # Counter for flashing effect

# Function to calculate RMS energy of a signal
def get_rms(signal):
    # Guard against empty signals
    if len(signal) == 0:
        return 0
    # Calculate RMS
    return np.sqrt(np.mean(np.square(signal)))

# Function to convert value to dB
def amplitude_to_db(amplitude):
    if amplitude <= 0:
        return -100  # Return a very small dB value for zero/negative amplitudes
    return 20 * math.log10(amplitude)

# Main processing loop
try:
    while True:
        # Read audio data from stream
        audiobuffer = stream.read(hop_size, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)
        
        # Calculate RMS energy
        current_energy = get_rms(signal)
        energy_history.append(current_energy)
        energy_history.pop(0)  # Remove oldest value
        
        # Update max energy if needed (with some decay over time)
        max_energy_seen = max(max_energy_seen * 0.99, current_energy)
        
        # Calculate smoothed energy (average of recent history)
        smoothed_energy = np.mean(energy_history)
        
        # Energy level as a percentage of max seen
        energy_level = min(1.0, smoothed_energy / (max_energy_seen + 1e-10))
        
        # Current dB level
        db_level = amplitude_to_db(smoothed_energy)
        
        # Run tempo detection
        is_beat = tempo_o(signal)
        
        # Only consider beats if audio energy is above threshold and not in silence
        valid_beat = is_beat and (db_level > silence_threshold) and (energy_level > energy_threshold)
        
        # Display current energy level regardless of beat detection
        energy_bar_len = 40
        energy_bar = "#" * int(energy_level * energy_bar_len)
        energy_bar = energy_bar.ljust(energy_bar_len)
        
        # Show energy level with threshold marker
        threshold_pos = int(energy_threshold * energy_bar_len)
        threshold_display = " " * threshold_pos + "█" + " " * (energy_bar_len - threshold_pos - 1)
        
        # Add threshold change message if needed with visual effect
        display_message = ""
        if threshold_changed and time.time() - threshold_change_time < 2.0:
            # Create a flashing effect by alternating the message display
            threshold_change_count += 1
            if threshold_change_count % 8 < 4:  # Flash every ~0.25 seconds
                # Make the message stand out with ** markers
                display_message = f" | ** {threshold_message} **"
            else:
                display_message = f" | {threshold_message}"
        else:
            threshold_changed = False
        
        # Print current energy level and threshold bar with proper terminal control
        sys.stdout.write("\033[2K")  # Clear current line
        sys.stdout.write("\033[1A")  # Move up one line
        sys.stdout.write("\033[2K")  # Clear the line above
        sys.stdout.write("\r")       # Return to start
        
        # Get current tempo estimate
        current_bpm = tempo_o.get_bpm()
        
        # Write the new two-line display with BPM
        sys.stdout.write(f"Energy: [{energy_bar}] {db_level:.1f}dB | BPM: {current_bpm:.1f}\n")
        sys.stdout.write(f"Thresh: [{threshold_display}] Energy: {energy_threshold:.2f} | Peak: {peak_threshold:.2f}{display_message}")
        sys.stdout.flush()
        
        # If a valid beat is detected
        if valid_beat:
            current_time = time.time()
            time_since_last_beat = current_time - last_beat_time
            
            # Only report if sufficient time has passed since last beat
            if time_since_last_beat > minimum_inter_onset:
                beat_count += 1
                last_beat_time = current_time
                
                # Beat visualization with energy level and BPM
                beat_display = "=" * 50
                current_bpm = tempo_o.get_bpm()
                print(f"\n\nBEAT {beat_count} | Energy: {energy_level:.2f} | {db_level:.1f}dB | BPM: {current_bpm:.1f}\t|{beat_display}|")
                
                # Store beat location if we're recording
                if record_duration:
                    beat_time = total_frames / float(samplerate)
                    beat_locations.append(beat_time)
        
        # Write to output if recording
        if outputsink:
            outputsink(signal, len(signal))
            total_frames += len(signal)
            
            # Stop if we've reached the recording duration
            if record_duration and total_frames >= samplerate * record_duration:
                break
        
        # Check for user input to adjust threshold (non-blocking)
        if term.enabled:
            key = term.check_keyboard_input()
            
            # Handle '[' and ']' keys for threshold adjustment
            if key == ']':  # Increase threshold (was UP arrow)
                old_threshold = energy_threshold
                energy_threshold = min(0.95, energy_threshold + 0.05)
                if old_threshold != energy_threshold:
                    # Update both thresholds and the detector
                    update_thresholds()
                    threshold_changed = True
                    threshold_change_time = time.time()
                    threshold_change_count = 0
                    threshold_message = f"Threshold increased to: {energy_threshold:.2f} (peak: {peak_threshold:.2f})"
            elif key == '[':  # Decrease threshold (was DOWN arrow)
                old_threshold = energy_threshold
                energy_threshold = max(0.01, energy_threshold - 0.05)
                if old_threshold != energy_threshold:
                    # Update both thresholds and the detector
                    update_thresholds()
                    threshold_changed = True
                    threshold_change_time = time.time()
                    threshold_change_count = 0
                    threshold_message = f"Threshold decreased to: {energy_threshold:.2f} (peak: {peak_threshold:.2f})"
        
except KeyboardInterrupt:
    print("\n*** Ctrl+C pressed, exiting")
except Exception as e:
    print(f"\nError: {str(e)}")
finally:
    # Always make sure we restore the terminal
    term.restore()

# Cleanup
print("*** Done recording")
if len(beat_locations) > 0:
    print(f"Detected {len(beat_locations)} beats at: ")
    for i, beat in enumerate(beat_locations):
        print(f"Beat {i+1}: {beat:.3f}s")

stream.stop_stream()
stream.close()
p.terminate()

if outputsink:
    print(f"Audio saved to: {output_filename}")

