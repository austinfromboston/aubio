#! /usr/bin/env python

# Use pyaudio to record audio from microphone and run aubio's tempo detection
# on the incoming audio stream to detect beats in real-time.
# If a filename is given as the first argument, it will record 10 seconds of
# audio to this location. Otherwise, the script will run until Ctrl+C is pressed.
# You can also specify an OSC host to send beat events via OSC.

# Examples:
#    $ ./python/demos/demo_beat_tracking.py
#    $ ./python/demos/demo_beat_tracking.py /tmp/recording.wav
#    $ ./python/demos/demo_beat_tracking.py --osc-host 127.0.0.1 --osc-port 8000
#    $ ./python/demos/demo_beat_tracking.py /tmp/recording.wav --osc-host 127.0.0.1 --osc-port 8000

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
import argparse
import shutil
from dataclasses import dataclass

# Import python-osc for OSC message sending
try:
    from pythonosc import udp_client

    osc_available = True
except ImportError:
    osc_available = False
    print("*** Note: python-osc not available. Install with: pip install python-osc")

# Try to import select module - needed for keyboard input
try:
    import select

    select_available = True
except ImportError:
    select_available = False
    print("*** Note: select module not available - keyboard controls disabled")


def safe_print(text):
    """Safely print text with blocking I/O handling"""
    while True:
        try:
            print(text, end="", flush=True)
            break
        except BlockingIOError:
            time.sleep(0.001)  # Wait a millisecond
        except Exception:
            break  # Give up on other errors


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
                print(
                    "*** Note: Not running in an interactive terminal - keyboard controls disabled"
                )
        except (AttributeError, OSError):
            self.enabled = False
            print(
                "*** Note: Could not determine terminal type - keyboard controls disabled"
            )

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
            if ch == "\x1b":  # ESC
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
                return "".join(seq)  # Return partial sequence

            # Try to read next character
            ch = self.read_char()
            if ch:
                self.escape_seq.append(ch)

                # Check if we have a complete sequence
                if (
                    len(self.escape_seq) >= 3
                    and self.escape_seq[0] == "\x1b"
                    and self.escape_seq[1] == "["
                ):
                    seq, self.escape_seq = self.escape_seq, []
                    self.escape_pending = False
                    return "".join(seq)

        return None

    def check_keyboard_input(self):
        """Check for keyboard input and return detected key"""
        # If we have a pending escape sequence, continue reading it
        if self.escape_pending:
            return self.read_escape_sequence()

        # Get the next character (may be regular key or start of escape sequence)
        key = self.read_escape_sequence()

        # Still support escape sequences for future expansion
        if key == "\x1b[A":
            return "UP"
        elif key == "\x1b[B":
            return "DOWN"
        elif key == "\x1b[C":
            return "RIGHT"
        elif key == "\x1b[D":
            return "LEFT"

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
stream = p.open(
    format=pyaudio_format,
    channels=n_channels,
    rate=samplerate,
    input=True,
    frames_per_buffer=buffer_size,
)

print("*** Starting real-time beat tracking")
print("*** Press Ctrl+C to stop")
if term.enabled:
    print("*** Press '[' to decrease or ']' to increase energy threshold")
    print("*** Press SPACE to realign to beat 1 of bar")
    # Set up terminal for raw input mode
    term.setup()

    # Print initial empty lines for energy display
    print("\n\n\n", end="")  # Three blank lines for energy bar, threshold, and status

# Parse command line arguments
parser = argparse.ArgumentParser(description="Real-time beat tracking with aubio")
parser.add_argument(
    "output_file", nargs="?", help="Optional output file to record 10 seconds of audio"
)
parser.add_argument("--osc-host", help="OSC host IP address to send beat events")
parser.add_argument(
    "--osc-port", type=int, default=8000, help="OSC port (default: 8000)"
)
args = parser.parse_args()

# Set up OSC client if host is provided and library is available
osc_client = None
if args.osc_host and osc_available:
    try:
        osc_client = udp_client.SimpleUDPClient(args.osc_host, args.osc_port)
        print(f"*** Sending OSC messages to {args.osc_host}:{args.osc_port}")
    except Exception as e:
        print(f"*** Error setting up OSC client: {e}")
        osc_client = None
elif args.osc_host and not osc_available:
    print("*** Cannot send OSC messages: python-osc library not available")

# Check if output file was provided
if args.output_file:
    # Record for 10 seconds
    output_filename = args.output_file
    record_duration = 10
    total_frames = 0
    outputsink = aubio.sink(output_filename, samplerate)
else:
    # Run indefinitely until Ctrl+C
    outputsink = None
    record_duration = None

# Initialize tempo detection
# We'll use a larger window size to better capture low frequency content
win_s = 1024  # FFT window size

# Initialize detectors
tempo_o = aubio.tempo("default", win_s, hop_size, samplerate)
onset_o = aubio.onset("default", win_s, hop_size, samplerate)

# Configure onset detector
onset_o.set_threshold(0.3)

# Parameters to avoid spurious detections
silence_threshold = -40  # in dB, avoid detections in silence (increased from -70)
minimum_inter_onset = 0.05  # in seconds, minimum time between two consecutive onsets
energy_threshold = 0.05  # minimum RMS energy level required for beat detection
peak_threshold = 0.2  # peak threshold for onset detection
tempo_o.set_threshold(peak_threshold)


# Function to synchronize thresholds
def update_thresholds():
    global energy_threshold, peak_threshold
    # Keep peak_threshold proportional to energy_threshold (maintain the same relative scale)
    peak_threshold = (
        energy_threshold * 4
    )  # Scale factor of 4 from initial values (0.05 -> 0.2)
    peak_threshold = min(0.99, peak_threshold)  # Ensure it doesn't exceed 0.99
    tempo_o.set_threshold(peak_threshold)  # Update the aubio detector


# For beat visualization and timing
last_beat_time = time.time()
beat_count = 0
beat_locations = []  # in seconds

# Parameters for segment detection
bpm_history = []  # Store recent BPM values
bpm_history_size = 8  # Store 8 beats worth of BPM
max_bpm_variation = 8.0  # BPM variation threshold (increased)
db_history = []  # Store recent dB values
db_variation_threshold = 8.0  # dB variation threshold (increased)

# For stability checking
# stable_energy_threshold = 2.4  # Maximum allowed energy variation
beats_to_skip = 2  # Number of beats to skip after segment change (4 bars)
minimum_segment_length = 0

# For bar tracking in 4/4 time
current_beat_in_bar = 0  # 0-3 for beats 1-4
bar_count = 0
last_beat_energy = 0  # Store energy of last beat for comparison
# energy_threshold_for_downbeat = (
#     1.015  # Multiplier for downbeat detection (1.5% increase)
# )
# allow_manual_realign = True  # Allow manual realignment of bars

# For energy-based downbeat detection and stability checking
beat_energy_history = []  # Store recent beat energies
energy_history_size = 8  # Store last 8 beats (2 bars)
beats_since_energy_change = 0  # Start stable

# For energy level visualization
energy_history = [0] * 20  # Keep a short history for smoother display
max_energy_seen = 0.1  # Start with a reasonable default

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


# Function to determine energy bar size based on terminal width
def get_energy_bar_size():
    try:
        terminal_width = shutil.get_terminal_size().columns
        # Reserve space for the text around the bar:
        # "Energy: []" and " -XX.XdB | BPM: XXX.X" (approximately 35 chars)
        available_width = terminal_width - 35
        # Ensure minimum size of 10 and maximum that fits
        return max(10, min(available_width, 80))
    except:
        return 40  # fallback to default size


try:  # ctrl-c detection
    # Main processing loop
    while True:
        try:
            # Read audio data from stream
            audiobuffer = stream.read(hop_size, exception_on_overflow=False)
            signal = np.frombuffer(audiobuffer, dtype=np.float32)
        except IOError as e:
            print(f"\nAudio input error: {e}")
            continue  # Try to recover
        except Exception as e:
            print(f"\nUnexpected audio error: {e}")
            break  # Exit the loop on serious errors

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

        # Run detectors with error handling
        try:
            is_beat = tempo_o(signal)
            is_onset = onset_o(signal)
        except Exception as e:
            print(f"\nDetector error: {e}")
            continue

        # Only consider beats if audio energy is above threshold and not in silence
        valid_beat = (
            is_beat
            and (db_level > silence_threshold)
            and (energy_level > energy_threshold)
        )

        # Display current energy level regardless of beat detection
        # Get dynamic bar length based on terminal width
        energy_bar_len = get_energy_bar_size()
        energy_bar = "#" * int(energy_level * energy_bar_len)
        energy_bar = energy_bar.ljust(energy_bar_len)

        # Show energy level with threshold marker
        threshold_pos = int(energy_threshold * energy_bar_len)
        threshold_display = (
            " " * threshold_pos + "█" + " " * (energy_bar_len - threshold_pos - 1)
        )

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

        # Get current tempo estimate
        current_bpm = tempo_o.get_bpm()

        # Prepare display lines
        energy_line = (
            f"Energy: [{energy_bar}] {db_level:.1f}dB | BPM: {current_bpm:.1f}"
        )
        thresh_line = f"Thresh: [{threshold_display}] Energy: {energy_threshold:.2f} | Peak: {peak_threshold:.2f}{display_message}"
        status_line = ""  # For OSC status

        # Set appropriate status message
        if osc_client and beats_since_energy_change < minimum_segment_length:
            status_line = "*** Waiting for energy level to stabilize..."
        elif osc_client:
            status_line = f"OSC ready: {args.osc_host}:{args.osc_port}"

        # Handle display updates with error handling
        try:
            # Only update energy display if no recent beat
            if not valid_beat or (time.time() - last_beat_time) > 0.1:
                # Combine all display text
                display_text = (
                    "\033[K\033[A\033[K\033[A\033[K\r"  # Clear lines and move cursor
                    + f"{energy_line}\n"
                    + f"{thresh_line}\n"
                    + f"{status_line}"
                )
                safe_print(display_text)
        except Exception as e:
            print(f"\nDisplay error: {e}")
            # Don't break on display errors - continue

        # If a valid beat is detected
        if valid_beat:
            current_time = time.time()
            time_since_last_beat = current_time - last_beat_time

            # Only report if sufficient time has passed since last beat
            if time_since_last_beat > minimum_inter_onset:
                # Calculate beat and bar information
                beat_in_bar = (current_beat_in_bar + 1) % 4 or 4  # Convert 0 to 4
                beat_display = "=" * get_energy_bar_size()

                # 1. Display beat event with error handling
                try:
                    # Combine all display text into a single string to minimize writes
                    beat_text = (
                        f"\n⚡ BEAT {beat_in_bar}/4 | BAR {bar_count} | Energy: {energy_level:.2f} | BPM: {current_bpm:.1f}\n"
                        + f"{beat_display}\n"  # Single line for beat marker
                        + f"Energy: [{' ' * get_energy_bar_size()}] {db_level:.1f}dB | BPM: {current_bpm:.1f}\n"
                        + f"Thresh: [{threshold_display}] Energy: {energy_threshold:.2f} | Peak: {peak_threshold:.2f}{display_message}"
                    )
                    safe_print(beat_text)
                except Exception as e:
                    print(f"\nBeat display error: {e}")
                    # Continue even if beat display fails

                # 2. Process beat data with separate error handling
                try:
                    # Store energy history
                    beat_energy_history.append(energy_level)
                    if len(beat_energy_history) > energy_history_size:
                        beat_energy_history.pop(0)
                except Exception as e:
                    print(f"\nBeat history error: {e}")

                # Check for musical changes (BPM and dB level)
                segment_change = False
                change_reason = ""
                bpm_variation = 0
                db_variation = 0

                try:
                    # Store current values
                    bpm_history.append(current_bpm)
                    db_history.append(db_level)
                    if len(bpm_history) > bpm_history_size:
                        bpm_history.pop(0)
                    if len(db_history) > bpm_history_size:
                        db_history.pop(0)

                    # Check BPM and dB variations
                    if len(bpm_history) > 4 and len(db_history) > 4:
                        # BPM variation
                        bpm_mean = np.mean(bpm_history[:-1])
                        bpm_variation = abs(current_bpm - bpm_mean)

                        # dB variation
                        db_mean = np.mean(db_history[:-1])
                        db_variation = abs(db_level - db_mean)

                        # Detect significant changes only after minimum beat count
                        if beats_since_energy_change >= minimum_segment_length:
                            if bpm_variation > max_bpm_variation:
                                segment_change = True
                                change_reason = f"BPM change: {bpm_variation:.1f}"
                            elif db_variation > db_variation_threshold:
                                segment_change = True
                                change_reason = f"Volume change: {db_variation:.1f}dB"
                except Exception as e:
                    print(f"\nSegment detection error: {e}")
                    segment_change = False

                # 4. Handle segment change if detected
                if segment_change:
                    beats_since_energy_change = (
                        0  # Reset beat counter on segment change
                    )
                    current_beat_in_bar = 0  # Reset to beat 1
                    bar_count += 1  # Start new bar count

                    # Status will be updated in main display loop
                    status_line = f"*** New segment detected - {change_reason}"

                    # Send new segment OSC message
                    if osc_client:
                        try:
                            osc_client.send_message(
                                "/segment",
                                [
                                    beat_count,
                                    float(energy_level),
                                    float(current_bpm),
                                    float(db_level),
                                    float(bpm_variation),
                                    float(db_variation),
                                    change_reason,
                                ],
                            )
                            print(
                                f"Sent segment change OSC message to {args.osc_host}:{args.osc_port}"
                            )
                        except Exception as e:
                            print(f"OSC segment message error: {e}")
                            # Don't break on OSC errors

                beats_since_energy_change += 1

                # 5. Update beat tracking
                current_beat_in_bar = (current_beat_in_bar + 1) % 4
                if current_beat_in_bar == 0:
                    bar_count += 1

                beat_count += 1
                last_beat_time = current_time
                last_beat_energy = energy_level

                # Send beat OSC message if we're stable enough
                if osc_client and beats_since_energy_change >= beats_to_skip:
                    try:
                        osc_client.send_message(
                            "/beat",
                            [
                                beat_count,
                                float(energy_level),
                                float(current_bpm),
                                float(db_level),
                                bar_count,
                                beat_in_bar,
                            ],
                        )
                    except Exception as e:
                        print(f"OSC beat message error: {e}")

                # Get human-readable beat number (1-4)
                beat_in_bar = current_beat_in_bar + 1

                # Store beat location if we're recording
                if record_duration:
                    beat_time = total_frames / float(samplerate)
                    beat_locations.append(beat_time)

        # Write to output if recording - independent error handling
        if outputsink:
            try:
                outputsink(signal, len(signal))
                total_frames += len(signal)

                # Stop if we've reached the recording duration
                if record_duration and total_frames >= samplerate * record_duration:
                    break
            except Exception as e:
                print(f"\nRecording error: {e}")
                # Stop recording on error
                outputsink = None

        # Check for user input to adjust threshold - independent error handling
        if term.enabled:
            try:
                key = term.check_keyboard_input()

                # Handle keyboard inputs if a key was pressed
                if key:
                    # Handle '[' and ']' keys for threshold adjustment
                    if key == "]":  # Increase threshold (was UP arrow)
                        old_threshold = energy_threshold
                        energy_threshold = min(0.95, energy_threshold + 0.05)
                        if old_threshold != energy_threshold:
                            # Update both thresholds and the detector
                            update_thresholds()
                            threshold_changed = True
                            threshold_change_time = time.time()
                            threshold_change_count = 0
                            threshold_message = f"Threshold increased to: {energy_threshold:.2f} (peak: {peak_threshold:.2f})"
                    elif key == "[":  # Decrease threshold (was DOWN arrow)
                        old_threshold = energy_threshold
                        energy_threshold = max(0.01, energy_threshold - 0.05)
                        if old_threshold != energy_threshold:
                            # Update both thresholds and the detector
                            update_thresholds()
                            threshold_changed = True
                            threshold_change_time = time.time()
                            threshold_change_count = 0
                            threshold_message = f"Threshold decreased to: {energy_threshold:.2f} (peak: {peak_threshold:.2f})"
                    elif key == " ":  # Space key to reset to beat 1
                        current_beat_in_bar = (
                            3  # Set to 3 so next beat will be 0 (beat 1)
                        )
                        print("\n*** Realigned to beat 1 for next beat")
            except Exception as e:
                print(f"\nKeyboard input error: {e}")
                term.enabled = False  # Disable keyboard input on error

# Handle Ctrl+C gracefully
except KeyboardInterrupt:
    print("\n*** Ctrl+C pressed, exiting")
    exit

# Always make sure we restore the terminal when exiting the loop
term.restore()

# Cleanup
print("*** Done recording")
if len(beat_locations) > 0:
    print(f"Detected {len(beat_locations)} beats at: ")
    for i, beat in enumerate(beat_locations):
        print(f"Beat {i + 1}: {beat:.3f}s")

stream.stop_stream()
stream.close()
p.terminate()

if outputsink:
    print(f"Audio saved to: {output_filename}")
