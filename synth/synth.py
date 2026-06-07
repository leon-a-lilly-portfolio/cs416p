import numpy as np
import sounddevice as sd
import mido
import threading
import queue

SAMPLERATE = 48000
BLOCK_SIZE = 256
AMPLITUDE = 0.708
ATTACK_SAMPLES = int(0.010 * SAMPLERATE)
RELEASE_SAMPLES = int(0.010 * SAMPLERATE)


def midi_note_to_freq(note):
    return 440.0 * (2 ** ((note - 69) / 12))


class Synth:
    def __init__(self):
        self.freq = None
        self.phase = 0
        self.state = 'idle'
        self.env_pos = 0
        self.env_level = 0.0
        self.lock = threading.Lock()

    def note_on(self, note):
        with self.lock:
            self.freq = midi_note_to_freq(note)
            self.phase = 0
            self.state = 'attack'
            self.env_pos = 0
            print(f"[NOTE ON] note={note} freq={self.freq:.2f}Hz")

    def note_off(self):
        with self.lock:
            if self.state != 'idle':
                self.state = 'release'
                self.env_pos = 0
                print("[NOTE OFF]")

    def generate(self, num_samples):
        with self.lock:
            if self.state == 'idle':
                return np.zeros(num_samples)

            phase_inc = self.freq / SAMPLERATE
            phases = (self.phase + np.arange(num_samples) * phase_inc) % 1.0
            wave = 2.0 * phases - 1.0  # sawtooth from -1 to 1
            self.phase = (self.phase + num_samples * phase_inc) % 1.0

            envelope = np.ones(num_samples)
            for i in range(num_samples):
                if self.state == 'attack':
                    self.env_level = self.env_pos / ATTACK_SAMPLES
                    self.env_pos += 1
                    if self.env_pos >= ATTACK_SAMPLES:
                        self.state = 'sustain'
                        self.env_level = 1.0
                elif self.state == 'sustain':
                    self.env_level = 1.0
                elif self.state == 'release':
                    self.env_level = 1.0 - (self.env_pos / RELEASE_SAMPLES)
                    self.env_pos += 1
                    if self.env_pos >= RELEASE_SAMPLES:
                        self.state = 'idle'
                        self.env_level = 0.0
                elif self.state == 'idle':
                    self.env_level = 0.0
                envelope[i] = self.env_level

            return wave * envelope * AMPLITUDE


synth = Synth()
event_queue = queue.Queue()


def audio_callback(outdata, frames, time, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    while not event_queue.empty():
        try:
            event = event_queue.get_nowait()
            if event[0] == 'on':
                synth.note_on(event[1])
            elif event[0] == 'off':
                synth.note_off()
        except queue.Empty:
            break
    samples = synth.generate(frames)
    outdata[:, 0] = samples


def midi_listener():
    ports = mido.get_input_names()
    print(f"[MIDI] Available ports: {ports}")
    with mido.open_input() as port:
        print(f"[MIDI] Listening on: {port.name}")
        for msg in port:
            try:
                print(f"[MIDI] {msg}")
                if msg.type == 'note_on' and msg.velocity > 0:
                    event_queue.put(('on', msg.note))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    event_queue.put(('off', None))
            except Exception:
                break


midi_thread = threading.Thread(target=midi_listener, daemon=True)
midi_thread.start()


with sd.OutputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
    print("Synth running. Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopped.")
