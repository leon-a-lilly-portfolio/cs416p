import numpy as np
from scipy.io import wavfile
import random
import argparse

SAMPLERATE = 48000

# All notes from A3 to A4 as frequencies
NOTE_FREQS = {
    'A3':  220.00,
    'A#3': 233.08,
    'B3':  246.94,
    'C4':  261.63,
    'C#4': 277.18,
    'D4':  293.66,
    'D#4': 311.13,
    'E4':  329.63,
    'F4':  349.23,
    'F#4': 369.99,
    'G4':  392.00,
    'G#4': 415.30,
    'A4':  440.00,
}

# major scale intervals in semitones from root
# 1  2  3  4  5  6  7
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]

def semitones_up(freq, semitones):
    return freq * (2 ** (semitones / 12))

def get_scale(root_freq):
    return [semitones_up(root_freq, i) for i in MAJOR_INTERVALS]

# scale degree index for each roman num
SCALE_DEGREE = {
    'I':   0,
    'ii':  1,
    'iii': 2,
    'IV':  3,
    'iv':  3,
    'V':   4,
    'vi':  5,
    'vii': 6,
}

# chord shapes
# major: root, maj 3rd, perf 5th
# minor: root, min 3rd, perf 5th
MAJOR_CHORD = [0, 4, 7]
MINOR_CHORD = [0, 3, 7]

# minor chord numerals
MINOR_NUMERALS = {'ii', 'iii', 'iv', 'vi'}

def get_chord_freqs(numeral, scale, root_freq):
    degree = SCALE_DEGREE[numeral]
    chord_root = scale[degree]
    shape = MINOR_CHORD if numeral in MINOR_NUMERALS else MAJOR_CHORD
    return [semitones_up(chord_root, i) for i in shape]

SONG_STRUCTURES = [
    "AABBCC",
    "ABABCD",
    "ABCDDD",
]

CHORD_LOOPS = [
    ['I', 'IV', 'ii', 'V'],
    ['I', 'vi', 'ii', 'V'],
    ['I', 'iii', 'IV', 'iv'],
    ['I', 'V', 'ii', 'V'],
    ['I', 'vi', 'IV', 'V'],
    ['IV', 'I', 'vi', 'IV'],
    ['I', 'V', 'vi', 'I'],
    ['I', 'IV', 'iv', 'I'],
    ['IV', 'V', 'I', 'I'],
    ['vi', 'IV', 'I', 'V'],
]

def assign_loops(structure):
    labels = list(dict.fromkeys(structure))
    loops = random.sample(CHORD_LOOPS, len(labels))
    return {label: loop for label, loop in zip(labels, loops)}

def expand_structure(structure, label_to_loop):
    return [label_to_loop[label] for label in structure]

tempo = random.randint(80, 160)
beats_per_measure = 4
measures_per_line = 4
eighth_notes_per_measure = 8

seconds_per_beat = 60.0 / tempo
seconds_per_measure = seconds_per_beat * beats_per_measure
seconds_per_eighth = seconds_per_beat / 2

samples_per_eighth = int(SAMPLERATE * seconds_per_eighth)

def sawtooth(freq, num_samples):
    t = np.linspace(0, num_samples / SAMPLERATE, num_samples, endpoint=False)
    wave = np.zeros(num_samples)
    harmonic = 1
    while True:
        harmonic_freq = freq * harmonic
        if harmonic_freq >= SAMPLERATE / 2:
            break
        wave += np.sin(2 * np.pi * harmonic_freq * t) / harmonic
        harmonic += 1
    return wave

def normalize(wave, amplitude=0.8):
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * amplitude
    return wave


def generate_melody(song, scale, root_freq, bass=False, harmony=False):
    all_samples = []
    samples_per_measure = samples_per_eighth * 8

    for chord_loop in song:
        for numeral in chord_loop:
            chord_freqs = get_chord_freqs(numeral, scale, root_freq)
            chord_root = chord_freqs[0]
            measure_samples = []

            for beat in range(beats_per_measure):
                for eighth in range(2):
                    if random.random() < 0.8:
                        freq = random.choice(chord_freqs)
                    else:
                        freq = random.choice(scale)
                    note = sawtooth(freq, samples_per_eighth)

                    if harmony:
                        harm_freq = find_harmony(freq, chord_freqs)
                        if harm_freq is not None:
                            note = note + sawtooth(harm_freq, samples_per_eighth)

                    measure_samples.append(note)

            measure = np.concatenate(measure_samples)

            if bass:
                bass_note = generate_bass_note(chord_root, samples_per_measure)
                measure = measure + bass_note

            all_samples.append(measure)

    return np.concatenate(all_samples)


def generate_bass_note(chord_root_freq, num_samples):
    bass_freq = chord_root_freq / 4
    return sawtooth(bass_freq, num_samples)


def find_harmony(melody_freq, chord_freqs):
    below = [f for f in chord_freqs if f < melody_freq]
    return max(below) if below else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--bass', action='store_true', default=False)
    parser.add_argument('--harmony', action='store_true', default=False)

    args = parser.parse_args()

    # pick key, scale, tempo
    root_name = random.choice(list(NOTE_FREQS.keys()))
    root_freq = NOTE_FREQS[root_name]
    scale = get_scale(root_freq)
    tempo = random.randint(80, 160)

    # timing
    seconds_per_beat = 60.0 / tempo
    seconds_per_eighth = seconds_per_beat / 2
    samples_per_eighth = int(SAMPLERATE * seconds_per_eighth)

    # song structure
    structure = random.choice(SONG_STRUCTURES)
    label_to_loop = assign_loops(structure)
    song = expand_structure(structure, label_to_loop)

    # generate
    audio = generate_melody(song, scale, root_freq, bass=args.bass, harmony=args.harmony)
    audio = normalize(audio)
    audio_int16 = (audio * 32767).astype(np.int16)

    # output
    if args.output:
        wavfile.write(args.output, SAMPLERATE, audio_int16)
    else:
        import sounddevice as sd
        sd.play(audio_int16, SAMPLERATE)
        sd.wait()

if __name__ == '__main__':
    main()
