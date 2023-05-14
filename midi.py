
'''MIDI and ABC file input/output.'''

import collections
from prettymidi import pretty_midi as pm

# Our ABC format is a simple text-based notation for music that allows us to
# notate chords as a single token (e.g. a#2c3e3g3). This representation is 
# useful for training, as it densifies our training data, and serializing this
# # to text makes it easier for us to inspect the content.

class InstrumentCategory:
  def __init__(self, name, keywords, center_pitch=60, octave_range=1):
    self.name = name
    self.keywords = keywords
    self.center_pitch = center_pitch
    self.octave_range = octave_range

# Our piano supports a range of +- 3 octaves around C4 (pitch index 60).
piano = InstrumentCategory('piano', ['piano', 'vocal', 'chorus', 'melody', 'lead'], octave_range=2)
# Our violin supports a range of +- 2 octaves around C5 (pitch index 72).
violin = InstrumentCategory('violin', ['violin', 'string'], center_pitch=72, octave_range=2)
# Our viola supports a range of +- 2 octaves around G4 (pitch index 67).
viola = InstrumentCategory('viola', ['viola', 'string'], center_pitch=67, octave_range=2)
# Our cello supports a range of +- 2 octaves around C4 (pitch index 60).
cello = InstrumentCategory('cello', ['cello', 'string'], octave_range=2)

pitches = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

def is_alpha_chord(alpha):
  '''Returns true if the alphanumeric pitch is a chord (contains more than one note).'''
  return len(alpha) > 3

def remove_first_note_from_alpha_chord(alpha):
  '''Removes the first note from an alphanumeric chord.'''
  if not is_alpha_chord(alpha):
    return ''
  if alpha[1] == '#':
    return alpha[3:]
  return alpha[2:]

def midi_to_alpha(midi_pitch):
  '''Converts a MIDI pitch index to alphanumeric format.'''
  octave = (midi_pitch // 12) - 1
  note = pitches[midi_pitch % 12]
  return f"{note}{octave}"

def alpha_to_midi(alpha, octave=0, accidental=0):
  '''Converts an alphanumeric pitch to MIDI format.'''
  return pitches.index(alpha) + accidental + (octave + 1) * 12

def normalize_transpose(pitch, target_instrument=piano):
  '''Transposes a pitch to be within the range of the target instrument.'''
  lower_threshold = target_instrument.center_pitch - target_instrument.octave_range * 12
  upper_threshold = target_instrument.center_pitch + target_instrument.octave_range * 12
  if pitch < lower_threshold:
    return pitch + 12 * ((lower_threshold - pitch) // 12 + 1)
  elif pitch > target_instrument.center_pitch + target_instrument.octave_range * 12:
    return pitch - 12 * ((pitch - upper_threshold) // 12 + 1)
  return pitch

def read_midi_file(filename, normalize=True, target_instrument=piano, verbose=False):
  '''Reads in a MIDI file and returns a notes dict.'''
  combined_instrument_notes = []
  mid = pm.PrettyMIDI(filename)
  pitch_mean = 0

  if len(mid.instruments) == 0:
    if verbose:
      print('No tracks found in midi file.')
    return
    
  for instrument in mid.instruments:
    if any(x in instrument.name.lower().strip() for x in target_instrument.keywords) and len(instrument.notes) > 0:
      if verbose:
        print('Opting to include instrument {}.'.format(instrument.name))
      combined_instrument_notes.extend(instrument.notes)

  if len(combined_instrument_notes) == 0:
    for instrument in mid.instruments:
      if len(instrument.notes) > 0:
        combined_instrument_notes.extend(instrument.notes)
    if len(combined_instrument_notes) != 0 and verbose:
      print('Using untitled tracks because no usable named tracks were found.')

  if len(combined_instrument_notes) == 0:
    if verbose:
      print('No usable tracks found in midi file ' + filename + '.')
      print([instrument.name for instrument in mid.instruments])
    return

  notes = collections.defaultdict(list)
  sorted_notes = sorted(combined_instrument_notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    pitch_mean += note.pitch
    notes['pitch'].append(note.pitch)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    notes['velocity'].append(note.velocity)
    prev_start = start

  pitch_mean = pitch_mean // len(notes['pitch'])

  if normalize:
    delta = pitch_mean - target_instrument.center_pitch
    notes['pitch'] = [normalize_transpose(pitch - delta, target_instrument) for pitch in notes['pitch']]
    notes['step'] = [round(x, 2) for x in notes['step']]
    notes['duration'] = [round(x, 2) for x in notes['duration']]

  condensed_notes = collections.defaultdict(list)
  for index in range(len(notes['pitch'])):
    if index == 0 or notes['step'][index] > 0:
      condensed_notes['pitch'].append(midi_to_alpha(notes['pitch'][index]))
      condensed_notes['step'].append(notes['step'][index])
      condensed_notes['duration'].append(notes['duration'][index])
      condensed_notes['velocity'].append(notes['velocity'][index])
    else:
      condensed_notes['pitch'][-1] += midi_to_alpha(notes['pitch'][index])

  return condensed_notes

def read_abc_file(filename):
  '''Reads in an ABC song file and returns a notes dict.'''
  notes = collections.defaultdict(list)
  with open(filename, 'r') as f:
    for line in f:
      pitch, step, duration, velocity = line.split(',')
      notes['pitch'].append(pitch.strip())
      notes['step'].append(float(step.strip()))
      notes['duration'].append(float(duration.strip()))
      notes['velocity'].append(int(velocity.strip()))

  return notes

def write_midi_file(notes, filename, instrument='Piano'):
  '''Writes a notes dict to disk as a MIDI file.'''
  mid = pm.PrettyMIDI()
  instrument = pm.Instrument(program=0, is_drum=False, name=instrument)
  prev_start = 0.0

  for index in range(len(notes['pitch'])):
    start = float(prev_start + notes['step'][index])
    end = float(start + notes['duration'][index])
    prev_start = start

    if 'velocity' in notes and len(notes['velocity']) > 0:
        velocity = notes['velocity'][index]
    else:
        velocity = 64

    note_index = 0
    while note_index < len(notes['pitch'][index]):
      if notes['pitch'][index][note_index + 1] == '#':
        pitch = alpha_to_midi(notes['pitch'][index][note_index], int(notes['pitch'][index][note_index + 2]), 1)
        note_index += 3
      else:
        pitch = alpha_to_midi(notes['pitch'][index][note_index], int(notes['pitch'][index][note_index + 1]))
        note_index += 2
      note = pm.Note(velocity=velocity, pitch=pitch, start=start, end=end)
      instrument.notes.append(note)

  mid.instruments.append(instrument)
  mid.write(filename)
  return mid

def write_abc_file(notes, filename):
  '''Write a text file in ABC format, where each line is pitch, step, duration, velocity.'''
  with open(filename, 'w') as f:
    for index in range(len(notes['pitch'])):
      f.write(f"{notes['pitch'][index]}, {notes['step'][index]}, {notes['duration'][index]}, {notes['velocity'][index]}\n")

