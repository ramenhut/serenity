
'''Pre-processing for MIDI files to simplify and improve the learning process.'''

import collections
from datetime import datetime
import midi
import numpy as np
import os
import tensorflow as tf
import zipfile

START_INDEX = 0
END_INDEX = 1
PAD_INDEX = 2
BASE_INDEX_COUNT = 3

top_k = 5000
max_tokens = 128
min_encode_tokens = 32
assert(max_tokens > min_encode_tokens)

source_path = './source'
processed_path = './processed'
checkpoint_path = './checkpoint'

seed = datetime.now().microsecond
np.random.seed(seed)
tf.random.set_seed(seed)

def zip_folder(folder_path, output_path):
  '''Zips a folder. Used to zip up the checkpoint files and dictionaries.'''
  with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_path):
      for file in files:
        zipf.write(os.path.join(root, file))

def unzip_folder(archive_path, output_path):
  '''Unzips a folder. Used to unzip the checkpoint files and dictionaries.'''
  with zipfile.ZipFile(archive_path, 'r') as zipf:
    zipf.extractall(output_path)

def get_nearest_token(value, dict):
  '''Returns the nearest token in the dictionary to the given value.'''
  return dict[min(dict.keys(), key=lambda x: abs(x - value))] + BASE_INDEX_COUNT

def convert_notes_to_tokens(notes, pitch_dict, step_dict, duration_dict, verbose=False):
  '''Converts note values to tokens that can be fed into a transformer network.'''
  recovered = 0
  output_notes = collections.defaultdict(list)
  for index in range(len(notes['pitch'])):
    pitch = notes['pitch'][index]
    step = notes['step'][index]
    duration = notes['duration'][index]

    # Recovery is critically important to the learning process. Skipping notes (because
    # they aren't represented in our dictionaries) severely impacts the model's ability
    # to learn. We can recover from this by finding the nearest token in the dictionary.
    # This is far from perfect, but it's better than skipping the note entirely.

    if pitch not in pitch_dict:
      if midi.is_alpha_chord(pitch):
        pitch = midi.remove_first_note_from_alpha_chord(pitch)
        while len(pitch) > 0:
          if pitch in pitch_dict:
            recovered += 1
            break
          pitch = midi.remove_first_note_from_alpha_chord(pitch)
        if len(pitch) == 0:
          continue
      else:
        continue

    if step not in step_dict:
      step = get_nearest_token(step, step_dict)
      recovered += 1

    if duration not in duration_dict:
      duration = get_nearest_token(duration, duration_dict)
      recovered += 1

    output_notes['pitch'].append(pitch_dict[pitch] + BASE_INDEX_COUNT)
    output_notes['step'].append(step_dict[step] + BASE_INDEX_COUNT)
    output_notes['duration'].append(duration_dict[duration] + BASE_INDEX_COUNT)

  if verbose:
    skipped = len(notes['pitch']) - len(output_notes['pitch'])
    print('Recovered ' + str(recovered) + ' tokens, and skipped ' + str(skipped) + 
          ' (' + str(round(100.0 * skipped / len(notes['pitch']), 2)) + '%) tokens.')
  return output_notes

def convert_tokens_to_notes(notes, pitch_rev_dict, step_rev_dict, duration_rev_dict):
  '''Converts tokens back to note values.'''
  notes['pitch'] = [pitch_rev_dict[note - BASE_INDEX_COUNT] for note in notes['pitch']]
  notes['step'] = [step_rev_dict[note - BASE_INDEX_COUNT] for note in notes['step']]
  notes['duration'] = [duration_rev_dict[note - BASE_INDEX_COUNT] for note in notes['duration']]

def convert_midi_files_to_abc(input_path, output_path, verbose=False):
  '''Converts MIDI files to ABC notation.'''
  files = os.listdir(input_path)
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  for file in files:
    try:
      notes = midi.read_midi_file(os.path.join(input_path, file), normalize=True)
    except:
        if verbose:
          print('Skipping conversion of ' + file + ' because it failed to load.')
        continue
    if not notes:
      if verbose:
        print('Skipping ' + file + ' because it was empty.')
      continue
    if len(notes['pitch']) < max_tokens:
      if verbose:
        print('Skipping ' + file + ' because it was too short.')
      continue
    midi.write_abc_file(notes, os.path.join(output_path, os.path.splitext(file)[0] + '.abc'))

def get_sorted_keys(dict, top_k=5000):
  '''Returns the top k keys in the dictionary, sorted by value.'''
  sorted_tuples = sorted(dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
  return [x[0] for x in sorted_tuples]

def generate_dictionaries(input_path, output_path, top_k=5000, verbose=False):
  '''Generates dictionaries for the pitch, step, and duration values.'''
  pitch_dict, step_dict, dur_dict = {}, {}, {}
  pitch_index, step_index, dur_index = 0, 0, 0

  files = os.listdir(input_path)
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  for file in files:
    try:
      if os.path.splitext(file)[1] == '.mid':
        notes = midi.read_mid_file(os.path.join(input_path, file))
      elif os.path.splitext(file)[1] == '.abc':
        notes = midi.read_abc_file(os.path.join(input_path, file))
      else:
        if verbose:
          print('Unsupported file format detected. Skipping ' + file)
        continue
    except:
      if verbose:
        print('Skipping ' + file + ' because it failed to load.')
      continue
    if not notes:
      continue
    
    for index in range(len(notes['pitch'])):
      if notes['pitch'][index] not in pitch_dict:
        pitch_dict[notes['pitch'][index]] = 1
      else:
        pitch_dict[notes['pitch'][index]] += 1

      if notes['step'][index] not in step_dict:
        step_dict[notes['step'][index]] = 1
      else:
        step_dict[notes['step'][index]] += 1

      if notes['duration'][index] not in dur_dict:
        dur_dict[notes['duration'][index]] = 1
      else:
        dur_dict[notes['duration'][index]] += 1

  pitch_set = get_sorted_keys(pitch_dict, top_k)
  step_set = get_sorted_keys(step_dict, top_k)
  dur_set = get_sorted_keys(dur_dict, top_k)

  pitch_dict = {pitch_set[i]: i for i in range(len(pitch_set))}
  step_dict = {step_set[i]: i for i in range(len(step_set))}
  dur_dict = {dur_set[i]: i for i in range(len(dur_set))}

  with open(os.path.join(output_path, 'pitch_dict.txt'), 'w') as f:
    f.write(str(pitch_dict))
  with open(os.path.join(output_path, 'step_dict.txt'), 'w') as f:
    f.write(str(step_dict))
  with open(os.path.join(output_path, 'dur_dict.txt'), 'w') as f:
    f.write(str(dur_dict))

def load_dictionary(path):
  with open(path, 'r') as f:
    return eval(f.read())
  
def get_reverse_dictionary(dict):
  return {v: k for k, v in dict.items()}

if __name__ == "__main__":
  convert_midi_files_to_abc(source_path, processed_path)
  generate_dictionaries(processed_path, checkpoint_path, top_k)