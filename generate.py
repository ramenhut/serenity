
'''Generates a new MIDI file based on a sample MIDI.'''

import argparse
import collections
import midi
import model
import numpy as np
import os
import process
import train
import tensorflow as tf

from datetime import datetime

prompt_length = process.min_encode_tokens
temperature, top_k = 1.0, 1
neutral_step_value = 0.6
neutral_duration_value = 1.0
minimum_step_value = 0.2

assert(prompt_length >= process.min_encode_tokens and prompt_length < process.max_tokens)

if os.path.exists(train.model_filename) and (
  not os.path.exists(process.checkpoint_path) or 
  os.path.getmtime(process.checkpoint_path) > os.path.getmtime(train.model_filename)):
  print('Unzipping checkpoint file...')
  process.unzip_folder(train.model_filename, './')

if not os.path.exists(process.checkpoint_path):
  print('You must run train.py or download a checkpoint before you can generate music.')
  exit()

pitch_dict = process.load_dictionary(process.checkpoint_path + '/pitch_dict.txt')
pitch_rev_dict = process.get_reverse_dictionary(pitch_dict)
step_dict = process.load_dictionary(process.checkpoint_path + '/step_dict.txt')
step_rev_dict = process.get_reverse_dictionary(step_dict)
dur_dict = process.load_dictionary(process.checkpoint_path + '/dur_dict.txt')
dur_rev_dict = process.get_reverse_dictionary(dur_dict)

def softmax(x):
  '''Computes the softmax of a given array.'''
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def select_top_k(x):
  '''Selects the top k values from a given array.'''
  indices = np.argsort(x)[-top_k:]
  probabilities = [x[i] for i in indices]
  return indices, probabilities

def apply_min_step(x):
  '''Restricts a given step to a minimum value.'''
  return max(x, minimum_step_value)

def decode(model, input_tokens, start_token=process.START_INDEX, end_token=process.END_INDEX, pad_token=process.PAD_INDEX, max_tokens=process.max_tokens):
  '''Decodes a sequence of tokens using the given model.'''
  input_tokens = np.array(input_tokens)
  input_tokens = input_tokens.reshape((1, -1))
  input_tokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
  output_tokens = [start_token]

  for _ in range(max_tokens):
    output_tokens = np.array(output_tokens)
    output_tokens = output_tokens.reshape((1, -1))
    output_tokens = tf.convert_to_tensor(output_tokens, dtype=tf.int32)
    predictions = model([input_tokens, output_tokens], training=False)
    predictions = predictions[:, -1:, :]
    predictions = predictions.numpy().flatten() * temperature
    indices, probabilities = select_top_k(predictions)
    probabilities = softmax(probabilities)
    predicted_id = np.random.choice(indices, p=probabilities)
    output_tokens = np.append(output_tokens, predicted_id)
        
    if predicted_id == end_token or len(output_tokens) >= max_tokens - 1:
      break
    
  if output_tokens[-1] != end_token:
    output_tokens = np.append(output_tokens, end_token)

  return output_tokens

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate a new midi file from a sample midi file.')
  parser.add_argument('input', help='path to input sample midi file.')
  parser.add_argument('output', help='path to output midi file (generated).')
  args = parser.parse_args()

  source_filename = args.input
  output_filename = args.output

  pitch_vocab_size = len(pitch_dict) + process.BASE_INDEX_COUNT
  step_vocab_size = len(step_dict) + process.BASE_INDEX_COUNT
  duration_vocab_size = len(dur_dict) + process.BASE_INDEX_COUNT

  neutral_step = process.get_nearest_token(neutral_step_value, step_dict)
  neutral_dur = process.get_nearest_token(neutral_duration_value, dur_dict)

  notes = collections.defaultdict(list)
  output_notes = collections.defaultdict(list)
  pitch_xformer = model.load_model(process.checkpoint_path + '/pitch_model_weights', input_vocab_size=pitch_vocab_size, target_vocab_size=pitch_vocab_size)
  step_xformer = model.load_model(process.checkpoint_path + '/step_model_weights', input_vocab_size=step_vocab_size, target_vocab_size=step_vocab_size)
  duration_xformer = model.load_model(process.checkpoint_path + '/duration_model_weights', input_vocab_size=duration_vocab_size, target_vocab_size=duration_vocab_size)

  notes = midi.read_midi_file(source_filename, verbose=True)

  if len(notes['pitch']) < prompt_length:
    print('Input sequence length is less than prompt length. Exiting...')
    exit()

  notes = process.convert_notes_to_tokens(notes, pitch_dict, step_dict, dur_dict)
  prompt_length = min(prompt_length, len(notes['pitch']))

  print('Generating pitch sequence...')
  encode_tokens = notes['pitch'][:prompt_length]
  encode_tokens = [process.START_INDEX] + encode_tokens + [process.END_INDEX] + \
    [process.PAD_INDEX] * (process.max_tokens - len(encode_tokens) - 2)
  output_tokens = decode(pitch_xformer, encode_tokens)
  output_tokens = [x for x in output_tokens if x != process.PAD_INDEX and x != process.START_INDEX and x != process.END_INDEX]
  output_notes['pitch'] = output_tokens

  print('Generating step sequence...')
  encode_tokens = notes['step'][:prompt_length]
  encode_tokens = [process.START_INDEX] + encode_tokens + [process.END_INDEX] + \
    [process.PAD_INDEX] * (process.max_tokens - len(encode_tokens) - 2)
  output_tokens = decode(step_xformer, encode_tokens)
  output_tokens = [x for x in output_tokens if x != process.PAD_INDEX and x != process.START_INDEX and x != process.END_INDEX]
  output_notes['step'] = output_tokens

  print('Generating duration sequence...')
  encode_tokens = notes['duration'][:prompt_length]
  encode_tokens = [process.START_INDEX] + encode_tokens + [process.END_INDEX] + \
    [process.PAD_INDEX] * (process.max_tokens - len(encode_tokens) - 2)
  output_tokens = decode(duration_xformer, encode_tokens)
  output_tokens = [x for x in output_tokens if x != process.PAD_INDEX and x != process.START_INDEX and x != process.END_INDEX]
  output_notes['duration'] = output_tokens

  if (len(output_notes['pitch']) != len(output_notes['step']) or 
      len(output_notes['pitch']) != len(output_notes['duration'])):
    print('Warning: Generated sequences are not the same length.')

    if len(output_notes['pitch']) < len(output_notes['step']):
      output_notes['step'] = output_notes['step'][:len(output_notes['pitch'])]
    if len(output_notes['pitch']) > len(output_notes['step']):
      output_notes['step'] = output_notes['step'] + [neutral_step] * (len(output_notes['pitch']) - len(output_notes['step']))

    if len(output_notes['pitch']) < len(output_notes['duration']):
      output_notes['duration'] = output_notes['duration'][:len(output_notes['pitch'])]
    if len(output_notes['pitch']) > len(output_notes['duration']):
      output_notes['duration'] = output_notes['duration'] + [neutral_dur] * (len(output_notes['pitch']) - len(output_notes['duration']))

  print('Writing MIDI file...')
  process.convert_tokens_to_notes(output_notes, pitch_rev_dict, step_rev_dict, dur_rev_dict)
  if minimum_step_value > 0:
    output_notes['step'] = [apply_min_step(x) for x in output_notes['step']]
  midi.write_midi_file(output_notes, output_filename)