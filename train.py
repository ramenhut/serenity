
'''Trains a transformer network to generate MIDI content.'''

from datetime import datetime
import midi
import model
import numpy as np
import os
import process
import random
import tensorflow as tf 

model_filename = 'model.ckpt'
epoch_count = 1200
batch_size = 32
max_window_count = 16
resume_training = False
train_pitch = False
train_step = False
train_duration = False

def compute_steps_per_epoch(files):
  '''Returns the number of iterations required per epoch.'''
  sample_count = 0
  for filename in files:
    try:
      notes = midi.read_abc_file(os.path.join(process.processed_path, filename))
    except:
      continue
    if not notes:
      continue
    note_count = len(notes['pitch'])
    sample_count += min(max_window_count, note_count - process.max_tokens - 2)
  return sample_count / batch_size

def generator(filenames, pitch_dict, step_dict, dur_dict, attribute='pitch', verbose=False):
  '''Continually produces batches of midi data as (encoder_input, decoder_input, decoder_output).'''
  encode_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)
  decode_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)
  output_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)
  cur_samples_in_batch = 0

  while 1:
    for filename in filenames:
      if verbose:
        print('Loading ' + os.path.join(process.processed_path, filename) + '...')
      try:
        notes = midi.read_abc_file(os.path.join(process.processed_path, filename))
        notes = process.convert_notes_to_tokens(notes, pitch_dict, step_dict, dur_dict)
      except:
        if verbose:
          print('Skipping ' + filename + ' because it failed to load.')
        continue
      if not notes:
        if verbose:
          print('Skipping ' + filename + ' because it was empty.')
        continue
      if len(notes[attribute]) < process.max_tokens:
        if verbose:
          print('Skipping ' + filename + ' because it was too short.')
        continue

      window_count = min(max_window_count, len(notes[attribute]) - process.max_tokens - 2)

      if window_count <= 0:
        if verbose:
          print('Training window is too small. Skipping file ' + filename + '.')
        continue

      for i in range(window_count):
        tokens = notes[attribute][i:i + process.max_tokens - 2]
        encode_tokens = tokens[:process.min_encode_tokens]
        decode_tokens = tokens[process.min_encode_tokens:]
        encode_tokens = [process.START_INDEX] + encode_tokens + [process.END_INDEX] + \
          [process.PAD_INDEX] * (process.max_tokens - len(encode_tokens) - 2)
        output_tokens = decode_tokens + [process.END_INDEX] + [process.PAD_INDEX] * \
          (process.max_tokens - len(decode_tokens) - 1)
        decode_tokens = [process.START_INDEX] + decode_tokens + [process.END_INDEX] + \
          [process.PAD_INDEX] * (process.max_tokens - len(decode_tokens) - 2)
        
        encode_batch[cur_samples_in_batch] = np.asarray(encode_tokens)
        decode_batch[cur_samples_in_batch] = np.asarray(decode_tokens)
        output_batch[cur_samples_in_batch] = np.asarray(output_tokens)
        cur_samples_in_batch += 1

        if cur_samples_in_batch == batch_size:
          yield [encode_batch, decode_batch], output_batch
          cur_samples_in_batch = 0
          encode_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)
          decode_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)
          output_batch = np.zeros((batch_size, process.max_tokens), dtype=np.int32)

if __name__ == "__main__":
  if not os.path.exists(process.processed_path):
    print('Run process.py before training.')
    exit()

  pitch_dict = process.load_dictionary(process.checkpoint_path + '/pitch_dict.txt')
  step_dict = process.load_dictionary(process.checkpoint_path + '/step_dict.txt')
  dur_dict = process.load_dictionary(process.checkpoint_path + '/dur_dict.txt')

  pitch_vocab_size = len(pitch_dict) + process.BASE_INDEX_COUNT
  step_vocab_size = len(step_dict) + process.BASE_INDEX_COUNT
  duration_vocab_size = len(dur_dict) + process.BASE_INDEX_COUNT

  target_files = os.listdir(process.processed_path)
  random.shuffle(target_files)
  validation_split = int(len(target_files) * 0.95 + 0.5)
  training_files = target_files[:validation_split]
  validation_files = target_files[len(training_files):]
  assert(len(training_files) > 0 and len(validation_files) > 0)

  training_steps = compute_steps_per_epoch(training_files)
  validation_steps = compute_steps_per_epoch(validation_files)
  pitch_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='pitch')
  pitch_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='pitch')
  step_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='step')
  step_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='step')
  duration_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='duration')
  duration_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='duration')

  if train_pitch:
    print('Training pitch:')

    if resume_training and os.path.exists(process.checkpoint_path + '/pitch_model_weights.index'):
      print('Loading pitch model from checkpoint.')
      pitch_xformer = model.load_model(process.checkpoint_path + '/pitch_model_weights', input_vocab_size=pitch_vocab_size, target_vocab_size=pitch_vocab_size)
    else:
      pitch_xformer = model.create_model(input_vocab_size=pitch_vocab_size, target_vocab_size=pitch_vocab_size)

    best_loss = 1000.0
    for epoch in range(epoch_count):
      now = datetime.now()
      current_time = now.strftime("%H:%M:%S")
      print('[Pitch] Epoch ' + str(epoch) + ' started at:', current_time)
      history = pitch_xformer.fit(pitch_training_gen, validation_data=pitch_validation_gen, 
                                  validation_steps=validation_steps, steps_per_epoch=training_steps, 
                                  epochs=1, verbose=1)
      pitch_xformer.summary()
      random.shuffle(training_files)
      pitch_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='pitch')
      pitch_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='pitch')

      if history.history['loss'][0] < best_loss:
        best_loss = history.history['loss'][0]
        print('Saving pitch model weights...')
        model.save_model(pitch_xformer, process.checkpoint_path + '/pitch_model_weights')

  if train_step:
    print('Training step:')

    if resume_training and os.path.exists(process.checkpoint_path + '/step_model_weights.index'):
      print('Loading step model from checkpoint.')
      step_xformer = model.load_model(process.checkpoint_path + '/step_model_weights', input_vocab_size=step_vocab_size, target_vocab_size=step_vocab_size)
    else:
      step_xformer = model.create_model(input_vocab_size=step_vocab_size, target_vocab_size=step_vocab_size)

    best_loss = 1000.0
    for epoch in range(epoch_count):
      now = datetime.now()
      current_time = now.strftime("%H:%M:%S")
      print('[Step] Epoch ' + str(epoch) + ' started at:', current_time)
      history = step_xformer.fit(step_training_gen, validation_data=step_validation_gen, 
                                 validation_steps=validation_steps, steps_per_epoch=training_steps, 
                                 epochs=1, verbose=1)
      step_xformer.summary()
      random.shuffle(training_files)
      step_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='step')
      step_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='step')

      if history.history['loss'][0] < best_loss:
        best_loss = history.history['loss'][0]
        print('Saving step model weights...')
        model.save_model(step_xformer, process.checkpoint_path + '/step_model_weights')

  if train_duration:
    print('Training duration:')

    if resume_training and os.path.exists(process.checkpoint_path + '/duration_model_weights.index'):
      print('Loading duration model from checkpoint.')
      duration_xformer = model.load_model(process.checkpoint_path + '/duration_model_weights', input_vocab_size=duration_vocab_size, target_vocab_size=duration_vocab_size)
    else:  
      duration_xformer = model.create_model(input_vocab_size=duration_vocab_size, target_vocab_size=duration_vocab_size)

    best_loss = 1000.0
    for epoch in range(epoch_count):
      now = datetime.now()
      current_time = now.strftime("%H:%M:%S")
      print('[Duration] Epoch ' + str(epoch) + ' started at:', current_time)
      history = duration_xformer.fit(duration_training_gen, validation_data=duration_validation_gen, 
                                     validation_steps=validation_steps, steps_per_epoch=training_steps, 
                                     epochs=1, verbose=1)
      duration_xformer.summary()
      random.shuffle(training_files)
      duration_training_gen = generator(training_files, pitch_dict, step_dict, dur_dict, attribute='duration')
      duration_validation_gen = generator(validation_files, pitch_dict, step_dict, dur_dict, attribute='duration')

      if history.history['loss'][0] < best_loss:
        best_loss = history.history['loss'][0]
        print('Saving duration model weights...')
        model.save_model(duration_xformer, process.checkpoint_path + '/duration_model_weights')

  print('Zipping up checkpoint files...')
  process.zip_folder(process.checkpoint_path, model_filename)