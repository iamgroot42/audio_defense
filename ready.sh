#!bin/bash

# Download audio dataset
wget --no-check-certificate https://nicholas.carlini.com/code/audio_adversarial_examples/commonvoice_subset.tgz
tar -xvzf commonvoice_subset.tgz
wget https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz
tar -xvzf cv_corpus_v1.tar.gz

# Install relevant libraries
pip3 install --user numpy scipy tensorflow-gpu pandas python_speech_features

# CLone Mozilla's DeepSpeech library
git clone https://github.com/mozilla/DeepSpeech.git

# Download DeepSpeech model
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz

# Convert to checkpoint file
python3 make_checkpoint.py

# Generate attack
python3 attack.py --in sample.wav --target "example" --out adversarial.wav

# Install deepspeech
cd DeepSpeech
pip install --user deepspeech-gpu

