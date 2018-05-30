#!/usr/bin/env python3

import speech_recognition as sr
import time

# obtain path to "english.wav" in the same folder as this script
import os

perturbed_samples = sorted(os.listdir('../perturbed_subset'))
clean_samples = sorted(os.listdir('../commonvoice_subset'))

d = {}
for sample in clean_samples:
    clean_sample = os.path.join('../commonvoice_subset', sample)
    sample_num = int(sample.split('.')[0].split('-')[1])
    filtered_perturbed_samples = filter(
        lambda x: int(x.split('.')[1].split('_')[1]) == sample_num,
        perturbed_samples
    )
    d_ = d.get(sample_num, {})
    d_[clean_sample] = map(
        lambda x: os.path.join('../perturbed_subset', x),
        filtered_perturbed_samples
    )
    d[sample_num] = d_

for i in xrange(100):
    sample_num = i
    clean_sample = d[sample_num].keys()[0]
    filtered_perturbed_samples = d[sample_num].values()

    s = [clean_sample]
    s.extend(filtered_perturbed_samples[0])
    for j, file in enumerate(s):
        print 'Sample number {}'.format(j)
        if (j == 0):
            print 'Original sample'
        else:
            print 'Generated sample'
        AUDIO_FILE = file

        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file

        # recognize speech using Sphinx
        try:
            prediction_sphinx = r.recognize_sphinx(audio)
            print("Sphinx thinks you said " + prediction_sphinx)
        except sr.UnknownValueError:
            prediction_sphinx = 'UNK'
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            prediction_sphinx = 'UNK'
            print("Sphinx error; {0}".format(e))

        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            prediction_google_speech = r.recognize_google(audio)
            print("Google Speech Recognition thinks you said " + prediction_google_speech)
        except sr.UnknownValueError:
            prediction_google_speech = 'UNK'
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            prediction_google_speech = 'UNK'
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        # recognize speech using Google Cloud Speech
        GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""{
          "type": "service_account",
          "project_id": "ENTER PROJECT ID",
          "private_key_id": "ENTER ID HERE",
          "private_key": "ENTER KEY HERE",
          "client_email": "ENTER CLIENT EMAIL",
          "client_id": "ENTER CLIENT ID",
          "auth_uri": "https://accounts.google.com/o/oauth2/auth",
          "token_uri": "https://accounts.google.com/o/oauth2/token",
          "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
          "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/333660242266-compute%40developer.gserviceaccount.com"
        }"""
        try:
            prediction_google_cloud = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            print("Google Cloud Speech thinks you said " + prediction_google_cloud)
        except sr.UnknownValueError:
            prediction_google_cloud = 'UNK'
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            prediction_google_cloud = 'UNK'
            print("Could not request results from Google Cloud Speech service; {0}".format(e))

        # recognize speech using Wit.ai
        WIT_AI_KEY = "ENTER KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            prediction_wit = r.recognize_wit(audio, key=WIT_AI_KEY)
            print("Wit.ai thinks you said " + prediction_wit)
        except sr.UnknownValueError:
            prediction_wit = 'UNK'
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            prediction_wit = 'UNK'
            print("Could not request results from Wit.ai service; {0}".format(e))

        # recognize speech using Microsoft Bing Voice Recognition
        BING_KEY = "ENTER KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
        try:
            prediction_bing = r.recognize_bing(audio, key=BING_KEY)
            print("Microsoft Bing Voice Recognition thinks you said " + prediction_bing)
        except sr.UnknownValueError:
            prediction_bing = 'UNK'
            print("Microsoft Bing Voice Recognition could not understand audio")
        except sr.RequestError as e:
            prediction_bing = 'UNK'
            print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))

        # recognize speech using Houndify
        # HOUNDIFY_CLIENT_ID = "ENTER ID HERE"  # Houndify client IDs are Base64-encoded strings
        # HOUNDIFY_CLIENT_KEY = "ENTER KEY HERE"  # Houndify client keys are Base64-encoded strings
        # try:
        #     prediction_houndify = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
        #     print("Houndify thinks you said " + prediction_houndify)
        # except sr.UnknownValueError:
        #     prediction_houndify = 'UNK'
        #     print("Houndify could not understand audio")
        # except sr.RequestError as e:
        #     prediction_houndify = 'UNK'
        #     print("Could not request results from Houndify service; {0}".format(e))

        # recognize speech using IBM Speech to Text
        IBM_USERNAME = "ENTER USERNAME"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        IBM_PASSWORD = "ENTER PASSWORD"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
        try:
            prediction_IBM = r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD)
            print("IBM Speech to Text thinks you said " + prediction_IBM)
        except sr.UnknownValueError:
            prediction_IBM = 'UNK'
            print("IBM Speech to Text could not understand audio")
        except sr.RequestError as e:
            prediction_IBM = 'UNK'
            print("Could not request results from IBM Speech to Text service; {0}".format(e))

        if (j > 0):
            files = ['google_cloud_p', 'google_speech_p', 'sphinx_p', 'IBM_p', 'Bing_p', 'wit_p']
        else:
            files = ['google_cloud_c', 'google_speech_c', 'sphinx_c', 'IBM_c', 'Bing_c', 'wit_c']
        predictions = [
            prediction_google_cloud,
            prediction_google_speech,
            prediction_sphinx,
            prediction_IBM,
            prediction_bing,
            prediction_wit
        ]
        files = map(lambda x: '{}.csv'.format(x), files)
        phrases = map(lambda x: x.strip('\n'), open('../target_phrases.txt', 'r').readlines())
        for pnum, save_file in enumerate(files):
            print save_file
            # import pdb
            # pdb.set_trace()
            if (os.path.exists(save_file)):
                f = open(save_file, 'r')
                lines = f.readlines()
                lines = map(lambda x: x.strip(), lines)
                f.close()
            else:
                lines = []
            print 'Previous number of lines:', len(lines)
            str_ = phrases[j - 1] + ': ' + predictions[pnum]
            lines.append(str_)
            f = open(save_file, 'w')
            for line in lines:
                f.write(line + '\n')
            f.close()

            f = open(save_file, 'r')
            lines = f.readlines()
            lines = map(lambda x: x.strip(), lines)
            f.close()
            print 'Updated number of lines:', len(lines)

