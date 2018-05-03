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
          "project_id": "privacy-btp-1479375866301",
          "private_key_id": "20979e786a58f56c2fb4467b8394c98278dc8509",
          "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCZ9l9mZs/7/A1T\nOSL07zXYlYH2GFaf0nVifKGbNGHOVTRhySdCIgrGtAB8woFb8BKnz18RKZ2V+RFX\nBULwVYWcDpFD84dzc/I1gfBQTy9dA1NkP0ccaVuVlu8Dun6Gf9Wc0Ny4NzjjoRuY\nIgnD9+g6AQ5Wr3lsORJiIfumFVs+9q9ZGubftt1ewlC3zgdS1JQCjpNPZiiI3XyP\nPX3+c72yXRSlDmhY/Ey8gcXnR7RCU03l0+EdvNQr8kLEEL8aHcYgurapJzyulAo6\nPcrLZcQPqkp/bJ6VOPR1Jlxpi+XQmYyvR6Cp8RQKd664sufftYc50w/wVXuzEXzP\nbmXfNTIDAgMBAAECggEAO30dNnlyUoh885OnpKFLGnEGQrr7uYw9q/zBCrGuOuXk\nNAZfw2dY55cEezBCgG4eHzX6oyyDxb9zij6vsyIwWnCsg2d1BlCeHTukSbuW7ucK\nkTj79oflCcNEfhnQqpJ5TLrNFebMdfO5sEoyoMRIuCTRUABEN+NDbaR40h79XzGj\nNhhlgqlLIz85t0OXnHDWgA4aiwQpWhGgK1kMVdYn11IQmHlSuYD1Y6Hru5dGuT3D\nP6VQz5wflPPRlqj7n5n0TTSCt9o4nlKeA+8wFgVafMcvXMuCEwezdH0DxGl/0H1J\nHK76QPDR4+w7ePwF5ETxFijf4Va6GdlJr5oJQjTA7QKBgQDSltF/YPJbdCreAkM8\nVKy+fGcTeIvz3DOgk9HjpiDLaoKc2rUURhwD11Lc8SDmrSXIwwsJ7EtN8gCHWvNo\nG7PPYKSniou39ryXFYpg1R87p+p9m99mrJMbKZ2Oi1DLA3kI7EIKBPrNkoGqKCIV\nDTmESB3JP3q/aY9AEu/RFMeq1wKBgQC7KZSRZ2VvFPEpMoiyS3Q6Tr4bHw2PCWOM\nvlIikBA0yCKHqDMXig5hZ69MTaKBfiSsnV9rE7END+ty1DwtHnnGt9Qa2mYTC6XO\n7DduA4uCrUayOZxdJvC5wZwPcyCR3fao9I2HE5H8HX00S4CLFOVeuRhGG31AtPB7\nYF0RwLzYtQKBgC2XEQuuhUlfQNiHTN8Gxc8HR5ljg4jrpxGgbtQF5xuil1w1zPXy\np6X9O6cxXJoT6hYog39GdJcPSSYEfqWPOcIvffX3fH/7HqDmvOpxuS1FEPLYh+jG\ne6Jpw/5UEs2gltdjrnhU06cljIS144sDLeyBYFFtOLmvtJ9+egSdpwxJAoGAdpgP\nWt9Qh8WWVkt+ELP9DuFMVrUji0oguVLzipEMo9VZA+qRjU3edNwVWN0spq7+oB4M\nEzZkDunSgG15QAG6mi9riCRxX4oW43oMWXqHkSMo08/rg58kFCBZOfTyY1tpa3+i\nmj/NVhp9doCNEBQKjy3r8qiCgljktNjHwHMHdq0CgYAJn2GaEXRTg88Q+GzaqswP\nMY1WU1B8/LhAvGZkWGI624fNaTJVFPQf0ywZsTbDAHtvlmqtZfs9HsANUfSSK1KG\nHktvwxNADgSBiMkt3X++RPD/RK7MbLibW8ZZvk90OUSrTkWKC2f87881ewwj3f4F\nt8VQ4ACpv1gUPsGQ+HLCzQ==\n-----END PRIVATE KEY-----\n",
          "client_email": "333660242266-compute@developer.gserviceaccount.com",
          "client_id": "102052380341366200607",
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
        WIT_AI_KEY = "ZRUWV7QVCTXD2INFMA7HV6JFM7YAUAQF"  # Wit.ai keys are 32-character uppercase alphanumeric strings
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
        BING_KEY = "2b6c9511608a49eb9d9c249851837fc9"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
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
        # HOUNDIFY_CLIENT_ID = "pvHBCo3qjsF19lKPZsuGDw=="  # Houndify client IDs are Base64-encoded strings
        # HOUNDIFY_CLIENT_KEY = "XaqOyvaXa6q7TfvY9eEy9bmlELX13zi_HMolTAT2OUN5PJY5KKTpO92i1A8gT3katPvLFHarX5AArGZqAcZSIQ=="  # Houndify client keys are Base64-encoded strings
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
        IBM_USERNAME = "ac8f6b68-68c3-44a5-a522-60d9729b2889"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        IBM_PASSWORD = "tE14z2c5aNPH"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
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

