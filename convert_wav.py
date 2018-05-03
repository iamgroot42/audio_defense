from pydub import AudioSegment
import glob
import os
files = glob.glob("cv_corpus_v1/cv-valid-test/sample-000*")
for i in files:
	print i
	sound = AudioSegment.from_mp3(i)
	head, tail = os.path.split(i)
	newname = "commonvoice_subset2/"+os.path.splitext(tail)[0]+'.wav'
	sound.export(newname, format="wav")
