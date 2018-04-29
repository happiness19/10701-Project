# 10701-Project
In order to run your own tests, go to audio.py and change the name of preprocessed audio files that you want to classify.
The preprocessed audio files are numpy arrays saved in text format that
contains number_inputs MFCC data of shape (13,469) of different english audio
files.
AR-Arabic, CA-Cantonese, FR-French, GE-German, HI-Hindi, JA-Japanese,
MA-Mandarin, MY-MAlay, RU-Russian, SP-Spanish, IT-Italian, KO-Korean
You can modify hyperparameters in model.py or change number of classes to 
whatever number of classes you want.
At last, when everything's ready, do python3 train.py and you'll see results
printed