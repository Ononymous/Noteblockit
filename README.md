# Noteblockit

## Description

Noteblockit is a project that turns any music file like an mp3 and convert it into a noteblock system in game. 

We developed a recurrent neural network model that converts an mp3 file into a midi file, which is a digital music score sheet storing data on playable sheet music. 

Mp3 to midi conversions only work with simple single-melody music files, but by developing a model that can seperate the mp3 spectrograms into four main instruments using filters, and merging the resulting 4 midi files into 1, we allow more complex songs to be converted into midi files.

[link to demo](https://tinyurl.com/noteblockit)

## Stats and Parameters of resulting model
| Parameter                    | Value                       |
|------------------------------|-----------------------------|
| Sample rate of audio (Hz)    | 16,000 (44kHz for wav file) |
| Max number of epochs         | 1000                        |
| Batches per epoch            | 344                         |
| Audio samples per batch      | 16                          |
| Length of each audio sample  | 6 sec                       |
| Number of frequency bins     | 4096                        |
| Loss function                | Element-wise Mean Squared Error |
| GPU used for training        | NVIDIA A100                 |

## How to approach this problem:
 - Minecraft Noteblock music has a lot of limitations
   - It can only play notes at a certain speed, so if a music has notes that are too close to each other, either have to make the whole music slower, or ignore that note.
   - It only has a very small range of pitch, which means notes that are too high or too low need to be transposed (move to middle)
   - It has 16 different musical instruments, but each of their notes are only about 0.5 seconds long, so can't use them as datasets of some sorts.
 - Finally decided to rely on Note Block Studio, which can help us solve the first 2 problems with just some clicks of button
   - It would be too much work coding a minecraft mod together with a machine learning model.
 - So now our task is reduced to audio -> (MIDI of multiplle instruments) instead of audio -> (minecraft world with prebuilt noteblock redstone structure for that audio)

- A question related to a previous problem: is it possible to make all 16 instruments of minecraft play their part?
   - Yes! It is possible. In fact, somebody already created the model. This model only requires a really short audio sample (2 seconds of audio) for reference, and it can separate the instrument of the reference audio from the mixture audio. And the result is pretty accurate!
   - This means we can make short audio samples of Minecraft instruments (about 2 seconds), and run this model 16 times to get all the different instruments' tracks!
   - However, this model is WAY too complex for us to understand, too difficult to replicate without just straight up copying all the code, and running this model once is really slow/
   - It took 5 minutes for separating just 1 source from a 3 minute mixture audio! Imagine running it 16 times for the same audio.
   - So, it is NOT a viable solution.
   - Github: https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation
   - Demo: https://replicate.com/retrocirce/zero_shot_audio_source_separation

## Data collection:
 - Minecraft has 16 different instruments, but some of them are unique, and we couldn't find a data set that contains all 16 of the audio samples
 - Thought about only doing a single audio-to-MIDI converter, which converts all notes to just piano notes, but at that point what is the point of the minecraft noteblock music?
   - Also, if we put the whole music audio in an audio-to-MIDI converter, there will be too much noise for the model to be accurate
 - Finally decided to separate the audio into different instruments, then put each into audio-to-MIDI converter. It would be much easier to combine individual MIDI files together after labeling their instruments.
   - Luckily, Note Block Studio accepts MIDI files
 - Decided to use the benchmark dataset MUSDB18, which most of the famous music-source-separation models use for training
   - Although it only has 4 instruments (bass, drum, vocal, other), and way less compared to the 16 instruments of minecraft, it allows us to create stable model/models that can accurately separate the sources for the audio-to-MIDI converter to work its magic.

## Model choice:
 - First question: which input should we put into the model? Waveform directly, or masks on spectrogram?
   - According to recent papers, inputting waveform directly into the model is getting better results compared to the spectrogram approach, but there are more research and tutorials online about the spectrogram approach than the waveform approach.
   - After finding a good tutorial about the spectrogram approach, decided to go with the spectrogram approach.
   - Also, the spectrogram approach is so much more intuitive and easy to understand compared to the waveform
 - Second question: which layer should we use as the back bone of the model (convolutional, or recurrent)
   - Convolutional neural network is really good at processing images, in this case the spectrogram of the audio. There are lots of benchmarks models that use this layer as the back bone, and achieve high scores. And also because of its natural, convolutional layers are much faster to train compared to recurrent layers. However, it is not without consequences. One of them being, convolutional layers only take inputs of a fixed size. This means that the input spectrogram needs to be cut into pieces, and get inputted into the model one by one. The model will have no memory of the piece it got previously, and it can not reference a previous sample.
   - Recurrent neural network is really good at processing data that comes in one-by-one. What makes a recurrent layer special is that it has outputs that connects back to itself. This means it can "remember" some of the things from the previous run, and use that as reference for processing the next input. This is really powerful, because now we can input the spectrogram column by column, one at a time, and the recurrent layer will have "memory" of what kind of spectrogram passed through previously. However, this is also not without drawbacks. Because it has more a lot more complex connections compared to a convolutional layer, it takes much more time to train a model. Also, a naive implementation of recurrent layer would tend to encounter a problem, which the layer's memory will zero-out after running for a while. This means that it will "forget" all the references it needs to process the next input. One solution/implementation that fixes this issue is called Long-Short Term Memory (LSTM), and most of the benchmark models that use the recurrent approach uses the LSTM layer. Not just any LSTM layer, but Bidirectional LSTM layers, which reads inputs from the both ends of the spectrogram instead of just from the beginning.

## Reference choice:
 - Decided to use other benchmark models that has a public code base as reference, for creating a model structure from scratch is not really realistic for data science beginners.
 - Had multiple choices at first:
   - Zero Shot Audio Source Separation (too complex, and takes too long to run)
   - Music Source Separation with Band-split RNN (a very new concept, although it uses RNN, too complex and too little resources)
   - Demucs (uses multiple encoder and decoder layers, called the U-Net structure, and has a 2 LSTM layers as its back bone)
   - Hybrid Demucs (very unique in the way that it uses both spectrogram and waveform was inputs to its model, and use them both)
   - Spleeter (available as a python package, one of the first models we tried. Somehow can also separate piano, even though it is not available on MUSDB18 database)
   - Wave-U-Net (the famous model structure that inspired multiple later models. Uses convolution layers as U-Net structure)
   - Open-Unmix (has a relatively simple model structure compared to the others, but still is ranked 13 in the benchmark test of MUSDB18. Inplemented in PyTorch, which is available and known to the public. Also has an easy to understand doc, probably because goal of Open-Unmix is to "providing a reference implementation based on deep neural networks")
 - Decided to go with Open-Unmix, for it is producing accurate results with a relatively easy to understand model structure.

 ## Training problems:
 - Training was SUPER slow the first time
    - Took about 10 minutes for each batch
    - Would take 10 x 344 x 1000 = 3,440,000 minutes for the whole 1000 epochs!
 - Problems with dataset
    - Compressed, and data loader was the bottleneck for the training
 - Switched to a uncompressed dataset of MUSDB18-HQ
    - Success! Only took 50 seconds per batch