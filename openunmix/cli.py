import torch
import torchaudio
import streamlit as st
import io
# import os
# import basic_pitch
# from basic_pitch.inference import predict
# from basic_pitch import ICASSP_2022_MODEL_PATH
# from midiutil import MIDIFile

import utils
import predict


# !python openunmix/cli.py \
#   gdrive/MyDrive/AudioSeparation/songs/Another.wav \
#   --model gdrive/MyDrive/AudioSeparation/gen_model_new/ \
#   --outdir gdrive/MyDrive/AudioSeparation/output/ \
#   --targets vocals drums bass \
#   --residual other \
#   --verbose

# midi_tracks = []

# instruments = {
#     "vocals": 0,  # Piano
#     "drums": 9,   # Drums are automatically assigned to Channel 9 in MIDI
#     "bass": 32,   # Bass
#     "other": 24   # Guitar
# }

class separator_args:
    def __init__(self, model, targets, outdir, ext, start, duration, no_cuda, audio_backend, niter, wiener_win_len, residual, aggregate, filterbank, verbose):
        self.model = model
        self.targets = targets
        self.outdir = outdir
        self.ext = ext
        self.start = start
        self.duration = duration
        self.no_cuda = no_cuda
        self.audio_backend = audio_backend
        self.niter = niter
        self.wiener_win_len = wiener_win_len
        self.residual = residual
        self.aggregate = aggregate
        self.filterbank = filterbank
        self.verbose = verbose

args = separator_args(
    model = "pre-trained/",
    targets = ["vocals", "drums", "bass"],
    outdir = "output/",
    ext = ".wav",
    start = 0.0,
    duration = None,
    no_cuda = False,
    audio_backend = "soundfile",
    niter = 1,
    wiener_win_len = 300,
    residual = "other",
    aggregate = None,
    filterbank = "torch",
    verbose = True
)

# def audio_to_midi(audio_data, rate):
#     basic_pitch_model = ICASSP_2022_MODEL_PATH
#     _, midi_data, _ = predict(audio_data, rate, basic_pitch_model)
#     return midi_data

# def create_merged_midi(midi_tracks, instruments, output_file):
#     merged_midi = MIDIFile(len(midi_tracks))

#     for idx, (track, instrument) in enumerate(zip(midi_tracks, instruments)):
#         merged_midi.addTrackName(idx, 0, track)
#         merged_midi.addProgramChange(idx, 0, 0, instrument)

#         for note in track.notes:
#             merged_midi.addNote(idx, 0, note.pitch, note.start, note.end, note.velocity)

#     with open(output_file, "wb") as output_midi:
#         merged_midi.writeFile(output_midi)
        

def separate(audio, rate):
    # midi_tracks = []
    torchaudio.set_audio_backend(args.audio_backend)
    aggregate_dict = None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    separator = utils.load_separator(
        model_str_or_path=args.model,
        targets=args.targets,
        niter=args.niter,
        residual=args.residual,
        wiener_win_len=args.wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=args.filterbank,
    )

    if rate != separator.sample_rate:
        audio = torchaudio.transforms.Resample(
        orig_freq=rate, new_freq=separator.sample_rate
    )(audio)

    # create separator only once to reduce model loading
    # when using multiple files

    separator.freeze()
    separator.to(device)

    # loop over the files
    estimates = predict.separate(
        audio=audio,
        rate=rate,
        aggregate_dict=aggregate_dict,
        separator=separator,
        device=device,
    )
    for target, estimate in estimates.items():
        audio_data = torch.squeeze(estimate).to("cpu").numpy()

        # midi_data = audio_to_midi(audio_data, separator.sample_rate)
        # midi_tracks.append(midi_data)
        
        wav_buffer = io.BytesIO()
        input_tensor = torch.tensor(audio_data)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.squeeze(0)

        torchaudio.save(
            wav_buffer,
            input_tensor,
            sample_rate=separator.sample_rate,
            format="wav",
            bits_per_sample=16,
        )
        target = "other" if target == "residual" else target
        
        st.write(f"Playing {target}:")
        wav_buffer.seek(0)
        st.audio(wav_buffer, format='audio/wav')

    # # Merge MIDI tracks
    # output_file = "merged_midi.mid"
    # create_merged_midi(midi_tracks, [instruments[target] for target in estimates.keys()], output_file)

    # # Display the merged MIDI file
    # st.write("Merged MIDI file:")
    # st.download_button("Download Merged MIDI", file_path_or_url=output_file, file_name="merged_midi.mid")



st.set_page_config(
    page_title="Noteblockit Demo",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# write header in the following format:
# st.sidebar.markdown('''
# # Sections
# - [Section 1](#section-1)
# - [Section 2](#section-2)
# ''', unsafe_allow_html=True)

st.sidebar.markdown('''
# Sections
- [Home](#home)
- [Music Source Separator](#separator)
- [Midi modifier and merger](#mido)
- [Credits](#credits)
# Other links
- [Github](https://github.com/Ononymous/Noteblockit)
- [Video Demo](https://drive.google.com/file/d/1Is4tc7p6udx7cNKU7OVFPuwBOaj5tm6i/view?usp=sharing)
- [Presentation](https://docs.google.com/presentation/d/19rsL1rv-XUaKtNN9ZmMs2X_wZAFRyLiQj_INUjV1AKA/edit?usp=sharing)
''', unsafe_allow_html=True)

st.header('Noteblockit Demo', anchor="home")
st.image("./noteblock.png")

max_duration = 30  # Maximum allowed duration in seconds

st.header('Helps separate audio into 4 tracks: vocals, drums, bass, and other, and combine them into a MIDI file.')
st.write(f"For memory usage limitations on Streamlit apps, only audio less than {max_duration} seconds can be processed.")
st.write(f"Uploaded audio longer than {max_duration} seconds will be trimmed to the first {max_duration} seconds.")

st.divider()

st.header("Music Source Separator", anchor="separator",)

uploaded_file = st.file_uploader(
    label="Choose a wav file to separate",
    type="wav",
    accept_multiple_files=False,
    key="audio_file_uploader",
    help="Upload a wav file to separate into 4 tracks",
)

if uploaded_file is not None:
    
    audio, rate = torchaudio.load(uploaded_file, format="wav")
    num_samples = audio.shape[-1]
    duration = num_samples / rate

    if duration > max_duration:
        max_samples = int(max_duration * rate)

        if len(audio.shape) == 2:
            # Trim the audio tensor for stereo (2 channels)
            audio = audio[:, :max_samples]
        elif len(audio.shape) == 1:
            # Trim the audio tensor for mono (1 channel)
            audio = audio[:max_samples]

        st.warning(f"Audio length is longer than {max_duration} seconds. Trimming down to the first {max_duration} seconds.", icon=None)
    st.write("File uploaded.")
    with st.spinner('Running separation...'):
        separate(audio, rate)
    st.success('Done!')
else:
    st.write("No file uploaded.")

st.divider()

st.header("Code for Midi modifier and merger", anchor="mido",)

st.write("Use the audio-to-midi converter of your choice to convert the output wav files into MIDI files.")
st.write("Name the MIDI files as follows: bass.mid, vocals.mid, drums.mid, other.mid")

st.code("pip install mido")

st.write("This code requires mido to be installed.")

st.code("""
import mido
from mido import MidiFile, MidiTrack, Message

# Load individual MIDI files
bass = MidiFile('bass.mid')
vocal = MidiFile('vocals.mid')
drums = MidiFile('drums.mid')
other = MidiFile('other.mid')

# Assign new MIDI programs (instruments)
# Replace the numbers with the desired instrument numbers (0-127)
bass_program = 32
vocal_program = 0
drums_program = 0
other_program = 24

bass_channel = 0
vocal_channel = 1
drums_channel = 9   # Channel 10 in human-readable form (0-based indexing)
other_channel = 3

# Create a function to insert a program change at the beginning of a track
def set_channel_and_program(track, channel, program):
    new_track = MidiTrack()
    new_track.append(Message('program_change', channel=channel, program=program, time=0))
    for msg in track:
        if not msg.is_meta:
            msg.channel = channel
        new_track.append(msg)
    return new_track

# Change instruments and prepare tracks for merging
def merge_tracks(tracks):
    merged_track = MidiTrack()
    for track in tracks:
        for msg in track:
            merged_track.append(msg)
    return merged_track

# Merge multiple tracks for each instrument
bass_merged = merge_tracks(bass.tracks)
vocal_merged = merge_tracks(vocal.tracks)
drums_merged = merge_tracks(drums.tracks)
other_merged = merge_tracks(other.tracks)

# Set channel and program for the merged tracks
bass_track = set_channel_and_program(bass_merged, bass_channel, bass_program)
vocal_track = set_channel_and_program(vocal_merged, vocal_channel, vocal_program)
drums_track = set_channel_and_program(drums_merged, drums_channel, drums_program)
other_track = set_channel_and_program(other_merged, other_channel, other_program)

# Combine tracks
merged_mid = MidiFile()
merged_mid.tracks.extend([bass_track, vocal_track, drums_track, other_track])

# Save the merged MIDI file
merged_mid.save('merged.mid')
""")

st.write("Input the merged.mid file into NoteBlock Studio to create the noteblock music.")

st.header("Thanks for using Noteblockit!", anchor="credits")
st.write("Created by Gen Tamada, Christy Yu, and Frank Zhong for UCSB Data Science Club 2022-2023")