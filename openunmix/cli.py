import torch
import torchaudio
import streamlit as st
import io
import os
import basic_pitch
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from midiutil import MIDIFile

import utils
import predict


# !python openunmix/cli.py \
#   gdrive/MyDrive/AudioSeparation/songs/Another.wav \
#   --model gdrive/MyDrive/AudioSeparation/gen_model_new/ \
#   --outdir gdrive/MyDrive/AudioSeparation/output/ \
#   --targets vocals drums bass \
#   --residual other \
#   --verbose
midi_tracks = []

instruments = {
    "vocals": 0,  # Piano
    "drums": 9,   # Drums are automatically assigned to Channel 9 in MIDI
    "bass": 32,   # Bass
    "other": 24   # Guitar
}

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

def audio_to_midi(audio_data, rate):
    basic_pitch_model = ICASSP_2022_MODEL_PATH
    _, midi_data, _ = predict(audio_data, rate, basic_pitch_model)
    return midi_data

def create_merged_midi(midi_tracks, instruments, output_file):
    merged_midi = MIDIFile(len(midi_tracks))

    for idx, (track, instrument) in enumerate(zip(midi_tracks, instruments)):
        merged_midi.addTrackName(idx, 0, track)
        merged_midi.addProgramChange(idx, 0, 0, instrument)

        for note in track.notes:
            merged_midi.addNote(idx, 0, note.pitch, note.start, note.end, note.velocity)

    with open(output_file, "wb") as output_midi:
        merged_midi.writeFile(output_midi)
        

def separate(audio, rate):
    midi_tracks = []
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

        midi_data = audio_to_midi(audio_data, separator.sample_rate)
        midi_tracks.append(midi_data)
        
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

    # Merge MIDI tracks
    output_file = "merged_midi.mid"
    create_merged_midi(midi_tracks, [instruments[target] for target in estimates.keys()], output_file)

    # Display the merged MIDI file
    st.write("Merged MIDI file:")
    st.download_button("Download Merged MIDI", file_path_or_url=output_file, file_name="merged_midi.mid")





st.title('Noteblockit Demo')
st.image("./noteblock.png")

max_duration = 30  # Maximum allowed duration in seconds

st.header('Helps separate audio into 4 tracks: vocals, drums, bass, and other, and combine them into a MIDI file.')
st.write(f"For memory usage limitations on Streamlit apps, only audio less than {max_duration} seconds can be processed.")
st.write(f"Uploaded audio longer than {max_duration} seconds will be trimmed to the first {max_duration} seconds.")

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

        st.write(f"Audio length is longer than {max_duration} seconds. Trimming down to the first {max_duration} seconds.")

    st.write("File uploaded. Separating...")
    separate(audio, rate)
else:
    st.write("No file uploaded.")