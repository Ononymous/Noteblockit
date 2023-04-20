import torch
import torchaudio
import streamlit as st
import io

import utils
import predict


# !python openunmix/cli.py \
#   gdrive/MyDrive/AudioSeparation/songs/Another.wav \
#   --model gdrive/MyDrive/AudioSeparation/gen_model_new/ \
#   --outdir gdrive/MyDrive/AudioSeparation/output/ \
#   --targets vocals drums bass \
#   --residual other \
#   --verbose


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


def separate(audio, rate):
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
        
        # Convert the audio_data to WAV format with the correct sample width
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
        
        st.write(f"Playing {target}:")
        wav_buffer.seek(0)  # Reset buffer position to the start
        st.audio(wav_buffer, format='audio/wav')

st.title('Noteblockit Demo')
st.image("./noteblock.png")

st.header('Helps separate audio into 4 tracks: vocals, drums, bass, and other, and combine them into a MIDI file.')
uploaded_file = st.file_uploader(
    label="Choose a wav file to separate",
    type="wav",
    accept_multiple_files=False,
    key="audio_file_uploader",
    help="Upload a wav file to separate into 4 tracks",
)

if uploaded_file is not None:
    max_duration = 30  # Maximum allowed duration in seconds
    
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