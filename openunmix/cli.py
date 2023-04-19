from pathlib import Path
import torch
import torchaudio
import json
import numpy as np
import tqdm
import streamlit as st

import utils
import predict
import data


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
    audio_backend = None,
    niter = 1,
    wiener_win_len = 300,
    residual = "other",
    aggregate = None,
    filterbank = "torch",
    verbose = True
)

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
@st.cache


def separate(audio, rate):

    if args.verbose:
        print("Using ", device)
    # parsing the output dict
    aggregate_dict = None

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
        st.write(f"Playing {target}:")
        st.audio(audio_data, format='audio/wav', sample_rate=separator.sample_rate)


uploaded_file = st.file_uploader(
    label="Choose a wav file to separate",
    type="wav",
    accept_multiple_files=False,
    key="audio_file_uploader",
    help="Upload a wav file to separate into 4 tracks",
)

if uploaded_file is not None:
    audio, rate = torchaudio.load(uploaded_file, format="wav")
    st.write("File uploaded.")
    separate(audio, rate)
else:
    st.write("No file uploaded.")