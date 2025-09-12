# acoustic_knowledge_discovery
This repository provides an end-to-end pipeline for characterizing large-scale bioacoustic datasets.

## Workflow 
1. **Upload data.** Add your files as described in [Formatting the dataset](#formatting-the-dataset). Now, you follow the instructions in the `train_demo.ipynb` notebook, which follow the workflow listed here. 
2. **Choose chunking size.** Each file is split into x-second segments (x is a user-specified integer).
3. **Run methods.** Choose the methods you want to run over your data from the following: 
* **Template matching**: finds a target pattern (template) within a larger audio file
* **EGCI**: quantifies the complexity and entropy of each audio chunk.

    Currently, the pipeline supports EGCI and template matching; more methods are planned.

4. **Embed & cluster.** Chunks are encoded into 10-dimensional vector embeddings using an autoencoder, then clustered with K-Means. 
5. **Visualize.** Results are converted to a graph and rendered with D3.js, allowing the user to interactively explore the data (e.g., viewing connections, filtering nodes, etc).  

# Required Software
Do not go to "set up" instructions without these softwares

1) Git, install from https://git-scm.com/book/ms/v2/Getting-Started-Installing-Git
2) UV, see install instructions at https://docs.astral.sh/uv/getting-started/installation/
3) Recommend VSCode 

# Install Instructions
1) Open a [terminal](https://code.visualstudio.com/docs/terminal/basics) and run the following commands

```
git clone https://github.com/UCSD-E4E/acoustic_knowledge_discovery.git
uv sync --dev
```

# How To Use

## Formatting the dataset

Use the example in the `sample_inputs` folder as a guide.

### What you need
1. **Two CSV files**
   - `file.csv` — file-level information
   - `anno.csv` — time-stamped annotations (labels)
2. **One directory** that contains your audio files, with this structure:

```text
[DIRECTORY]/
  files/           # REQUIRED: all audio files referenced in the CSVs
  XC-templates/    # OPTIONAL: template audio files (only if using template matching)
```

### More Information
1. `file.csv` (required) : Contains file-level metadata. Must include the following columns:
    - **file_name**: the exact filename of an audio file inside ```[DIRECTORY]/files```

    You may add any other columns (e.g., time_of_day, season, location, etc).
If a single entry has multiple values, separate them with a / (e.g., bat/blip)

2. `anno.csv` (required) : Time-based labels for each file. Must include the following columns:
    - **file_name**: the exact filename of an audio file inside ```[DIRECTORY]/files```
    - **offset_time**: when the annotation starts within that file
    - **end_time_of_annotation**: when the annotation ends within that file
    - **annotation**: what the label refers to (e.g., species name)
    - **confidence**: how sure you are, from `0` (not confident) to `1` (certain)

3. Audio Directory 
    - `files/` (REQUIRED): all audio files referenced by file.csv and anno.csv.
    - `XC-templates/` (OPTIONAL): include template audio here if you plan to run template matching.


## Run the Code
Open the notebook `train_demo.ipynb` and follow the steps outlined in the notebook