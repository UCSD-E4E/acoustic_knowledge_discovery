# # %%
# %load_ext autoreload
# %reload_ext autoreload
# %autoreload 2

# %%
import librosa
import numpy as np    
import os
import signal
import subprocess
import json
import uuid
from IPython.display import HTML

# %% [markdown]
# Extractor (Input Validation, Chunking, Mapping From file_csv and annotations_csv to chunks)

# %%
from acoustic_knowledge_discovery import extractors

# %%
extractor = extractors.Extractor(
    #"sample_inputs/file.csv", #file metadata csv
    #"/home/s.kamboj.400/mount/gorongosa_recordings_small_test.csv",
    "/home/s.kamboj.400/mount/mexico_recordings_small.csv",
    #"sample_inputs/anno.csv", #annoations csv
    #"/home/s.kamboj.400/mount/gorongosa_recordings_anno.csv",
    "/home/s.kamboj.400/mount/mexico_recordings_anno.csv",
    #"sample_inputs/inputDtory",#input directory with a files subdirectory for audio files & a XC-templates subdirectory for XC templates
    #"/home/s.kamboj.400/mount/",
    "/home/s.kamboj.400/unzipped-coral/PaolaMexico/",
    chunk_size = 5) #chunk size in seconds
chunkDS = extractor.forward()
print("Finished initializing chunkDS")


# %%
# for row in chunkDS.chunk_ds["train"]:
#     print(row)
# chunkDS.chunk_ds

# %% [markdown]
# Chunk DS Augmentations

# %%
# def add_raw_audio_batched(batch):
#     raws = []
#     for path, start in zip(batch["file_path"], batch["chunk_start"]):
#         # os.fspath handles both str and Path
#         #print("os.fspath(path) is ", os.fspath(path))
#         y, _ = librosa.load(
#             os.fspath(path),
#             offset=float(start),
#             duration=float(extractor.chunk_size),
#             sr=None         # keep original sampling rate     
#         )
#         raws.append(y.astype(np.float32))

#     return {"raw_audio": raws}


# %%
# # Add raw audio column in dataset
# chunkDS.chunk_ds = chunkDS.chunk_ds.map(add_raw_audio_batched, 
#                                         batched=True, 
#                                         num_proc=1) #increasing this number a small amount can make it faster
#                                         # ONLY INCREASE ON MAC OR LINUX

# %%
# chunkDS.chunk_ds #make sure that new column raw_audio is added

# %% [markdown]
# File-level Operations

# %%
# from acoustic_knowledge_discovery.features.feature_preprocessing_pipeline import FeatureSequential
# from acoustic_knowledge_discovery import features

# THRESHOLD = 0.6

# #TODO BUG: Change template matching code so that it reads audio files from file_path instead of extractor.base_dir / "files". 
# # Right now, template matching does NOT work if the audio files are not in extractor.base_dir / "files" (which most probably will not be!)
# pipeline = FeatureSequential(
#     features.TemplateMatching(
#         extractor.base_dir / "files",
#         extractor.base_dir / "XC-templates",
#         THRESHOLD,
#         chunk_size=extractor.chunk_size
#     ),
#     # features.anotherFunctionThatInheretsFromFeaturePreprocessor
# )

# chunkDS = pipeline.forward(chunkDS)  

# %%
# #verify properly loaded after feature extraction
# for row in chunkDS.chunk_ds["train"]:
#     print(row["Annotation"], row["Confidence"])

# %% [markdown]
# Chunk-level Operations

# %%
#Insert EGCI
from acoustic_knowledge_discovery import features

print("Starting EGCI computation...")
egci = features.EGCI()
chunkDS= egci(chunkDS, extractor.chunk_size, num_proc=1)
print("Finished EGCI")


# %%
#verify EGCI column properly added
# for row in chunkDS.chunk_ds["train"]:
#     print(row["chunk_id"], row["EGCI"])
# chunkDS.chunk_ds

# %% [markdown]
# Binning!

# %%
from acoustic_knowledge_discovery.postprocessing import MakeBins2dFloat
makebins = MakeBins2dFloat(chunkDS)
egci_bin={
    "low": ( (0.0, 34), (0.0,0.34) ), 
    "medium": ((0.34,0.67),(0.34,0.67)), 
    "high": ((0.67,1),(0.67,1))
}

print("Starting Binning...")
chunkDS = makebins("EGCI", egci_bin)
print("Finished Binning.")

# # %%
# for row in chunkDS.chunk_ds["train"]:
#     #label_str = chunkDS.chunk_ds["train"].features["EGCI"].int2str(row["EGCI"])
#     print(row["chunk_id"], row["EGCI"])
# chunkDS.chunk_ds

# %%
# chunkDS.chunk_ds["train"][0]

# %%
# # Tests that MakeBins1dFloat does work, but it does not make sense to use it on this dataset
# from acoustic_knowledge_discovery.postprocessing import MakeBins1dFloat
# makebins1d = MakeBins1dFloat(chunkDS)
# OneDBin={
#     "low": (0, 5) , 
#     "medium": (5, 10), 
#     "high": (10, 15)
# }

# chunkDS = makebins1d("chunk_start", OneDBin)

# for row in chunkDS.chunk_ds["train"]:
#     print(row["chunk_id"], row["chunk_start"])
# chunkDS.chunk_ds

# %% [markdown]
# # AutoEncoder and Clustering

# %%
from acoustic_knowledge_discovery.features import AutoEncoderProcessor

print("Starting AutoEncoder processing...")
autoEncoder = AutoEncoderProcessor(model_path = "sample_models/dim_10_muha.pt", num_dims=10)
chunkDS = autoEncoder(chunkDS, extractor.chunk_size)
print("Finished AutoEncoder processing.")
# chunkDS.chunk_ds

# %%
from acoustic_knowledge_discovery.postprocessing import Cluster_API

#HEY HEY HEY: You should note that more num_clusters means more possible nodes
# num_clusters should be some number that is less than your dataset
print("Starting Clustering...")
clustering_pipe = Cluster_API(chunkDS, num_clusters=2)
chunkDS = clustering_pipe("sample_models/dim_10_muha.pt_embeddings")
print("Finished Clustering.")
# chunkDS.chunk_ds

# %% [markdown]
# # Convert System to a graph representation

# %%
def normalize(v):
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):   # convert numpy array to list
        return v.tolist()
    return [v]  # if it is scalar, then make it list

print("Starting graph representation creation...")
# %%
#test = chunkDS.chunk_ds.select_columns(["file_name", " time_of_day", "season", "species", "chunk_id", "Annotation", "EGCI", "sample_models/dim_10_muha.pt_embeddings_clusters"])
test = chunkDS.chunk_ds.select_columns(["file_name", "Site", "Season", "Month of recording", "Country", "Depth (m)", "Human activity", "Type of boats", "chunk_id", "EGCI", "sample_models/dim_10_muha.pt_embeddings_clusters"])
nodes = test["train"].to_pandas().melt()
nodes = nodes.assign(value=nodes["value"].apply(normalize)).explode("value", ignore_index=True) #necessary so list values are separated
nodes["id"] = nodes["variable"] + "_" + nodes["value"].apply(str)
nodes["group"] = nodes["variable"].apply(lambda x: list(nodes["variable"].unique()).index(x))
nodes[["id", "group"]].head()

# %%
edge_list = test["train"].to_pandas().melt(id_vars="file_name")
edge_list = edge_list.assign(value=edge_list["value"].apply(normalize)).explode("value", ignore_index=True)
edge_list = edge_list.dropna(subset=["value"])# drop Nan
edge_list["target"] = edge_list["variable"] + "_" + edge_list["value"].apply(str)
edge_list["source"] = "file_name" + "_" + edge_list["file_name"]
edge_list["value"] = 1
edge_list[["source", "target", "value"]].head()

# %%
import json

with open("acoustic_knowledge_discovery/d3-visualization/graph_representation.json", mode="w") as f:
    json.dump({
        "nodes": list(nodes[["id", "group"]].drop_duplicates().T.to_dict().values()),  
        "links":  list(edge_list[["source", "target", "value"]].T.to_dict().values()),
    }, f
)
    
print("Finished graph representation creation.")

# # %% [markdown]
# # # Visualization

# # %%
# !npm install

# # %%
# command = ["npx", "http-server", "-p", "8080"] 
# target_directory = "acoustic_knowledge_discovery/d3-visualization/"

# process = subprocess.Popen(
#     command,
#     cwd=target_directory,
#     #shell=True,
#     preexec_fn=os.setsid  
# )

# # Persist the group leader PID so you can kill it later even after kernel restarts & kill the entire tree
# with open("/tmp/http_server_pid.json", "w") as f:
#     json.dump({"pgid": process.pid}, f)

# print(f"Server started. PGID={process.pid}")

# # %%
# #NOTE : You can also open up http://127.0.0.1:8080 in your browser to see it displayed on a larger screen
# HTML(f"""
# <iframe style="background-color: white;" 
#         src="http://127.0.0.1:8080/?nocache={uuid.uuid4()}" 
#         width="1200" height="1000"></iframe>
# """)

# # %%
# # Kill the entire process group (gracefully) to stop d3vis port
# with open("/tmp/http_server_pid.json") as f:
#     pgid = json.load(f)["pgid"]


# os.killpg(pgid, signal.SIGTERM)

# # %%
# #process.kill()

# # %%



