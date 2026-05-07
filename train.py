# %%
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
    "/home/c.zhang.97/mount/gorongosa_recordings.csv", #file metadata csv
    #"/home/s.kamboj.400/mount/gorongosa_recordings_small_test.csv",
    #"/home/s.kamboj.400/mount/mexico_recordings.csv",
    "/home/c.zhang.97/mount/gorongosa_recordings_anno.csv", #annoations csv
    #"/home/s.kamboj.400/mount/gorongosa_recordings_anno.csv",
    #"/home/s.kamboj.400/mount/mexico_recordings_anno.csv",
    "/home/c.zhang.97/mount/",#input directory with a files subdirectory for audio files & a XC-templates subdirectory for XC templates
    #"/home/s.kamboj.400/mount/",
    #"/home/s.kamboj.400/unzipped-coral/PaolaMexico/",
    chunk_size = 5) #chunk size in seconds
chunkDS = extractor.forward()


# %%
for row in chunkDS.chunk_ds["train"]:
    print(row)
chunkDS.chunk_ds

# %% [markdown]
# File-level Operations

# %%
from acoustic_knowledge_discovery.features.feature_preprocessing_pipeline import FeatureSequential
from acoustic_knowledge_discovery import features

TEMPLATE_MATCHING_THRESHOLD = 0.6

pipeline = FeatureSequential(
    features.TemplateMatching(
        extractor.base_dir / "XC_results", # link to directory with XC templates
        TEMPLATE_MATCHING_THRESHOLD,
        chunk_size=extractor.chunk_size
    ),
)

chunkDS = pipeline.forward(chunkDS)  

# %%
#verify properly loaded after feature extraction
for row in chunkDS.chunk_ds["train"]:
    #print(row["Annotation"], row["Confidence"])
    print(row)

# %% [markdown]
# Chunk-level Operations

# %%
#Insert EGCI
from acoustic_knowledge_discovery import features

# %%
egci = features.EGCI()
chunkDS= egci(chunkDS, extractor.chunk_size, num_proc=1)


# %%
#I commented this out for now; it was causing it to not run...
# FREQUENCY_ACTIVITY_THRESHOLD= 0.5
# frequency_bin={
#     'Low': (20, 200), 
#     'Mid': (200, 2000),
#     'High': (2000, 20000),
# }

# freq= features.Frequency(frequency_bin, FREQUENCY_ACTIVITY_THRESHOLD, extractor.chunk_size)
# chunkDS= freq(chunkDS)
# %%
# # For each chunk, get the loudness
# BIN_DICT = {
#     "Low":    (-80.0, -40.0),
#     "Medium": (-40.0, -25.0),
#     "High":   (-25.0, 10.0),
#     "Other" : (10.0, 3000.0)
# }

# vol_feature = features.Volume(BIN_DICT=BIN_DICT, chunk_size= extractor.chunk_size)
# chunkDS= vol_feature(chunkDS)

# %%
# from acoustic_knowledge_discovery.features import ACI

# BIN_DICT = {
#     "Low": (0.0, 0.5),
#     "Medium": (0.5, 2.0),
#     "High": (2.0, 10.0),
# }

# activity_feature = features.Activity(BIN_DICT=BIN_DICT, chunk_size=extractor.chunk_size,)
# chunkDS = activity_feature(chunkDS)

# %%
# # verify EGCI column properly added
# for row in chunkDS.chunk_ds["train"]:
#     print(row)
chunkDS.chunk_ds

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

chunkDS = makebins("EGCI", egci_bin)

# %%
# for row in chunkDS.chunk_ds["train"]:
    #label_str = chunkDS.chunk_ds["train"].features["EGCI"].int2str(row["EGCI"])
    # print(row["chunk_id"], row['Volume_dB'], row['Volume_bin'])
chunkDS.chunk_ds

# %%
chunkDS.chunk_ds["train"][0]

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

autoEncoder = AutoEncoderProcessor(model_path = "sample_models/dim_10_muha.pt", num_dims=10)
chunkDS = autoEncoder(chunkDS, extractor.chunk_size)
chunkDS.chunk_ds

# %%
from acoustic_knowledge_discovery.postprocessing import Cluster_API

#HEY HEY HEY: You should note that more num_clusters means more possible nodes
# num_clusters should be some number that is less than your dataset
clustering_pipe = Cluster_API(chunkDS, num_clusters=2)
chunkDS = clustering_pipe("sample_models/dim_10_muha.pt_embeddings")
chunkDS.chunk_ds

# %% [markdown]
# # Convert System to a graph representation

# %%
def normalize(v):
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):   # convert numpy array to list
        return v.tolist()
    return [v]  # if it is scalar, then make it list

# %%
#test = chunkDS.chunk_ds.select_columns(["file_name", " time_of_day", "season", "species", "chunk_id", "Annotation", "EGCI", "sample_models/dim_10_muha.pt_embeddings_clusters"])
#TODO : change the column names to match dataset
#"Frequency" used to be here but deleted for now because it was causing errors
test = chunkDS.chunk_ds.select_columns(["file_name", "Season","Time of Day", "chunk_id", "EGCI", "sample_models/dim_10_muha.pt_embeddings_clusters", "Annotation"])
nodes = test["train"].to_pandas().melt()
nodes = nodes.assign(value=nodes["value"].apply(normalize)).explode("value", ignore_index=True) #necessary so list values are separated
nodes["id"] = nodes["variable"] + "_" + nodes["value"].apply(str)
nodes["group"] = nodes["variable"].apply(lambda x: list(nodes["variable"].unique()).index(x))
nodes[["id", "group"]].head()

# %%
#connect to file name rather than chunk name
# edge_list = test["train"].to_pandas().melt(id_vars="file_name")
# edge_list = edge_list.assign(value=edge_list["value"].apply(normalize)).explode("value", ignore_index=True)
# edge_list = edge_list.dropna(subset=["value"])# drop Nan
# edge_list["target"] = edge_list["variable"] + "_" + edge_list["value"].apply(str)
# edge_list["source"] = "file_name" + "_" + edge_list["file_name"]
# edge_list["value"] = 1

edge_list = test["train"].to_pandas().melt(id_vars=["chunk_id"])
edge_list = edge_list.assign(value=edge_list["value"].apply(normalize)).explode("value", ignore_index=True)
edge_list = edge_list.dropna(subset=["value"])# drop Nan
edge_list["target"] = edge_list["variable"] + "_" + edge_list["value"].astype(str)
edge_list["source"] = "chunk_id" + "_" + edge_list["chunk_id"]
edge_list["value"] = 1
edge_list[["source", "target", "value"]].head()

# %%
from acoustic_knowledge_discovery.features.recommender_prune import recommender_prune

# %%
# Build chunk embedding dictionary
emb_column = "sample_models/dim_10_muha.pt_embeddings"
chunk_embs = {
    str(row["chunk_id"]): row[emb_column]
    for row in chunkDS.chunk_ds["train"]
}

pruned_edges = recommender_prune(
    edge_list[["source", "target", "value"]],
    chunk_embeddings=chunk_embs,
    keep_ratio=0.5, 
    gamma=0.6
)

print("original edges:", len(edge_list))
print("pruned edges:", len(pruned_edges))

# %%
import json

with open("acoustic_knowledge_discovery/d3-visualization/graph_representation.json", mode="w") as f:
    json.dump({
        "nodes": list(nodes[["id", "group"]].drop_duplicates().T.to_dict().values()),  
        "links":  list(pruned_edges[["source", "target", "value"]].T.to_dict().values()),
    }, f
)

# %% [markdown]
# # Visualization

# %%
#!npm install

# %% [markdown]
# 1. In terminal, type ` cd acoustic_knowledge_discovery/d3-visualization/ `, assuming that you are in the acoustic_knowledge_discovery parent directory
# 2. Run ` npx http-server -p 8080 ` in your terminal*
# 3. Open ` http://127.0.0.1:8080 ` in Safari/Chrome/your choice of browser to interact with the visualization.
# 
# 
# *NOTE: If you are getting the error that npx does not exist, then download it here: https://nodejs.org/en/download

# %% [markdown]
# The following 3 lines of code open up the visualization in the jupyter notebook, but it currently only works for MacOS users. 

# %%
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

# %%
# #NOTE : You can also open up http://127.0.0.1:8080 in your browser to see it displayed on a larger screen
# HTML(f"""
# <iframe style="background-color: white;" 
#         src="http://127.0.0.1:8080/?nocache={uuid.uuid4()}" 
#         width="1200" height="1000"></iframe>
# """)

# %%
# # Kill the entire process group (gracefully) to stop d3vis port
# with open("/tmp/http_server_pid.json") as f:
#     pgid = json.load(f)["pgid"]


# os.killpg(pgid, signal.SIGTERM)

# %%
#process.kill()

# %%



