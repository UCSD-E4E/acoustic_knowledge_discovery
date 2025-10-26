#from ABC import abstract
from abc import ABC, abstractmethod
from ..dataset import ChunkDataset
from datasets import Dataset, DatasetDict, Sequence, Value, Audio
from pathlib import Path
from typing import Union
import soundfile as sf
import pandas as pd
import math
import os
import librosa
from collections import defaultdict

def audio_duration_seconds(path: str) -> float:
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return info.frames / float(info.samplerate), float(info.samplerate)
        if info.duration and info.samplerate:  
            return float(info.duration), float(info.samplerate)
    except Exception as e:
        print(
        f"Could not determine duration for: {path}. "
        f"Received this error: {e}"
        )

def split_all_delimited_columns(df, delim="/", except_cols=None):
    def _split(x):
        # keep as is, if list or tuple since that does not have delim
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, str) and delim in x:
            parts = [p.strip() for p in x.split(delim)]
            return [p for p in parts if p]  # drop empty strings
        return [x] if pd.notna(x) else [] #just wrap as list unless it is null. then, just return empty list
    
    def _needs_split(series):
        return series.apply(lambda v: isinstance(v, str) and delim in v).any()
    
    if except_cols is None:
        except_set = set()
    elif isinstance(except_cols, (str, bytes)):
        except_set = {str(except_cols)}
    else:
        except_set = set(except_cols)

    # find columns where any row contains the delimiter
    cols_to_split = [
        c for c in df.columns
        if c not in except_set and _needs_split(df[c])
    ]
    for c in cols_to_split:
        df[c] = df[c].apply(_split)

    return df

def split_label_cell(x: object) -> list[str]:
    if isinstance(x, str) and "/" in x:
        return [p.strip() for p in x.split("/") if p.strip()]
    if isinstance(x, str) and x.strip():
        return [x]
    else:
        return []
    




class Extractor():
    """Extracts some data and formats into useful output

    Takes raw data (such as an ecological dataset like kaleidoscope or raven)
    Outputs a KnowledgeDataset
    See `../dataset.py`
    """

    # def __init__(
    #         filename_col:str, ....) #Going to add stuff about files here
    #TODO : If skip_last is False, then no chunks are inserted. This is a bug.
    def __init__(
            self,
            files_csv: Union[str, Path], #metadata csv for files
            annos_csv: Union[str, Path], #annotations (ie timestamp) for each file 
            base_dir: Union[str, Path], #directory where files are stored
            chunk_size: int = 5, #size of chunks in seconds
            skip_last= True, #Block last chunk less than chunk_size
        ):
        """Initialize the Extractor with file paths and columns"""
        try:
            self.base_dir = Path(base_dir).resolve()
            self.chunk_size = chunk_size
        except Exception as e:
            raise ValueError(f"base_dir {base_dir} produced error: {e}")

        #Validates the files_csv (makes sure required columns are present) & adds duration column to be used later in chunking
        df = pd.read_csv(files_csv)
        if "file_name" not in df.columns:
            raise ValueError("files_csv must contain a 'file_name' column")
        
        #get durations for each file 
        durations = []
        srs = []
        indices_to_keep=[]
        file_paths=[]
        root= self.base_dir / "files"
        #creates a dictionary that maps each unique filename to a list of all pathlib.Path objects with that name within the root directory. makes lookup later O(1)
        name_index = defaultdict(list)
        for p in root.rglob("*"):
            if p.is_file():
                name_index[p.name].append(p)
        #print(f"Indexed {len(name_index)} unique filenames under {root.as_posix()}")


        for idx, row in df.iterrows():
            fn_raw = row.get("file_name")
            if pd.isna(fn_raw) or str(fn_raw).strip() == "":
                raise ValueError("files_csv must contain a 'file_name' column with no missing values")
            
            # # Normalize and strip accidental leading "files/" segments
            # fn = Path(str(fn_raw)).as_posix().lstrip("/")  #as_posix() converts to string & is good for cross platform compatibility
            # if fn.startswith("files/"):
            #     fn = fn[len("files/"):]
            
            # 1) If CSV gives a subpath, try it directly
            direct_path = root / fn_raw
            candidate_path = None
            if direct_path.exists() and direct_path.is_file():
                candidate_path = direct_path
            else:
                # use the index from before to search by base name
                base = Path(fn_raw).name
                hits = name_index.get(base, [])
                if len(hits) >= 1:
                    candidate_path = hits[0] #even if found multiple times, just take the first occurence
                
            if candidate_path is None:
                #raise ValueError(f"File '{fn_raw}' not found anywhere under {root.as_posix()}")
                print(f"File '{fn_raw}' not found anywhere under {root.as_posix()}, removing from df")
                continue

            # TODO handle this bad IO case for the user
            # if Path("files") / Path("files") in file_path:
            #     raise ValueError(f"Instead of {file_path}, just use something like {file_path.replace(Path("files") / Path("files"), "")}")
            
            try: 
                curr_duration, sr = audio_duration_seconds(candidate_path.as_posix()) #.as_posix() converts to string & is good for cross platform compatibility
            except Exception as e:
                print(f"Could not read audio file {candidate_path}, skipping. Error: {e}")
                continue    

            if not (curr_duration > 0):
                print(f"Non-positive duration for: {candidate_path}, skipping")
                continue

            # File has processed correctly if reached here
            durations.append(curr_duration)
            srs.append(sr)
            indices_to_keep.append(idx)
            file_paths.append(candidate_path.as_posix())
        
        #print("indices to keep are ", indices_to_keep)
        df = df.loc[indices_to_keep].copy()
        df['duration'] = durations
        df['SR'] = srs
        df["file_path"] = file_paths

        # create chunks for each file
        chunk_rows = []
        keep_cols = [c for c in df.columns if c != "duration"]
        for _, row in df.iterrows():
            duration = float(row["duration"])
            # start times: 0, 5, 10, … (start < duration)
            n_chunks = max(1, math.ceil(duration / chunk_size))
            starts=[]
            for start in range(0, n_chunks * chunk_size, chunk_size):
                if start < duration and (
                    start + chunk_size < duration and skip_last
                ):
                    starts.append(start)

            for start in starts:
                rec = {k: row[k] for k in keep_cols} #insert the columns to keep from the files_csv
                rec["chunk_start"] = start
                rec["chunk_id"] = f"{str(row["file_name"])}_{start}"
                rec["Annotation"] = []
                rec["Confidence"] = []
                rec["SR"] = row["SR"]
                chunk_rows.append(rec)

        chunk_df = pd.DataFrame(chunk_rows)

        # read & validate annos_csv 
        annos = pd.read_csv(annos_csv)
        required = ["file_name", "offset_time", "end_time_of_annotation", "annotation", "confidence"]
        missing = [c for c in required if c not in annos.columns]
        if missing:
            raise ValueError(f"annos_csv is missing columns: {missing}")
        # ensure types
        annos["offset_time"] = annos["offset_time"].astype(float)
        annos["end_time_of_annotation"] = annos["end_time_of_annotation"].astype(float)
        annos["confidence"] = annos["confidence"].astype(float)

        # attach annotations to overlapping chunks. it can be attached to multiple chunks
        # Overlap rule: (anno_end > chunk_start) AND (anno_start < chunk_end)
        for fname, group in annos.groupby("file_name"):
            mask = (chunk_df["file_name"] == fname)
            idxs = chunk_df.index[mask]
            if len(idxs) == 0:
                continue

            # for each chunk of this file, find overlapping annotations
            for i in idxs:
                cs = float(chunk_df.at[i, "chunk_start"]) #chunk_start
                ce = cs + chunk_size #chunk_end
                overlapping = group[(group["end_time_of_annotation"] > cs) & (group["offset_time"] < ce)]
                if overlapping.empty:
                    continue

                labels: list[str] = []
                confs: list[float] = []
                for _, a in overlapping.iterrows():
                    parts = split_label_cell(a["annotation"])  # one or many labels
                    if not parts:
                        continue
                    labels.extend(parts)
                    confs.extend([float(a["confidence"])] * len(parts)) #extend the same confidence for all the annotations if needed

                # assign arrays (keeps empty lists if none)
                if labels:
                    chunk_df.at[i, "Annotation"] = labels
                    chunk_df.at[i, "Confidence"] = confs
        
        chunk_df = split_all_delimited_columns(chunk_df, delim="/", except_cols={"file_path"})
        for col in ("Annotation", "Confidence"):
            if col not in chunk_df.columns:
                chunk_df[col] = [[] for _ in range(len(chunk_df))]
            else:
                # ensure list-typed cells (empty list if missing)
                chunk_df[col] = chunk_df[col].apply(
                    lambda v: v if isinstance(v, list) else ([] if pd.isna(v) or v == "" else [v])
                )

        train_ds = Dataset.from_pandas(chunk_df, preserve_index=False)
        train_ds = train_ds.cast_column("Annotation", Sequence(Value("string")))
        train_ds = train_ds.cast_column("Confidence", Sequence(Value("float32")))


    #     # Get the audio
    #     train_ds = train_ds.map(self.get_audio_data)
    #     train_ds = train_ds.cast_column("audio", Audio(sampling_rate=train_ds[0]["audio"]["sampling_rate"]))


        self.chunk_ds = DatasetDict({"train": train_ds})

    # def get_audio_data(self, batch):
    #     file_path = self.base_dir / "files" / batch["file_name"]
    #     y, sr = librosa.load(path=file_path, offset=batch["chunk_start"], duration=self.chunk_size)
    #     batch["audio"] = {
    #         "array": y,
    #         "sampling_rate": sr
    #     }
    #     return batch

    def forward(self) -> ChunkDataset:
        """Run the extractor and return a ChunkDataset"""
        return ChunkDataset(self.chunk_ds)
