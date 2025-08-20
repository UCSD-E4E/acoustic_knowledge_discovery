#from ABC import abstract
from abc import ABC, abstractmethod
from ..dataset import KnowledgeDataset
from datasets import load_dataset
from pathlib import Path
from typing import Union

class Extractor():
    """Extracts some data and formats into useful output

    Takes raw data (such as an ecological dataset like kaleidoscope or raven)
    Outputs a KnowledgeDataset
    See `../dataset.py`
    """

    # def __init__(
    #         filename_col:str, ....) #Going to add stuff about files here
    def __init__(
            self,
            files_csv: Union[str, Path], #metadata csv for files
            annos_csv: Union[str, Path], #annotations (ie timestamp) for each file 
            base_dir: Union[str, Path] #directory where files are stored
        ):
        """Initialize the Extractor with file paths and columns"""
        try:
            self.base_dir = Path(base_dir).resolve()
        except Exception as e:
            raise ValueError(f"base_dir {base_dir} produced error: {e}")

        try:
            self.file_ds = load_dataset("csv", data_files=files_csv)["train"]
        except Exception as e:
            raise ValueError(f"files_csv {files_csv} produced error: {e}")
        
        try:
            self.anno_ds = load_dataset("csv", data_files=annos_csv)["train"]
        except Exception as e:
            raise ValueError(f"annos_csv {annos_csv} produced error: {e}")
        
        #validate that the files and annotations have the required columns
        if "file_path" not in self.file_ds.column_names:
            raise ValueError("files_csv must contain a 'file_path' column")
        required_cols = {"file_path", "offset_time", "end_time"}
        missing = required_cols - set(self.anno_ds.column_names)
        if missing:
            raise ValueError(f"annotations_csv missing required column(s): {missing}")

        # #add ID column to annotations
        # def _make_id(example):
        #     example["ID"] = f"{example['file_path']}_{example['offset_time']}"
        #     return example
        # self.anno_ds = self.anno_ds.map(_make_id)

        print("columns in file_ds:", self.file_ds.column_names)
        print("columns in anno_ds:", self.anno_ds.column_names)


    def forward(self) -> KnowledgeDataset:
        """Run the extractor and return a KnowledgeDataset"""
        return KnowledgeDataset(file_ds=self.file_ds, anno_ds=self.anno_ds)
