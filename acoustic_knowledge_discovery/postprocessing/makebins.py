from ..dataset import ChunkDataset
from datasets import  Value, Sequence, ClassLabel
from typing import Dict, Tuple
import numpy as np
from datetime import date

Bounds1D = Tuple[float, float]
Bounds2D = Tuple[Tuple[float, float], Tuple[float, float]]
BoundsDate = Tuple[date, date]


class MakeBins2dFloat():
    """ Convert 2D floats (ie EGCI) to bins """
    def __init__(self, chunk_ds : ChunkDataset): 
        self.chunk_ds= chunk_ds

    def __call__(self, 
                 column_name: str, 
                 bin_dict: Dict[str, Bounds2D], #label is string, value is Bounds2D
                 include_right: bool = False,   # False -> [lo, hi), True -> [lo, hi]. Setting it to True means that the first match is THE match
                ) -> ChunkDataset:
        """Convert continuous features into categorical bins."""

        out_col = f"{column_name}_bin"
        chunk_ds = self.chunk_ds.chunk_ds

        #make sure that column is binnable (aka numeric)
        try:
            chunk_ds = chunk_ds.cast_column(column_name, Sequence(Value(dtype="float64")))
        except Exception as e:
            raise ValueError(f"Column {column_name} is not 2D sequence of numbers and cannot be binned. Original error: {e}")
        
        bin_labels = list(bin_dict.keys()) 
        bin_labels_with_other = bin_labels + ["Other"] 
        # bounds_arr: shape (K, 4) -> [x_lo, x_hi, y_lo, y_hi] per bin label
        bounds_arr = np.array(
            [[bx[0], bx[1], by[0], by[1]] for (bx, by) in (bin_dict[name] for name in bin_labels)],
            dtype=float
        )

        def _batch_fn(batch):
            # batch[column_name] is a list of [x, y] aka all the values in the column for all rows
            arr = np.asarray(batch[column_name], dtype=float)
            #if malformed, then set everything equal to Other  
            if arr.ndim != 2 or arr.shape[1] != 2:
                print("other filled in")
                return {out_col: ["Other"] * len(batch[column_name])}

            #separate x and y coordinates into 1d arrays
            x, y = arr[:, 0], arr[:, 1]
            #get number of bin labels (without "Other") 
            K = bounds_arr.shape[0]
            # match[i, k] will be True if point i falls inside bin k.
            match = np.zeros((x.size, K), dtype=bool)

            # Check each bin; first True wins (by label order)
            for k in range(K):
                x_lo, x_hi, y_lo, y_hi = bounds_arr[k] #bounds for bin k
                if include_right:
                    #[lo, hi]
                    m = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
                    print(f"x is {x}, x_lo is {x_lo}, x_hi is {x_hi} ")
                    print(f"y is {y}, y_lo is {y_lo}, y_hi is {y_hi} ")
                else:
                    #[lo, hi)
                    m = (x >= x_lo) & (x <  x_hi) & (y >= y_lo) & (y <  y_hi)
                    print(f"x is {x}, x_lo is {x_lo}, x_hi is {x_hi} ")
                    print(f"y is {y}, y_lo is {y_lo}, y_hi is {y_hi} ")
                #store into column k
                match[:, k] = m

            any_match = match.any(axis=1) #if matched inside atleast one bin, any_match[i] is true
            first_idx = match.argmax(axis=1)  # first True along K, or 0 if none
            idxs = np.where(any_match, first_idx, len(bin_labels))  #if 0, then its lable is "Other"

            return {out_col: [bin_labels_with_other[i] for i in idxs]}
        
        # Apply to every split in the DatasetDict
        chunk_ds = chunk_ds.map(_batch_fn, batched=True)

        # Cast to categorical with stable label ordering
        class_label = ClassLabel(names=bin_labels_with_other)
        chunk_ds = chunk_ds.cast_column(out_col, class_label)
        
        #delete original column & rename the bin to the original name
        chunk_ds= chunk_ds.remove_columns(column_name)
        chunk_ds= chunk_ds.rename_column(f"{column_name}_bin", column_name)

        # Return a new ChunkDataset wrapper
        return ChunkDataset(chunk_ds)


class MakeBins1dFloat():
    def __init__(self, chunk_ds: ChunkDataset):
        self.chunk_ds= chunk_ds.chunk_ds
    
    def __call__(self,
        column_name: str,
        bin_dict: Dict[str, Bounds1D],   # label -> (lo, hi)
        include_right: bool = False # False -> [lo, hi), True -> [lo, hi]
    ) -> ChunkDataset:      
        out_col = f"{column_name}_bin"

        try:
            chunk_ds = self.chunk_ds.cast_column(column_name, Value(dtype="float64"))
        except Exception as e:
            raise ValueError(
                f"Column {column_name} is not a numeric float and cannot be binned. Original error: {e}"
            )
        
        bin_labels = list(bin_dict.keys())
        bin_labels_with_other = bin_labels + ["Other"]
        #creates array of bounds per bin label
        bounds_arr = np.array(
            [[*bin_dict[name]] for name in bin_labels],
            dtype=float
        )

        def _batch_fn(batch):
            # batch[column_name] is list of floats for this batch
            arr = np.asarray(batch[column_name], dtype=float)

            if arr.ndim != 1:
                return {out_col: ["Other"] * len(batch[column_name])}

            N = arr.size
            K = bounds_arr.shape[0]

            match = np.zeros((N, K), dtype=bool) # if point i falls inside region K then this is true

            for k in range(K):
                lo, hi = bounds_arr[k]
                if include_right:
                    # [lo, hi]
                    m = (arr >= lo) & (arr <= hi)
                else:
                    # [lo, hi)
                    m = (arr >= lo) & (arr < hi)
                match[:, k] = m

            any_match = match.any(axis=1)
            first_idx = match.argmax(axis=1)
            idxs = np.where(any_match, first_idx, len(bin_labels))

            return {out_col: [bin_labels_with_other[i] for i in idxs]}
        

        chunk_ds = chunk_ds.map(_batch_fn, batched=True)
        class_label = ClassLabel(names=bin_labels_with_other)
        chunk_ds = chunk_ds.cast_column(out_col, class_label)

        #delete original column & rename the bin to the original name
        chunk_ds= chunk_ds.remove_columns(column_name)
        chunk_ds= chunk_ds.rename_column(f"{column_name}_bin", column_name)

        return ChunkDataset(chunk_ds)



class MakeBinsDate():
    #bin = { 
    #   "Dry" : ( date(2024, 5, 10), date(2024, 6, 10)),
    #   "Rainy" : ( date (2024, 6, 12), date(2024, 7, 13))
    #}
    def __init__(self, chunk_ds: ChunkDataset):
        self.chunk_ds= chunk_ds.chunk_ds

    def __call__(self,
                column_name: str,
                bin_dict: Dict[str, BoundsDate]):
        pass
        out_col = f"{column_name}_bin"

        try:
            chunk_ds = self.chunk_ds.cast_column(column_name, Value(dtype="float64"))
        except Exception as e:
            raise ValueError(
                f"Column {column_name} is not a numeric float and cannot be binned. Original error: {e}"
            )
        
        bin_labels = list(bin_dict.keys())
        bin_labels_with_other = bin_labels + ["Other"]
    
#now implement for date and time


    



