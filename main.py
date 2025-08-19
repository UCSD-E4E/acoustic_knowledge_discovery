# Example workflow

from acoustic_knowledge_discovery import RNN_File_FP, EGCI_Chunk_FP, Cluster_Chunk_FP, CoralExtractor, D3Visualizer

# Load some data
extractor = CoralExtractor()
knowledge_ds = extractor("path to data")

# Run the Pipeline
knowledge_ds = RNN_File_FP(knowledge_ds)
knowledge_ds = EGCI_Chunk_FP(knowledge_ds)
knowledge_ds = Cluster_Chunk_FP(knowledge_ds)

# Collect the results
graph_ds_format = knowledge_ds.to_graph_format()

vis = DS3Visualizer(Output_path = "KDSFJLKSD")
vis(graph_ds_format) #Outputs all artificats for running the D3 visualizers
# Does this have interaction with python? Is this part of the process?
# Idk? This may need more thoughts