# # Example workflow

# from acoustic_knowledge_discovery import EGCI, Extractor, D3Visualizer
# #Cluster_Chunk_FP


# # Load some data
# extractor = Extractor(
#         files_csv="sample_inputs/file.csv",
#         annos_csv="sample_inputs/anno.csv",
#         base_dir="sample_inputs/inputDtory/files"
# )
# knowledge_ds = extractor()

# # Run the Pipeline
# # knowledge_ds = RNN_File_FP(knowledge_ds)
# knowledge_ds = EGCI(knowledge_ds)
# # knowledge_ds = Cluster_Chunk_FP(knowledge_ds)

# # Collect the results
# graph_ds_format = knowledge_ds.to_graph_format()

# vis = D3Visualizer(Output_path = "test_vis")
# vis(graph_ds_format) #Outputs all artificats for running the D3 visualizers
# # Does this have interaction with python? Is this part of the process?
# # Idk? This may need more thoughts