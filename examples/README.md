
### A typical example workflow would consist of:

 - generate_data.py: generate some random sequence/structure training pairs using RNAfold

 - train.py: select one of the models from rnadeep/models.py (currently those are all SPOTRNA reimplementations.) and train with your generated test/validation data.

 - predicty.py: either test the performance of the model on different data or generate the output matrices for each sequence.

 - mlforensics.py: do the postprocessing of generating a secondary structures from output matrices.

