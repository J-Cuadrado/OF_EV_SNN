To make the codes in this folder work, you will need to create the directory 'files' inside the 'mvsec_dataset' directory, and then create the directories 'raw' and 'saved' inside it.

Please place all unzipped files from the MVSEC dataset on 'files/raw'.

Generated files will be stored in 'files/saved'.

In addition, the following files have been included:
- Two dataset class scripts have been included: _mvsec_dataset_indoor.py_ for training with indoor sequences, and _mvsec_dataset_outdoor_v2.py_ for outdoor sequences. In addition, the file _indexes.py_ contains the files belonging to each data split.
- The file _poolingNet_cat_1res_mvsec.py_ contains the network architecture, modified to account for the new tensor resolution.
- The file _train_3dNet_mvsec.py_ shows the training loop, and the file _test_mvsec.py_ computes the model's metrics.
