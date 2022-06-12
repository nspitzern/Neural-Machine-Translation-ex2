Before runnign this part please verify that you have Seaborn module installed.

To run part 3 use the following command:

python dump_part3.py -p [PATH_TO_DATA_FILES]

The -p flag indicates the path to the data files (the folder should contain train.src, train.trg, dev.src, dev.trg,
test.src and test.trg).

To tweek any other hyper-parameter use the config_part3.py module.

* Notice that the training script saves the mapping of words to indices and indices to words for the training and dev sets,
and also the models of the encoder and decoder. Therefore, running the evaluation should be in the same directory as the those files.