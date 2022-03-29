# Final Project for the Course Implementing ANNs with Tensorflow
## from Jannis Lippold, Nikolas Wintering and Steffen Meinert

### Repository structure
- ./train_models.ipynb:         model training loop, executable in google colab
- ./calculate_prob_a_b.ipynb:   calculates the weights vector for prob_loss, saved in weights_tensor.npy
- ./calculate_metrics.ipynb:    calculates evaluation metrics using checkpoints of models after training __n__ epochs
- ./plot_metrics.ipynb:         creates plots from the metrics
- ./create_ds.ipynb:            creates the used dataset from a directory of images
- ./src/data_util/:             data pipeline to map images to (input,target) pairs in grayscale/CIELAB color space
- ./src/iizuka/:                implementation of Iizuka et. al's model
- ./src/zhang/:                 implementation of zhangs simpel model, not using probability distribution
- ./src/zhang_prob/:            implementation of zhangs model, using the probability distribution

### Usage of train_models.ipynb
One can execute this file in google colab. It will automatically clone this repository onto the machine and start to train the model chosen in the **Parameter** section. There, one has to update the paths to the used dataset, the location of checkpoint- and logfiles. The dataset should consist of (input,target) pairs where the input is a grayscale image with values normalized to [-1;1] and the target is a CIELAB encoded image of the same height and width. A data pipeline to convert ordinary image datasets to this structure can be found in 'src/data_util/data_pipeline_util.py'.
