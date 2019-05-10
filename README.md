# Musci Genre Recognition 

Music genre recognition is an interesting area of research, dealing with
recognizing and labeling the genre of the music. Research done in this area
has been few since the algorithm to recognize music and input data are
complex. An extraordinary range of information is hidden inside the music
waveforms ranging from perceptual to auditory-which inevitably makes
large-scale applications challenging.

Our project focuses on classifying any given music into its respective
genre such as pop, rock, disco etc. 

Dataset Used: GTZAN ( http://opihi.cs.uvic.ca/sound/genres.tar.gz )

Usage
-----



```shell
mkdir data
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar zxvf genres.tar.gz
cd ..
pip install -r requirements.txt #install all these packages
python3 create_data_pickle.ipynb
python3 train_model.ipynb #old code trained for 100 epochs
python3 newTrain.ipynb # My code of cnn plus LSTM

```




