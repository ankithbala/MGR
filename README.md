

Usage
-----



```shell
mkdir data
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar zxvf genres.tar.gz
cd ..
pip install -r requirements.txt
python3 create_data_pickle.ipynb
python3 train_model.ipynb #old code trained for 100 epochs
python3 newTrain.ipynb # My code of cnn plus LSTM

```




