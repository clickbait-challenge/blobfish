# blobfish

The Blobfish Clickbait Detector by [Davide Fonzi](https://www.linkedin.com/in/davide-fonzi/), [Nicolò Pratelli](https://www.linkedin.com/in/pratelli-1991/) e [Lavinia Salicchi](https://www.researchgate.net/profile/Lavinia_Salicchi)

![python](https://img.shields.io/badge/python%20tested-3.6.7-blue.svg)

#### Download and usage

1. Download [training data](https://www.clickbait-challenge.org/#data) and extract in the datasets  /datasets/ subfolders:
	* /1/: [clickbait16-train-170331.zip](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170331.zip)
	* /2/: [clickbait17-train-170630.zip](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170630.zip)
	* /3/: [clickbait17-unlabeled-170429.zip](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-unlabeled-170429.zip)
2. git clone https://github.com/tira-io/blobfish

3. cd blobfish

4. install all dependencies for python 3.x
   ```python
   pip3 install -r requirements.txt
   ```


5. for training

   ```python
   python3 main.py
   ```

6. for test
   ```python
   python3 test.py -i test_data -o output -m model
   ```
   **test_data**: directory where is located test dataset

   **output**: path of the output text

   **model**: name of the choosen model (*WordEmbNet*, *LingNet*, *FullNetPost*, *FullNet*)

   if you want to test your trained model you must change path of the model into test.py
    ```python
    dir = "longTraining/models/"
    if type_model == "FullNet":
	 	model_path = dir + "path/to/model/model_name.hdf5"
    if type_model == "FullNetPost":
    	model_path = dir + "path/to/model/model_name.hdf5"
    if type_model == "LingNet":
    	model_path = dir + "path/to/model/model_name.hdf5"
    if type_model == "WordEmbNet":
    	model_path = dir + "path/to/model/model_name.hdf5"
    ```


7. for evaluating
   ```python
   python3 eval.py test_data/truth.jsonl ./output/results.jsonl output.prototext
   ```


