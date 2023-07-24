# Whisper Benchmarking
This tool helps to benckmark Whisper. The confihguration is setup with Hydra, and all settings can ve found in the conf.yaml file. 
For all different benchmarks a benchmark config file is given. This tells the dataset name and language. 
The benchmarking can both benchmark the Open Whisper implementation of whisper and the Hugging Face implementation. 

## Datasets 
For all the datasets a PyTorch dataloader has been created. To use the benchmarking new paths to the locaation of datasets have to be given. It is easily edited in the files in /src/dataset_loaders

## Running benchmarks 
To run the shortform benchmarks it can be done as the following
```
python main.py model=Large benchmark=ftspeech batch_size=128
```
Batch size can be set for running the short form. For longform batchsize is only supported for huggingface models. 
Longform can look like the following: 
```
python main.py model=Large model=$model benchmark=rev16 batch_size=16 whisper_version=huggingface
```
Here the whisper version is given by an argument. 

It is also possible to benchmark how long it takes to run inference. The code will first do a single warmup inference over the file, and then do the average of 10 runs and provide the result. 

```
python3 main.py model=large benchmark=longform_time batch_size=128 whisper_version=huggingface
```
