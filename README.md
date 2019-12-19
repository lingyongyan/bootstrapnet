<!--
 * @Author: your name
 * @Date: 2019-12-19 02:08:48
 * @LastEditTime: 2019-12-19 02:14:18
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /bootstrapnet/README.md
 -->
## End-to-End Bootstrapping Neural Network for Entity Set Expansion
Source code for [AAAI2020](https://aaai.org/Conferences/AAAI-20/) paper:[End-to-End Bootstrapping Neural Network for Entity Set Expansion]()

### Requirements
- pytorch >= 1.2.0
- torchtext >= 0.3.1

### Dataset
- We use the dataset provided by [Zupon et al.(2019)](https://github.com/clulab/releases/tree/master/naacl-spnlp2019-emboot/data), and you can also download them as well as all preprocessed data from [google driver](https://drive.google.com/file/d/1Ow6Rf_LIilKvm0dVuJSF5dGigTMosOQq/view?usp=sharing)
- We use the [Glove](https://nlp.stanford.edu/projects/glove/) for entity and pattern embedding initialization ( You can use the pre-trained [Bert](https://github.com/google-research/bert) for embedding initialization)

### Preprocess:
- To accelarate the model training and testing process, we use following preprocess steps:
```shell
cd preprocess
python generate_graph.py # generate graph data in numerical representation for quick reading
python generate_dev.py # generate development data
python generate_initialization.py # generate entity and pattern initialized embeddings.
```
- All prepreocessed files are included in [google driver](https://drive.google.com/file/d/1Ow6Rf_LIilKvm0dVuJSF5dGigTMosOQq/view?usp=sharing), just unzip it and put them in "data" directory.

### Execution
- To execute our code, please run:
```shell
python run_conoll.py # for conoll dataset
# python run_onto.py # for onto dataset
```
### Citation
Please cite the following paper if you find our code is helpful.
```bibtex
@inproceedings{yan_end_2020,
  author = "Lingyong, Yan and 
    Xianpei, Han and
    Ben, He and
    Le, Sun",
  title = "End-to-End Bootstrapping Neural Network for Entity Set Expansion",
  booktitle = "Thirty-Fourth AAAI Conference on Artificial Intelligence",
  year = "2020"
}
```