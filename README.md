# knn-nbr-analysis

This repository hosts our code for the Recommender Systems university course, focusing on the analysis of ['Modeling Personalized Item Frequency Information for Next-basket Recommendation'](https://arxiv.org/pdf/2006.00556.pdf) paper.

## Requirements

The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt` or with the conda environment file `conda env create -f environment.yml`.

## Evaluation

The evaluation of the models is done using the `eval.py` script. The script can be run with the following command:

```bash
python src/eval.py <path_to_history_file> <path_to_future_purchases_file> <optional_arguments>
```

The script takes two required arguments, the first one is the path to the purchase history file, the second one is the path to the future purchases file. The script also takes many optional arguments, which can be seen by running `python src/eval.py --help`. For example, to run the evaluation on the TaFang dataset, run the following command:

```bash
python src/eval.py ./data/TaFang_history_NB.csv ./data/TaFang_future_NB.csv
```

## Dataset

Further datasets such as lastfm-1k or mmtd can be parsed by using `dataset.py` script. The following command can be used (dataset_name is either lastfm or mmtd):

```bash
python src/data/dataset.py <dataset_name> <path_to_dataset_file> <optional_arguments>
```

Optional arguements include months for baskets, which selects the month interval to create baskets, and listen threshold which removes songs that were listened less than n times in the dataset.
