# Data Version Control
Trying out DVC with Git

## Objective

- Change the train.py file to work with PyTorch
- Repeat the whole [DVC tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial) again, with committing to GitHub.
  - Data MUST not be present on GitHub, but only available locally
  - While committing, mention the accuracy of each model (trained twice, each time 10 epochs (or more)).
  - Also publish metrics.csv file
- Go to [Actions](https://medium.com/swlh/automate-python-testing-with-github-actions-7926b5d8a865) Tab:
  - click Python application
  - add a test.py file that checks:
    - you have NOT uploaded data.zip
    - you have NOT uploaded model.h5 file
    - your accuracy of the model is more than 70% (read from metrics.csv)
    - the accuracy of your model for cat and dog is independently more than 70% (read from metrics.csv file)



## What is DVC?

Data Version Control, or DVC, is a data and ML experiment management tool  that takes advantage of the existing toolset (Git, CI/CD). DVC can be used as a Python library, simply install it with a package  manager like pip or conda. Depending on the type of remote storage you  plan to use, you might need to install optional dependencies: s3, azure, gdrive, gs, oss, ssh, etc.

DVC's features can be grouped into functional components:

- Data and model versioning
- Data and model access
- Data pipelines
- Metrics, parameters and plots
- Experiments

**Data Versioning**

![](https://github.com/gokul-pv/DataVersionControl/blob/master/Images/data_versioning.jpg)

DVC lets us capture the versions of our data and models in git commits,  while storing them on-premises or in cloud storage. It also provides a  mechanism to switch between these different components. The result is a  single history for data, code and ML models that we can traverse. 

![](https://github.com/gokul-pv/DataVersionControl/blob/master/Images/project_versioning.jpg)

## Preparation

Let's first download the code and set up a Git repository:

```bash
git clone https://github.com/iterative/example-versioning.git
cd example-versioning
```

The train.py file was then modified to work with PyTorch. Let's add some data and then train the first model. We can download the data using `dvc get`, which is similar to `wget` and can download any file or directory tracked in DVC repository.

**First model version**

```bash
dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/data.zip
unzip -q data.zip
rm -f data.zip
```

Now we have the data downloaded as zip file. Note that the data is downloaded from github here but it could any cloud storage. We than unzip the data and delete the zip file. We now have 1000 images for training and 800 images for validation. The data is in the structure:

```
data
├── train
│   ├── dogs
│   │   ├── dog.1.jpg
│   │   ├── ...
│   │   └── dog.500.jpg
│   └── cats
│       ├── cat.1.jpg
│       ├── ...
│       └── cat.500.jpg
└── validation
   ├── dogs
   │   ├── dog.1001.jpg
   │   ├── ...
   │   └── dog.1400.jpg
   └── cats
       ├── cat.1001.jpg
       ├── ...
       └── cat.1400.jpg
```



We can now capture the current state data using `dvc add`.

```bash
dvc add data
```

This command is used instead of `git add` on files or directories that are too large to be tracked with Git: usually input datasets, models, some intermediate results, etc. It tells Git to ignore the directory and puts it into the cache (while keeping a file link to it in the workspace, so we can continue working the same way as before). This is achieved by creating a tiny, human-readable .dvc file that serves as a pointer to the cache.

Now, after we train the model, we have set it up to generate the `model.h5` file which is basically the weights of our head model and also `metrics.csv` which is the metrics file storing the loss, accuracy, class-wise accuracy for each epoch during training/validation. The simplest way to capture the current version of the model is to use dvc `add again`.

```bash
python train.py
dvc add model.h5
```
> We manually added the model output here, which isn't ideal. The preferred way of capturing command outputs is with `dvc run`

Now let's commit the current state of the model and the data along with metrics.

```bash
git add data.dvc model.h5.dvc metrics.csv .gitignore
git commit -m "First model, trained with 1000 images"
git tag -a "v1.0" -m "model v1.0, 1000 images"
git status
```

DVC does not commit the data directory and model.h5 file with Git. Instead, DVC add stores them in the cache (usually in .dvc/cache) and adds them to .gitignore. In this case, we created data.dvc and model.h5.dvc, which contain file hashes that point to cached data. We then git commit these .dvc files.

**Second model version**

Let's imagine that our image dataset doubles in size. The next command extracts 500 new cat images and 500 new dog images into `data/train`:
```bash
dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/new-labels.zip
unzip -q new-labels.zip
rm -f new-labels.zip
```
Now our dataset has 2000 images for training and 800 images for validation. We will use these new labels and retrain the model:

```bash
dvc add data
python train.py
dvc add model.h5
```

```bash
git add data.dvc model.h5.dvc metrics.csv
git commit -m "Second model, trained with 2000 images"
git tag -a "v2.0" -m "model v2.0, 2000 images"
```

We've now tracked a second version of the dataset, model, and metrics in DVC and committed the [`.dvc`](https://dvc.org/doc/user-guide/project-structure/dvc-files) files that point to them with Git.

## Switching between workspace Versions

The DVC command that helps get a specific committed version of data is designed to be similar to git checkout. All we need to do in our case is to additionally run `dvc checkout` to get the right data into the workspace.There are two ways of doing this: a full workspace checkout or checkout of a specific data or model file. Let's consider the full checkout first.

```bash
git checkout v1.0
dvc checkout
```
These commands will restore the workspace to the first snapshot we made: i.e it checkouts the model, data, code and all of it from the v1. DVC optimizes this operation to avoid copying data or model files each time. So `dvc checkout` is quick even if you have large datasets, data files, or models.

On the other hand, if we want to keep the current code, but go back to the previous dataset version, we can target specific data, like this:

```bash
git checkout v1.0 data.dvc
dvc checkout data.dvc
```

Now if we run `git status` you'll see that data.dvc is modified and currently points to the v1.0 version of the dataset, while code and model files are from the v2.0 tag.

## Automating capturing

When you have a script that takes some data as an input and produces other  data outputs, a better way to capture them is to use dvc run. In our example, `train.py` the model file `model.h5`, and the metrics file `metrics.csv`.

```shell
dvc run -n train -d train.py -d data \
          -o model.h5 -M metrics.csv \
          python train.py
```

`dvc run` writes a pipeline stage named train (specified using the -n option) in dvc.yaml. It tracks all outputs (-o) the same way as dvc add does. Unlike dvc add, dvc run also tracks dependencies (-d) and the command (python train.py) that was run to produce the result.

## GitHub Actions: Automate Python Testing

Test_.py was added to check the below mentioned four teste cases and a CI task was created usig GitHub Actions to check the same on each push and pull request.

- you have NOT uploaded data.zip
- you have NOT uploaded model.h5 file
- your accuracy of the model is more than 70% (read from metrics.csv)
- the accuracy of your model for cat and dog is independently more than 70% (read from metrics.csv file)



## Reference 

1. [https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)
2. [https://github.com/iterative/example-versioning](https://github.com/iterative/example-versioning)
3. https://medium.com/swlh/automate-python-testing-with-github-actions-7926b5d8a865

