This code mainly includes three parts: training, evaluation score and test result output.

The path to some files needs to be configured in the dataload. where “all” is the sum of all validations.



train：

```shell
python3 excute train.py
```

validate:

```bash
python3 excute_validate.py
```

 test:

```shell
python3 excuete_test.py
```



score：

It is to facilitate the ensemble for offline scoring.



test_for_readme.py：
Used to calculate model size and inference time.

