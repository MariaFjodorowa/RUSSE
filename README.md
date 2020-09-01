Вариант решения https://github.com/nlpub/russe-wsi-kit .

Как запускать (протестировано под Linux):

- распаковать этот архив в корень репозитория https://github.com/nlpub/russe-wsi-kit
1. для запуска ноутбука ELMo.ipynb:

- создать виртуальную среду (https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)
- установить tensorflow==1.15.3 и tensorflow-gpu==1.15.3, скачав их с https://pypi.org/project/tensorflow/1.15.3/#files и https://pypi.org/project/tensorflow-gpu/1.15.3/#files (просто через pip install теперь можно установить только версию 2.0, но она не подойдет)
- ```pip install -r elmo-requirements.txt```

1. для запуска ноутбука sentence-transformers.ipynb:

- создать виртуальную среду (https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)
- ```pip install -r sent-tr-requirements.txt```

Примечание. Для sentence-transformers все ставится под Python 3.6, для ELMo можно и под Python 3.7.
