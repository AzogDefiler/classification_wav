# classification_wav
Классификатор: нейронная сеть - 4 conv1d layers / 1 dense layers + softmax,
такая топология была выбрана по причине того что для преобразования звуковой дорожки использовались Мел-кепстральные коэффициенты (MFCC) https://habr.com/post/140828/ , после чего получается фиксированный размер входных данных, заведомо не хотел юзать RNN.

-----------------------------------------------------------
## необходимые фреймворки и либы:
* keras
* numpy
* pandas
* matplotlib
* librosa - pip install librosa
* soundfile - pip install SoundFile
* sklearn
* wave
* glob
* struct
* jupyter notebook

-----------------------------------------------------------
1) закинуть в папку 'data_v_7_stc/test' валидационные данные
2) для теста обученной сетки запустить 'classification_testing.ipynb' в jupyter notebook

* 'classification_prepare_train.ipynb' - файл для обучения нейронки.
* 'feat.npy' - преобразованный датасет
* 'label.npy' - лейблы
* в папке 'weights' обученная модель с весами

-----------------------------------------------------------
точность на тестовых данных = 0.9885,
на валид-ых в файле 'result.txt'

-----------------------------------------------------------
на всякий случай прилагаются 2 файла 'classification_testing.py' и 'classification_prepare_train.py' если не установлен jupyter notebook
