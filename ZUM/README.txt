---------------- do pobrania: Restaurant_Reviews.tsv ----------------
Plik Restaurant_Reviews.tsv to dane źródłowe do naszych modeli ML.
Pochodzą z portalu Kaggle.com: https://www.kaggle.com/datasets/ehabashraf/restaurant-reviewstsv
Tabela zawiera dwie kolumny: Review (text) oraz Liked (0/1).
Są to oceny klientów w języku angielskim różnych restauracji oraz flaga czy opinia była pozytywna.
Dane składają się z 977 unikalnych opiniii ważą 25kB.

---------------- do pobrania: glove.6B.50d.txt ----------------
Plik glove.6B.50d.txt to pre-trenowne wektory slow o wymiarze 50, uzyskane za pomoca algorytmu GloVe.
Pochodza ze strony: https://nlp.stanford.edu/projects/glove/

---------------- _NLP_praca_domowa.ipynb ----------------
W pliku _NLP_praca_domowa.ipynb znajduje się notebook z treścią i rozwiązaniem pracy domowej z części NLP.
Znajdziemy tutaj następujące części:
-> pobranie bibliotek i modułów
-> pobranie danych
-> czyszczenie i przygotowanie danych
-> wizualizacja chmur słów (negative, positive) oraz rozkład klas
-> podział danych
-> przygotowanie 4 modeli (LSTM, CNN, Pre-Trained Embedding z glove50d, DistilBERT), w tym:
    -> tokenizacja, 
    -> padding, 
    -> kompilacja, 
    -> trenowanie
    -> ewaluacja
    -> podsumowanie (macierz omyłek i raport klasyfikacji)
    -> przykłowe uzycie modelu 
-> funkcja predict_sentiment(), ktora dostaje tekst i jeden z 4 modeli do wyboru i zwraca informacje czy text nacechowany jest negatywnie czy pozytywnie
-> uzycie predict_sentiment() na przykladowych zdaniach

---------------- Jak korzystac z plikow? ----------------
1. Pobrac pliki.
2. Uruchomić _NLP_praca_domowa.ipynb z dodaniem dwóch pozostałych plików do środowiska do tego folderu w którym znajduje się notebook.


---------------- Opis wybranego modelu: DistilBERT ----------------
-- teoria --
Ogólnie model DistilBERT to zoptymalizowana wersja oryginalnego modelu BERT (Bidirectional Encoder Representations from Transformers), 
który został skrócony, utrzymując jednocześnie wysoką zdolność do rozumienia kontekstu w tekście. 
Jest oparta na architekturze Transformer i oferuje znaczną redukcję wymagań zasobów obliczeniowych, 
co pozwala na bardziej efektywne szkolenie i korzystanie z modelu do zadań przetwarzania języka naturalnego. 

-- implementacja --
W pierwszej kolejności tekst jest tokenizowany za pomocą tokenizatora DistilBertTokenizer.
Nasępnie mamy przygotowanie danych jako tensorflow.data.Dataset do trenowania.
Kolejno załadowana jest gotowa architektura modelu DistilBERT (można poczytać np. tutaj https://huggingface.co/distilbert-base-uncased)
I na koniec kompilacja, trenowanie i ewalucja.

-- wyniki --
Dla mojego zadania klasyfikacji model DistilBERT osiągnął 0.8 Accuracy.
Jest to najwyższy wynik w porównaniu do pozostałych trzech modeli (0.75 w LSTM, 0.77 w CNN, 0.54 w pretreined embedding).
Dlatego też wybrałam ten model do opisania i rekomendowałabym użycie jego produkcyjnie, 
choć uważam, że można byłoby jeszcze polepszyć wynik dokładając więcej danych do treningu i tuningując parametry.
