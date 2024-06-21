Aby wykorzystać model, wystarczy zainstalować moduły z folderu "scrabble_board_recognition"
oraz uruchomić odpowiednią funkcję podając w parametrze ścieżkę do zdjęcia z plnszą Scrabble.
Poniżej przedstawiono jak zainstalować paczkę oraz przykład użycia modelu.

Dodatkowo zamieszczono lokalizację pełnego zbioru danych wykorzystanego w treningu i testach
oraz poszczególne funkcje wykonane podczas realizacji projektu.

Aby wyjść z działającego programu należy wcisnąć klawisz ESC.
Aby przejść do kolejnego kroku należy wcisnąć dowolny inny klawisz np. spację.

-------------------------------------------- Instalacja modułów --------------------------------------------
# Należy ustawić ścieżkę na folder "scrabble_board_recognition"
pip install -e .

-------------------------------------------- Przykłady użycia modelu --------------------------------------------
# Litery rozpoznane na obrazach wypisują się w konsoli.
python -m corner_training.torch_model test_ims "IMG_3419.JPEG"
python -m corner_training.torch_model test_ims "IMG_3429.JPEG"
python -m corner_training.torch_model test_ims "IMG_8281.JPEG"
python -m corner_training.torch_model test_ims "IMG_8285.JPEG"

-------------------------------------------- Zbiór danych --------------------------------------------
Zbiór danych można pobrać z: https://drive.google.com/drive/folders/1o0JUSApr0vw-tiXm3JTIUkPb--umVgak?usp=sharing
Zbiór zawiera trzy rodzaje danych:
1. "photos" - zdjęcia plansz (.JPEG),
2. "photos_labeled" - zdjęcia plansz z oznakowanymi punktami charakterystycznymi(do modelu detekcji rogów i orientacji planszy) (.JPEG),
3. "boards" - plansza opisana w pliku .xlsx (do modelu klasyfikacji liter)
Zbiór danych jest podzielony na zbiór treningowy "train_data" (99 zdjęć) oraz na zbiór testowy (25 zdjęć).

-------------------------------------------- Wizualizacja augmentacji --------------------------------------------
# Wizualizacja zagmentowanych zdjęć z 5 punktami i nałożoną siatką do wycięcia liter
python -m corner_training.prepare_dataset vis_aug_corners --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data"

# Wizualizacja wyciętych, zaugmentowanych liter (na podstawie punktów oznaczonych na zdjęciach z "photos_labeled")
python -m corner_training.prepare_dataset vis_aug_letters --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" 8330

-------------------------------------------- Przygotowanie zbiorów danych do treningów --------------------------------------------
python -m corner_training.prepare_dataset create_dataset_corners --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --out-file=corners_train_demo.h5 --num-epochs=1 --repeat-count=1
python -m corner_training.prepare_dataset create_dataset_letters --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --out-file=letters_train_demo.h5 --num-epochs=1 --min-repeat=1 --max-repeat=1 --repeat-scale=1
# Wizualizacja do debbugowania
python -m corner_training.torch_model show_dataset --dataset-path=corners_train_demo.h5 --device=cpu

-------------------------------------------- Model do detekcji rogów i orientacji planszy --------------------------------------------
# Trening
python -m corner_training.torch_model train_corners --train-set-path=corners_train_demo.h5 --model-save-path=corners_model_demo.pt --stats-save-path=corner_model_losses_demo.npz --batch-size=4 --num-epochs=1 --testset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
# Statystyki
python -m corner_training.torch_model show_corners_stats --stats-path=corner_model_losses.npz
# Test
python -m corner_training.torch_model score_checkpoint_corners --checkpoint-path=corners_model.pt --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu

-------------------------------------------- Model do klasyfikacji liter --------------------------------------------
# Trening
python -m corner_training.torch_model train_letters --train-set-path=letters_train_demo.h5 --model-save-path=letters_model_demo.pt --stats-save-path=letters_model_losses_demo.npz --batch-size=4 --num-epochs=4 --testset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
# Statystyki
python -m corner_training.torch_model show_letters_stats --stats-path=letters_model_losses.npz
# Test modelu wykorzystany do weryfikcji oznakowań exceli (dla alert_loss=0.7 funkcja zwraca wizualizacje obrazu z naniesionymi literami z excela w miejsca na obrazie co do których model był niepewny lub zwrócił inną wartość, co pozwoliło półautometycznie weryfikować poprawność exceli)
python -m corner_training.torch_model score_checkpoint_letters --letters-checkpoint=letters_model.pt --dataset_dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --alert_loss=0.3 --device=cpu 4037 8335
# Test właściwy
python -m corner_training.torch_model score_checkpoint_letters --letters-checkpoint=letters_model.pt --dataset_dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
