-------------------------------------------- about dataset --------------------------------------------
Dataset can be downloaded from: https://drive.google.com/drive/folders/1o0JUSApr0vw-tiXm3JTIUkPb--umVgak?usp=sharing
Dataset consists of three types of data:
1. "photos" - photos of boards (.JPEG),
2. "photos_labeled" - photos of boards with labeled characteristic points (for the corner detection and board orientation model) (.JPEG),
3. "boards" - boards written in .xlsx files (for letter classification)
Dataset is divided into a training set "train_data" (99 photos) and a test set (25 photos).

-------------------------------------------- installation --------------------------------------------
# path set to scrabble_board_recognition folder
pip install -e .

-------------------------------------------- example of using the model --------------------------------------------
# letters are written to the console
python -m corner_training.torch_model test_ims --corners-checkpoint=corners_model.pt --letters-checkpoint=letters_model.pt --device=cpu "C:\Users\Monika\Desktop\PJATK\MGR\data\test_data\photos\IMG_8282.JPEG"

-------------------------------------------- visualization of augmented data --------------------------------------------
# visualization of augmented photos with 5-point labels and adding a grid to the cut-out letters
python -m corner_training.prepare_dataset vis_aug_corners --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data"

# visualization of cut-out, augmented letters (based on point labels)
python -m corner_training.prepare_dataset vis_aug_letters --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" 8330

-------------------------------------------- prepare dataset to training --------------------------------------------
python -m corner_training.prepare_dataset create_dataset_corners --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --out-file=corners_train_demo.h5 --num-epochs=1 --repeat-count=1
python -m corner_training.prepare_dataset create_dataset_letters --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --out-file=letters_train_demo.h5 --num-epochs=1 --min-repeat=1 --max-repeat=1 --repeat-scale=1
# visualization to debbug
python -m corner_training.torch_model show_dataset --dataset-path=corners_train_demo.h5 --device=cpu

-------------------------------------------- model for detecting corners and board orientation --------------------------------------------
# traning
python -m corner_training.torch_model train_corners --train-set-path=corners_train_demo.h5 --model-save-path=corners_model_demo.pt --stats-save-path=corner_model_losses_demo.npz --batch-size=4 --num-epochs=1 --testset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
# visualization stats
python -m corner_training.torch_model show_corners_stats --stats-path=corner_model_losses.npz
# test
python -m corner_training.torch_model score_checkpoint_corners --checkpoint-path=corners_model.pt --dataset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu

-------------------------------------------- model for letter classification --------------------------------------------
# traning
python -m corner_training.torch_model train_letters --train-set-path=letters_train_demo.h5 --model-save-path=letters_model_demo.pt --stats-save-path=letters_model_losses_demo.npz --batch-size=4 --num-epochs=4 --testset-dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
# visualization stats
python -m corner_training.torch_model show_letters_stats --stats-path=letters_model_losses.npz
# test of the model for classification as label verification (for alert_loss=0.7 I visualized where the model is uncertain, which helped correct labeling errors on the training tooth and the test, you can give boards and watch where it is uncertain and draw)
python -m corner_training.torch_model score_checkpoint_letters --letters-checkpoint=letters_model.pt --dataset_dir="C:\Users\Monika\Desktop\PJATK\MGR\data\train_data" --alert_loss=0.3 --device=cpu 4037 8335
# test
python -m corner_training.torch_model score_checkpoint_letters --letters-checkpoint=letters_model.pt --dataset_dir="C:\Users\Monika\Desktop\PJATK\MGR\data\test_data" --device=cpu
