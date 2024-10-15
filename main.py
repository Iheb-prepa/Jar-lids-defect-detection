import argparse
from anomalib.models import Patchcore, Padim, EfficientAd, ReverseDistillation
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib import TaskType
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train(folder_datamodule, model):
    engine.fit(datamodule=folder_datamodule, model=model)


def test(model, folder_datamodule, ckpt_path):
    # Computes AUROC and F1-Score measures
    engine.test(model=model, datamodule=folder_datamodule, ckpt_path=ckpt_path)


def predict(model, folder_datamodule, ckpt_path):
    # Performs prediction on the test set
    prediction_results = engine.predict(model=model,
                                    dataloaders=folder_datamodule.test_dataloader(), return_predictions=True, ckpt_path=ckpt_path)
    return prediction_results


def plot_confusion_matrix(prediction_results):
    # Compute and plot confusion matrix
    true_labels, predicted_labels = [], []
    for current_batch in prediction_results:
        for i in range(len(current_batch['label'])):
            image_true_label = current_batch['label'][i]
            image_pred_label = current_batch['pred_labels'][i]
            true_labels.append(image_true_label) 
            predicted_labels.append(image_pred_label)

    # Construct confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    plt.savefig("confusion_matrix")
    print("Generated the confusion matrix at the current directory.")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or Test anomaly detection model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], 
                        help="Choose between 'train' or 'test' mode.")
    args = parser.parse_args()

    # Initialize the datamodule, model and engine
    model = Padim()
    engine = Engine()


    folder_datamodule = Folder(
        name="cans_defect_detection",
        root=str(Path.cwd() / "cans_defect_detection_dataset"),
        normal_dir="intact",
        abnormal_dir="damaged",
        task=TaskType.CLASSIFICATION,
        image_size=(256, 256),
        num_workers=6,                     
    )
    folder_datamodule.setup()

    # Train or test based on the command-line argument
    if args.mode == 'train':
        print("Training the model...")
        train(folder_datamodule, model) # The training generates a checkpoint file that is useful for the next two steps.
    elif args.mode == 'test': # computes evaluation metrics on the test set, also generates the predictions in the results folder.
        print("Testing the model...")
        ckpt_path = '' # Put with your checkpoint path, for example: ckpt_path= 'results/Padim/cans_defect_detection/v1/weights/lightning/model.ckpt'
        test(model, folder_datamodule, ckpt_path)
    elif args.mode == 'predict': # Predicts on the test set and computes confusion matrix
        ckpt_path = '' # change with your checkpoint path, for example: ckpt_path= 'results/Padim/cans_defect_detection/v1/weights/lightning/model.ckpt'
        prediction_results = predict(model, folder_datamodule, ckpt_path)
        plot_confusion_matrix(prediction_results)




    
    

    