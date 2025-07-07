import os
import json
import torch
import argparse
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from plot_metrics import plot_confusion_matrix, plot_roc
from torch.cuda.amp import autocast
from utils import read_test_data, MyDataSet
from tqdm import tqdm
import sys
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from main_model import CAME as create_model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    print(args)

    dataset_path = args.test_data_path
    # class_name
    for _, dirs, _ in os.walk(dataset_path):
        class_name = dirs
        break
    class_name.sort()
    print(class_name)

    num_classes = len(class_name)
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(256)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=num_classes).to(device)

    # load model weights
    model_weight_path = args.model_path # "model_weight/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)) 
    model.eval()

    # load test dataset
    test_images_path, test_images_label = read_test_data(dataset_path)
    test_dataset = MyDataSet(images_path=test_images_path,
                        images_class=test_images_label,
                        transform=data_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=test_dataset.collate_fn)

    # y_true; y_pred; y_scores
    y_true = []
    y_pred = []
    y_scores = [[] for i in range(num_classes)]
    test_loader = tqdm(test_loader, file=sys.stdout)
    with torch.no_grad():
        for test_data in test_loader:
            test_loader.desc = "[testing...]"
            images, labels = test_data
            labels = labels.numpy().tolist()
            # y_true
            y_true += labels

            # predict
            with torch.cuda.amp.autocast():
                output = torch.squeeze(model(images.to(device))).cpu()
            output = output.to(torch.float32)

            if output.dim() == 1: # if only 1 image in last batch
                output = output.unsqueeze(0) 
            predict = torch.softmax(output, dim=1)  # score
            
            # y_scores
            for index, row in enumerate(predict.T):
                y_scores[index] += [float(i) for i in row.tolist()]

            predict_cla = torch.argmax(predict, dim=1) # class

            # y_pred
            y_pred += predict_cla.numpy().tolist()

    # report
    y_true = [class_indict[str(i)] for i in y_true]
    y_pred = [class_indict[str(i)] for i in y_pred]
    print(classification_report(y_true, y_pred, target_names=class_name, digits=4))
    plot_confusion_matrix(  cm = confusion_matrix(y_true, y_pred), 
                            normalize    = True,
                            target_names = class_name,
                            title        = "Confusion Matrix")
    # roc
    plot_roc(y_true, y_scores, class_name)

    # 25.7.7 add MCC and kappa
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc)

    kappa = cohen_kappa_score(y_true, y_pred)
    print("Cohen's kappa:", kappa)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_data_path', type=str, default="../test_set")
    parser.add_argument('--model_path', type=str, default="model_weight/best_model.pth")

    opt = parser.parse_args()
    main(opt)

