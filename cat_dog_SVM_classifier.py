import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import time
import pickle

# Define file directories
file_dir = r"D:\progigy\data-full"
output_dir = "./output/SVM_trained.pth"
out_report_dir = './output/classification_report.txt'
TRAIN = "train"
TEST = "test"

def get_data(file_dir):
    """
    Load and transform the data using PyTorch's ImageFolder and DataLoader.
    """
    print("[INFO] Loading data...")
    data_transform = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(254),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    datasets_img = {}
    dataloaders = {}
    for file in [TRAIN, TEST]:
        dir_path = os.path.join(file_dir, file)
        print(f"[DEBUG] Checking directory: {dir_path}")
        if not os.path.isdir(dir_path):
            print(f"[ERROR] Directory does not exist: {dir_path}")
            continue

        datasets_img[file] = datasets.ImageFolder(dir_path, transform=data_transform[file])
        dataloaders[file] = torch.utils.data.DataLoader(datasets_img[file], batch_size=8, shuffle=True, num_workers=4)
    
    class_names = datasets_img[TRAIN].classes
    datasets_size = {file: len(datasets_img[file]) for file in [TRAIN, TEST]}
    for file in [TRAIN, TEST]:
        print(f"[INFO] Loaded {datasets_size[file]} images under {file}")
    print(f"Classes: {class_names}")

    return datasets_img, datasets_size, dataloaders, class_names

def get_vgg16_modified_model(weights=models.VGG16_BN_Weights.DEFAULT):
    """
    Retrieve the VGG-16 pre-trained model and remove the classifier layers.
    """
    print("[INFO] Getting VGG-16 pre-trained model...")
    vgg16 = models.vgg16_bn(weights)
    for param in vgg16.features.parameters():
        param.requires_grad = False
    features = list(vgg16.classifier.children())[:-7]
    vgg16.classifier = nn.Sequential(*features)
    return vgg16

def get_classification_report(truth_values, pred_values):
    """
    Generate a classification report and confusion matrix based on ground truth and predicted labels.
    """
    report = classification_report(truth_values, pred_values, target_names=class_names, digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all')
    print('[Evaluation Model] Showing detailed report\n')
    print(report)
    print('[Evaluation Model] Showing confusion matrix')
    print(f'                       Predicted Label              ')
    print(f'                         0            1         ')
    print(f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}')
    print(f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}')
    
def save_classification_report(truth_values, pred_values, out_report_dir):
    """
    Save the classification report and confusion matrix to a text file.
    """
    print('[INFO] Saving report...')
    c_report = classification_report(truth_values, pred_values, target_names=class_names, digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all')
    matrix_report = ['                       Predicted Label              ',
                     f'                         0            1         ',
                     f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}',
                     f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}']
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_report_dir), exist_ok=True)

    with open(out_report_dir, 'w') as f:
        f.write(c_report)
        f.write('\n')
        for line in matrix_report:
            f.write(line)
            f.write('\n')
            
def get_features(vgg, file=TRAIN):
    """
    Extract features and labels from the VGG-16 model for a given dataset.
    """
    print(f"[INFO] Getting '{file}' features...")
    svm_features = []
    svm_labels = []
    data_batches_len = len(dataloaders[file])
    for i, data_batch in enumerate(dataloaders[file]):
        print(f"\r[FEATURE] Loading batch {i + 1}/{data_batches_len} ({len(data_batch[1])*(i+1)} images)", end='', flush=True)
        inputs, labels = data_batch
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            features = vgg(inputs)
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            features = vgg(inputs)
            features = features.detach().numpy()
            labels = labels.detach().numpy()
        
        for index in range(len(labels)):
            feature = features[index]
            label = labels[index]
            svm_features.append(feature)
            svm_labels.append(label)
            
    print("\n[FEATURE] Features loaded")
    return svm_features, svm_labels

def svm_classifier(train_data, test_data):
    """
    Train an SVM classifier on the extracted features and evaluate its performance.
    """
    since = time.time()
    FEATURE_INDEX = 0
    LABEL_INDEX = 1
    print('[INFO] Getting model...')
    train_features = np.array(train_data[FEATURE_INDEX])
    train_labels = np.array(train_data[LABEL_INDEX])
    test_features = np.array(test_data[FEATURE_INDEX])
    test_labels = np.array(test_data[LABEL_INDEX])
    
    svm_model = SVC(gamma="auto")
    print('[INFO] Fitting...')
    svm_model.fit(train_features, train_labels)
    print('[INFO] Model completed')
    print('[INFO] Testing...')
    pred_labels = svm_model.predict(test_features)
    print('[INFO] Printing classification report')
    get_classification_report(test_labels, pred_labels)
    elapsed_time = time.time() - since
    print(f"[INFO] Model produced in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    save_classification_report(test_labels, pred_labels, out_report_dir)
    return svm_model

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    vgg16 = get_vgg16_modified_model()
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    svm_train_features, svm_train_labels = get_features(vgg16, TRAIN)
    svm_test_features, svm_test_labels = get_features(vgg16, TEST)
    svm_model = svm_classifier(
        [svm_train_features, svm_train_labels],
        [svm_test_features, svm_test_labels],
    )
    print('[INFO] Saving model...')
    pickle.dump(svm_model, open(output_dir, 'wb'))
    print('[INFO] Done')
