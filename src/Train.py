#LIBRARIES
import torch
import albumentations as A
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations import ToTensorV2
from timeit import default_timer
import numpy as np
import random
#SCRIPTS
from Model import linkNetModel, DEVICE
from Utils import saveCheckpoint, loadCheckpoint, getLoaders, trainStep, printTrainTime, multiClassAccuracy, multiclassCrossEntropyDiceLoss, savePredictionMultiClass

#HYPERPARAMETERS
EPOCHS = 400
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 480
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

#PATHS
TRAIN_IMAGES = "dataset\\train"
TRAIN_MASKS = "dataset\\train_labels"
TEST_IMAGES ="dataset\\test"
TEST_MASKS = "dataset\\test_labels"
VALIDATION_IMAGES = "dataset\\val"
VALIDATION_MASKS = "dataset\\val_labels"
CLASS_DICT_PATH = "dataset\\class_dict.csv"

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    patience = 25
    patienceCounter = 0
    bestDice = 0

    try:
        class_dict = pd.read_csv(CLASS_DICT_PATH)
        classColorMap = {
            idx: (row["r"], row["g"], row["b"]) for idx, row in class_dict.iterrows()
        }
        print(f"'{CLASS_DICT_PATH}' dosyasından {len(classColorMap)} adet sınıf/renk eşleşmesi yüklendi.")
    except FileNotFoundError:
        print(f"HATA: '{CLASS_DICT_PATH}' dosyası bulunamadı. Lütfen 32 sınıflı CamVid için renk/ID eşleşme dosyasını kontrol edin.")
        exit()

    trainTransform = A.Compose([
        A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05,scale_limit=0.05, rotate_limit=10, p=0.5),
        A.GaussianBlur(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0),
        ToTensorV2()
    ])

    testTransform = A.Compose([
        A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0),
        ToTensorV2()
    ])

    validationTransform = A.Compose([
        A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0),
        ToTensorV2()
    ])

    trainDataLoader, testDataLoader, validationDataLoader = getLoaders(trainImages= TRAIN_IMAGES,
                                                                       trainMasks= TRAIN_MASKS,
                                                                       testImages= TEST_IMAGES,
                                                                       testMasks=TEST_MASKS,
                                                                       validationImages=VALIDATION_IMAGES,
                                                                       validationMasks=VALIDATION_MASKS,
                                                                       classDictPath=CLASS_DICT_PATH,
                                                                       trainTransform=trainTransform,
                                                                       testTransform=testTransform,
                                                                       validationTransform=validationTransform,
                                                                       batchSize=BATCH_SIZE,
                                                                       numWorkers=NUM_WORKERS,
                                                                       pinMemory=PIN_MEMORY)
    
    #MODEL / LOSS FUNC / OPTIMIZER / SCALER
    model = linkNetModel.to(DEVICE)
    lossFn = multiclassCrossEntropyDiceLoss
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()
    scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    factor=0.5,
    patience=10,        # kaç epoch gelişme olmazsa LR düşsün
    threshold=1e-3,     # küçük değişiklikleri yok say
    threshold_mode='rel',# en iyi diceScore’a göre göreceli değişim
    )

    if LOAD_MODEL == True:
        print("CHECKPOINT BULUNDU MODEL YÜKLENİYOR")
        loadCheckpoint(checkpointFile="myCheckpoint.pth",
                       model=model,
                       optimizer=optimizer)

    startTrainTimer = default_timer()

    for epoch in range(EPOCHS):
        print(f"-----EPOCH = {epoch}-----")
        trainStep(model=model,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  scaler=scaler,
                  device=DEVICE)
        
        diceScore = multiClassAccuracy(model=model,
                                       dataLoader=testDataLoader,
                                       device=DEVICE,
                                       numClasses=32,
                                       returnDiceScore=True)
        
        scheduler.step(diceScore)
            
        for paramsGroup in optimizer.param_groups:
            currentLr = paramsGroup["lr"]
            print(f"CURRENT LR = {currentLr}")

        if diceScore > bestDice:
            bestDice = diceScore
            print("EN İYİ MODEL KAYDEDİLİYOR")
            patienceCounter = 0
            saveCheckpoint(model=model,
                           optimizer=optimizer)
            savePredictionMultiClass(model=model,
                                 dataLoader=validationDataLoader,
                                 classColorMap=classColorMap,
                                 device=DEVICE)
        else:
            patienceCounter += 1
            print(f"{patienceCounter}'tur gelişme görülmedi.")

            if patienceCounter == patience:
                print("EARLY STOPPING TRIGGERED")
                break
    
    endTrainTimer = default_timer()

    printTrainTime(startTimer=startTrainTimer,
                   endTimer=endTrainTimer,
                   device=DEVICE)