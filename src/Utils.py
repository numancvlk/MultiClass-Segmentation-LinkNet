#LIBRARIES
import torch 
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

#SCRIPTS
from Dataset import MyDataset
from Model import DEVICE

def saveCheckpoint(model,optimizer, fileName = "myCheckpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint,fileName)
    print("MODEL KAYEDİLDİ")

def loadCheckpoint(checkpointFile, model, optimizer):
    checkpoint = torch.load(checkpointFile, map_location="cuda" if torch.cuda.is_available() else "cpu")
    print("MODEL YÜKLENİYOR")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def printTrainTime(startTimer, endTimer, device):
    totalTime = endTimer - startTimer
    print(f"Total training time is {totalTime} on the {device}")

def getLoaders(trainImages,
               trainMasks,
               testImages,
               testMasks,
               validationImages,
               validationMasks,
               classDictPath,
               batchSize,
               numWorkers,
               pinMemory,
               trainTransform = None,
               testTransform = None,
               validationTransform = None):
    
    trainDatas = MyDataset(imagesDir=trainImages,
                           masksDir= trainMasks,
                           classDictPath=classDictPath,
                           transform=trainTransform)
    
    testDatas = MyDataset(imagesDir=testImages,
                          masksDir=testMasks,
                          classDictPath=classDictPath,
                          transform=testTransform)
    
    validationDatas = MyDataset(imagesDir=validationImages,
                                masksDir=validationMasks,
                                classDictPath=classDictPath,
                                transform=validationTransform)
    
    trainDataLoader = DataLoader(dataset=trainDatas,
                                 batch_size=batchSize,
                                 shuffle=True,
                                 num_workers=numWorkers,
                                 pin_memory=pinMemory)
    
    testDataLoader = DataLoader(dataset=testDatas,
                                batch_size=batchSize,
                                shuffle=False,
                                num_workers=numWorkers,
                                pin_memory=pinMemory)
    
    validationDataLoader = DataLoader(dataset=validationDatas,
                                      batch_size=batchSize,
                                      shuffle=False,
                                      num_workers=numWorkers,
                                      pin_memory=pinMemory)
    
    return trainDataLoader, testDataLoader, validationDataLoader

def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn,
              scaler,
              device: torch.device = DEVICE):
    
    model.train()
    loop = tqdm(dataLoader)

    for batch, (xTrain,yTrain) in enumerate(loop):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device).long()

        with torch.autocast(device_type=DEVICE):
            trainPred = model(xTrain)
            loss = lossFn(trainPred, yTrain)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss = loss.item())


def multiClassAccuracy(model: torch.nn.Module,
                       dataLoader: DataLoader,
                       device: torch.device = "cuda",
                       numClasses: int = 32,
                       returnDiceScore: bool = False):
    
    model.eval()
    
    totalCorrect = 0
    totalPixels = 0
    totalDice = 0.0
    totalIoU = 0.0

    with torch.no_grad():
        for xBatch, yBatch in dataLoader:
            xBatch, yBatch = xBatch.to(device), yBatch.to(device)
            
            logits = model(xBatch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            totalCorrect += (preds == yBatch).sum().item()
            totalPixels += preds.numel()

            batchDice = 0.0
            batchIoU = 0.0
            for cls in range(numClasses):
                predCls = (preds == cls).float()
                trueCls = (yBatch == cls).float()
                intersection = (predCls * trueCls).sum()
                union = predCls.sum() + trueCls.sum()

                # Dice
                if union.item() == 0:
                    dice = 1.0  
                else:
                    dice = (2.0 * intersection) / (union + 1e-6)
                batchDice += dice

                # IoU
                unionIoU = union - intersection
                if unionIoU.item() == 0:
                    iou = 1.0 
                else:
                    iou = intersection / (unionIoU + 1e-6)
                batchIoU += iou

            batchDice /= numClasses
            batchIoU /= numClasses

            totalDice += batchDice.item() * xBatch.size(0)
            totalIoU += batchIoU.item() * xBatch.size(0)

    diceScore = totalDice / len(dataLoader.dataset)
    iouScore = totalIoU / len(dataLoader.dataset)
    accuracy = totalCorrect / totalPixels * 100

    print(f"ACCURACY = {accuracy:.5f}% | DICE SCORE = {diceScore:.5f} | IoU SCORE = {iouScore:.5f}")

    if returnDiceScore:
        return diceScore
        
def multiclassCrossEntropyDiceLoss(pred, target, numClasses=None, smooth=1e-5):
    if numClasses is None:
        numClasses = pred.shape[1]

    ce = F.cross_entropy(pred, target.long())

    predSoft = F.softmax(pred, dim=1)
    diceLoss = 0.0
    valid_classes = 0

    for cls in range(numClasses):
        predCls = predSoft[:, cls, :, :]
        targetCls = (target == cls).float()

        if targetCls.sum() == 0:
            continue

        intersection = (predCls * targetCls).sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (predCls.sum(dim=(1,2)) + targetCls.sum(dim=(1,2)) + smooth)
        diceLoss += (1 - dice).mean()
        valid_classes += 1

    diceLoss /= max(valid_classes, 1)
    loss = 0.7 * ce + 0.3 * diceLoss
    return loss

def savePredictionMultiClass(model: torch.nn.Module,
                             dataLoader: torch.utils.data.DataLoader,
                             classColorMap:dict,
                             device: torch.device = DEVICE,
                             folder = "saved_images/"):
    
    model.eval()
    os.makedirs(folder,exist_ok=True)

    for idx, (xBatch,yBatch) in enumerate(dataLoader):
        xBatch, yBatch = xBatch.to(device), yBatch.to(device)

        with torch.no_grad():
            preds = model(xBatch)
            preds = torch.argmax(torch.softmax(preds, dim=1),dim=1)

        preds = preds.cpu().numpy()
        yBatch = yBatch.cpu().numpy()

        for i in range(preds.shape[0]):
            predMask = np.zeros((preds.shape[1], preds.shape[2], 3), dtype=np.uint8)
            trueMask = np.zeros((preds.shape[1], preds.shape[2], 3), dtype=np.uint8)

            for cls, color in classColorMap.items():
                predMask[preds[i] == cls] = color
                trueMask[yBatch[i] == cls] = color

            # Kaydet
            predImg = Image.fromarray(predMask)
            trueImg = Image.fromarray(trueMask)

            predImg.save(os.path.join(folder, f"pred_{idx}_{i}.png"))
            trueImg.save(os.path.join(folder, f"true_{idx}_{i}.png"))