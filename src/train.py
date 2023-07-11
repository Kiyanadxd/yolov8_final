from ultralytics import YOLO
from multiprocessing import freeze_support
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
           
if __name__ =="__main__":
    freeze_support()
# Load a model
    model_file = r'F:\Project_recog\falldetection-master\src\runs\detect\train\weights\best.pt'
    model_file_cfg = r'F:\Project_recog\falldetection-master\model_file\isFall.yaml'
    #data_path = r'C:\code\walking_distraction\persons'
    model = YOLO(model_file)  # load a pretrained model (recommended for training)


    # Train the model
      
    model.train(data=model_file_cfg,device='cpu', epochs=50)
    metrics = model.val()  # evaluate model performance on the validation set
