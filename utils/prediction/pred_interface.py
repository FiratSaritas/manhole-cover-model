import torch
import torch.nn.functional as F
import pandas as pd

class PredictionInterface():
    """
    This class implements the Prediction interface for the model
    """
    
    def __init__(self, model):
        """
        Initializes the Prediction Interface

        Params:
        -------------------
        model: (torch.model)     Neural Network class Pytorch     

        """
        self.model = model

    def predict_one(self,imag:torch.Tensor,label_dict: dict, top_n: int = 1):
        """
        Returns true and predicted labels for prediction for one image
        Params:
        ---------
        model:           Pytorch Neuronal Net
        imag:            Image
        label_dict:      Dictionary of labels
        top_n:           Number of top predictions to return
        returns:
        ----------
        (y_true, y_pred, y_images, y_prob):
            y_true       True labels
            y_pred:      Predicted Labels
            y_prob:      Predicted Probability (empty if return_prob = False)
            y_images:    Images (empty if return_images = False)
        """
        self.model.eval()
        image = imag
        y_prob = []
        with torch.no_grad():
            y_probs = F.softmax(self.model(image), dim = -1) 
            y_prob.append(y_probs.cpu()) 
        y_prob = torch.cat(y_prob, dim = 0)
        y_prob = y_prob[0]
        n_classes, n_probs = [], []
        example_prob, example_label = torch.sort(y_prob, dim=0)
        for i in range(top_n):
            label = example_label[-i-1]
            n_classes.append(list(label_dict.keys())[list(label_dict.values()).index(label)])
            n_probs.append(float(example_prob[-i-1]))
        zipped = list(zip(n_classes, n_probs))
        df = pd.DataFrame(zipped, columns=['Prediction', 'Probability'])
        df.index += 1 
        result = df.to_dict('index')
        return result

    