import pyprob #https://github.com/probprog/pyprob #!pip install pyprob
from pyprob import Model 
from pyprob.distributions import Categorical, Uniform
import numpy as np
import torch
import cv2
from pyprob import PriorInflation, InferenceEngine, InferenceNetwork
import matplotlib.pyplot as plt


class ImagesGen(Model):
  def __init__(self, name='ImagesModel', opt = None):
    super().__init__(name=name)
    self.opt = opt
    
    self.height, self.width = 75, 75
    self.colors = [
        (0,0,255),##r
        (0,255,0),##g
        (255,0,0),##b
        (0,156,255),##o
        (128,128,128),##k
        (0,255,255)##y
    ]
    self.size = 5

  def forward(self):
    # Initialize images with 255 (white background)
    img = np.ones((self.height,self.width, 3), dtype=np.uint8)*255
    objects = []
    for color_id,color in enumerate(self.colors):  
        center = self.center_generate(objects)
        shape = pyprob.sample(Categorical(probs=[0.5,0.5]), name=f"{color_id}_shape").item()
        if shape:
            start = (center[0]-self.size, center[1]-self.size)
            end = (center[0]+self.size, center[1]+self.size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, self.size, color, -1)
            objects.append((color_id,center,'c'))
    
    #Use kernel from proposal network
    
    


    rendered_img = torch.tensor(img) # I had to add this for some reason --> check if it is right
    pyprob.tag(rendered_img, name="semantic_image")
    #one_hot_rendered_img = self.from_grey_to_one_hot(rendered_img)
    #rendered_img = self.smooth_semantic(one_hot_rendered_img, alpha = 0.1)
    #reshaped_rendered_img = rendered_img.view(12,-1).permute(1,0)
    #pyprob.observe(Categorical(probs=reshaped_rendered_img),name='observed_semantic_image')
  
  def center_generate(self, objects):
    while True:
        pas = True
        center_x = pyprob.sample(Categorical(probs=[0] * self.size + [1/(self.height-2*self.size)]*(self.height-2*self.size)), name=f"{len(objects)}_center_x").item()
        center_y = pyprob.sample(Categorical(probs=[0] * self.size + [1/(self.height-2*self.size)]*(self.height-2*self.size)), name=f"{len(objects)}_center_y").item()
       #self.np.random.randint(0+size, img_size - size, 2)        
        center = torch.tensor([center_x, center_y])
        if len(objects) > 0:
            for name,c,shape in objects:
                if torch.sum(((center - c) ** 2)) < ((self.size * 2) ** 2):
                    pas = False
        if pas:
            return center

def plot_trace(trace_image, title):
    img = np.array(trace_image)
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(img)
    plt.show()
  
if __name__ == "__main__":
    model = ImagesGen()
    
    for i in range(10):
        gt_trace = next(model._trace_generator(inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS))
        gt_sem_image = gt_trace["semantic_image"].view(75,75, 3)
        plot_trace(gt_sem_image, "Ground Truth Semantic Image")