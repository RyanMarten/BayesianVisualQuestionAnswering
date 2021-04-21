import torch
from gensim.models import KeyedVectors
import pyprob #https://github.com/probprog/pyprob #!pip install pyprob
from pyprob import Model 
from pyprob.distributions import Categorical, Uniform, Normal
from pyprob import InferenceEngine
import matplotlib.pyplot as plt

wv = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# define global question template variables
color_names = ["red", "green", "blue", "orange", "gray", "yellow"]
non_relational_qs = [
    "What shape is the {} object?",
    "Is the {} object on the left?",
    "Is the {} object on the top?"
]
relational_qs = [
    "What shape is the object closest to the {} object?",
    "What shape is the object furthest from the {} object?",
    "How many objects are the same shape as the {} object?"
]
all_qs = [non_relational_qs, relational_qs]

def text_to_vec(text):
    return torch.tensor([wv[w] for w in text.split() if w in wv]).mean(dim=0)

class QuestionsGen(Model):
    def __init__(self, name="QuestionsModel", opt=None):
        super().__init__(name=name)
        self.opt = opt
        self.colors = 6
        
    def compile_question(self, color, qtype, qsubtype, template):
        # compile a question latent
        color_vec = [0, 0, 0, 0, 0, 0]
        type_vec = [0, 0]
        subtype_vec = [0, 0, 0]
        color_vec[color] = 1
        type_vec[qtype] = 1
        subtype_vec[qsubtype] = 1
        question_vec = color_vec + type_vec + subtype_vec
        
        # get the text of the question
        question_text = all_qs[qtype][qsubtype].format(color_names[color])
        
        return question_text, question_vec
        
    def forward(self):
        c_i = pyprob.sample(Categorical(logits=[1 for _ in range(self.colors)]))
        t_i = pyprob.sample(Categorical(logits=(1,1)))
        st_i = pyprob.sample(Categorical(logits=(1,1,1)))
        tmp_i = pyprob.sample(Categorical(logits=(1,1)))
        cand_q_text, cand_q_latent = self.compile_question(c_i, t_i, st_i, tmp_i)
        pyprob.tag(cand_q_latent, name="question_latent")
        pyprob.tag(cand_q_text, name="question_text")
        cand_vec = text_to_vec(cand_q_text)
        
        pyprob.observe(Normal(cand_vec, 0.0001), name="observed_question")
        return cand_q_text, cand_q_latent

if __name__ == "__main__":
	model = QuestionsGen(Model)
	posterior_dist = model.posterior(
		num_traces=100,
		inference_engine = InferenceEngine.IMPORTANCE_SAMPLING,
		observe = {'observed_question': text_to_vec("What shape is the object closest to the green object?")},
		initial_trace=None
	)
	print(posterior_dist.sample()["question_text"])
	print(posterior_dist.sample()["question_latent"])