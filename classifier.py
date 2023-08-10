import torch.nn as nn
import torch

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      #self.fc1 = nn.Linear(153600,768)
      self.fc1 = nn.Linear(768,512)

      self.fc2 = nn.Linear(512,206)
      # dense layer 2 (Output layer)
      self.fc3 = nn.Linear(206,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):

      #pass the inputs to the model  
      _, cls_hs = self.bert(input_ids, attention_mask=attention_mask,return_dict=False)
      x = self.fc1(cls_hs)
      # #NEW PART
      # cls_hs = self.bert(input_ids, attention_mask=attention_mask,return_dict=False)
      # hidden_states = cls_hs[2]
      # # concatenate the tensors for all layers
      # # use "stack" to create new dimension in tensor
      # token_embeddings = torch.stack(hidden_states, dim=0)
      # # remove dimension 1, the "batches"
      # token_embeddings = torch.squeeze(token_embeddings, dim=1)
      # # swap dimensions 0 and 1 so we can loop over tokens
      # token_embeddings = token_embeddings.permute(1,0,2)
      # # intialized list to store embeddings
      # token_vecs_sum = []
      # # "token_embeddings" is a [Y x 12 x 768] tensor
      # # where Y is the number of tokens in the sentence
      # # loop over tokens in sentence
      # token_embeddings = torch.sum(token_embeddings, dim=1)
      # # for token in token_embeddings:
      # # # "token" is a [12 x 768] tensor
      # # # sum the vectors from the last four layers
      # #     sum_vec = torch.sum(token[-4:], dim=0)
      # #     token_vecs_sum.append(sum_vec)
      # token_embeddings = token_embeddings.flatten()
      # x = self.fc1(token_embeddings)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc3(x)
      
      # apply softmax activation
      #x = self.softmax(x)

      # return x.unsqueeze(dim=0)
      return x