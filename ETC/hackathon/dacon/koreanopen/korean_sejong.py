import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '../_data/dacon/korean/'
train = pd.read_csv(os.path.join(path, 'train.csv'), encoding='utf-8')
test = pd.read_csv(os.path.join(path, 'test.csv'), encoding='utf-8')

train.head(5)

print(train.info(), end='\n\n')
print(test.info())

feature = train['label']

plt.figure(figsize=(10,7.5))
plt.title('Label Count', fontsize=20)

temp = feature.value_counts()
plt.bar(temp.keys(), temp.values, width=0.5, color='b', alpha=0.5)
plt.text(-0.05, temp.values[0]+20, s=temp.values[0])
plt.text(0.95, temp.values[1]+20, s=temp.values[1])
plt.text(1.95, temp.values[2]+20, s=temp.values[2])

plt.xticks(temp.keys(), fontsize=12) # x축 값, 폰트 크기 설정
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 레이아웃 설정
# plt.show() # 그래프 나타내기

max_len = np.max(train['premise'].str.len())
min_len = np.min(train['premise'].str.len())
mean_len = np.mean(train['premise'].str.len())

print('Max Premise Length: ', max_len)
print('Min Premise Length: ', min_len)
print('Mean Premise Lenght: ', mean_len, '\n')

max_len = np.max(train['hypothesis'].str.len())
min_len = np.min(train['hypothesis'].str.len())
mean_len = np.mean(train['hypothesis'].str.len())

print('Max Hypothesis Length: ', max_len)
print('Min Hypothesis Length: ', min_len)
print('Mean Hypothesis Lenght: ', mean_len)

from collections import Counter

plt.figure(figsize=(10,7.5))
plt.title('Premise Length', fontsize=20)

plt.hist(train['premise'].str.len(), alpha=0.5, color='orange')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 레이아웃 설정

# plt.show()

train['premise'] = train['premise'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
test['premise'] = test['premise'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]", "")
train.head(5)

train['hypothesis'] = train['hypothesis'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
test['hypothesis'] = test['hypothesis'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]", "")
train.head(5)

import os
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

def seed_everything(seed:int = 66):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

MODEL_NAME = 'klue/roberta-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 3

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

print(model)
print(config)


train_dataset, eval_dataset = train_test_split(train, test_size=0.2, shuffle=True, stratify=train['label'])

tokenized_train = tokenizer(
    list(train_dataset['premise']),
    list(train_dataset['hypothesis']),
    return_tensors="pt",
    max_length=256, # Max_Length = 190
    padding=True,
    truncation=True,
    add_special_tokens=True
)

tokenized_eval = tokenizer(
    list(eval_dataset['premise']),
    list(eval_dataset['hypothesis']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

print(tokenized_train['input_ids'][0])
print(tokenizer.decode(tokenized_train['input_ids'][0]))

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        
        return item

    def __len__(self):
        return len(self.label)

def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []

    for v in label:
        num_label.append(label_dict[v])
    
    return num_label


train_label = label_to_num(train_dataset['label'].values)
eval_label = label_to_num(eval_dataset['label'].values)

train_dataset = BERTDataset(tokenized_train, train_label)
eval_dataset = BERTDataset(tokenized_eval, eval_label)

print(train_dataset.__len__())
print(train_dataset.__getitem__(19997))
print(tokenizer.decode(train_dataset.__getitem__(19997)['input_ids']))


def compute_metrics(pred): #validation을 위한 metrics function
        
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions

  # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

        return {
      'accuracy': acc,
  }
  
  
training_ars = TrainingArguments(
    output_dir='./result',
    num_train_epochs=7,
    per_device_train_batch_size=32,
    save_total_limit=5,
    save_steps=500,
    evaluation_strategy='steps',
    eval_steps = 500,
    load_best_model_at_end = True,
)

trainer = Trainer(
    model=model,
    args=training_ars,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained('./result/best_model')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Tokenizer_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

MODEL_NAME = './result/checkpoint-4000'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(tokenizer.vocab_size)
model.to(device)

print(tokenizer)

test_label = label_to_num(test['label'].values)

tokenized_test = tokenizer(
    list(test['premise']),
    list(test['hypothesis']),
    return_tensors="pt",
    max_length=128,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

test_dataset = BERTDataset(tokenized_test, test_label)

print(test_dataset.__len__())
print(test_dataset.__getitem__(1665))
print(tokenizer.decode(test_dataset.__getitem__(6)['input_ids']))

dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
output_pred = []
output_prob = []

for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device)
        )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
print(pred_answer)

def num_to_label(label):
    label_dict = {0: "entailment", 1: "contradiction", 2: "neutral"}
    str_label = []

    for i, v in enumerate(label):
        str_label.append([i,label_dict[v]])
    
    return str_label

answer = num_to_label(pred_answer)
print(answer)



df = pd.DataFrame(answer, columns=['index', 'label'])

df.to_csv(path + '0214_1.csv', index=False)

print(df)





'''
tokenized_train = tokenizer(  # 토큰화
    list(train_dataset['premise']), # 토큰화할 문장
    list(train_dataset['hypothesis']), # 토큰화할 문장
    return_tensors="pt", # 토큰화한 결과를 파이토치의 tensor로 반환
    max_length=256, # Max_Length = 190 # 토큰화할 문장의 최대 길이
    padding=True, # 토큰화할 문장의 길이가 최대 길이보다 작을 경우, 패딩을 적용하여 최대 길이로 맞춤
    truncation=True, # 토큰화할 문장의 길이가 최대 길이보다 클 경우, 잘라내어 최대 길이로 맞춤
    add_special_tokens=True # 토큰화할 문장에 특수 토큰을 추가하는 경우, 특수 토큰을 추가함


class BERTDataset(torch.utils.data.Dataset): # torch.utils.data.Dataset을 상속받음
        def __init__(self, pair_dataset, label): #  self는 클래스 자신을 의미,pair_dataset은 파일명, label은 라벨
        self.pair_dataset = pair_dataset  # 파일명을 저장
        self.label = label # 라벨을 저장하는 변수

    def __getitem__(self, idx):  # 파일명을 반환하는 함수
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()} 
        # clone()는 객체를 복사하는 함수, detach()는 객체를 분리하는 함수
        item['label'] = torch.tensor(self.label[idx]) # tensor로 변환
        
        return item # return_tensors = "pt"를 설정하면 파이토치의 tensor로 반환하게 된다.

    def __len__(self):   # __len__()는 객체의 길이를 반환하는 함수
        return len(self.label) # 길이를 반환
 
       
training_ars = TrainingArguments(  # 옵션을 설정하는 함수
    output_dir='./result', # 결과 파일을 저장할 경로
    num_train_epochs=7, # 학습 횟수
    per_device_train_batch_size=6, # 학습 데이터의 배치 크기
    save_total_limit=5, # 학습 결과를 저장할 파일의 최대 개수
    save_steps=500, # 학습 결과를 저장할 때 간격
    evaluation_strategy='steps', # 평가 방법
    eval_steps = 500, # 평가 반복 횟수
    load_best_model_at_end = True, # 학습 결과를 저장할 때 가장 좋은 결과를 저장할지 여부)
    
    
trainer = Trainer( # 학습을 위한 함수
    model=model, #
    args=training_ars, # args는 옵션을 설정하는 함수
    train_dataset=train_dataset, # 학습 데이터
    eval_dataset=eval_dataset, # 평가 데이터
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics, # 평가 방법)

'''