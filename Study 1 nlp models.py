"""
pip install protobuf==3.20.0
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==3.3.1

export CUDA_VISIBLE_DEVICES=0
python train.py --data dummydata.csv --extra_features
python train.py --data data_all_45_850_577_rea_order.csv --pretrain_epochs 1 --finetuning_epochs 1 --seed 420 --backbone deberta

"""
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import pandas as pd
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForMaskedLM
# from transformers import DistilBertModel

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForPreTraining, AdamW
import transformers

from torch.utils.data import Dataset, TensorDataset
import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np
import random
import sys
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr import visualization as viz
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import sklearn.metrics as metrics
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

import os

CUDA = (torch.cuda.device_count() > 0)





class MyDataset(torch.utils.data.Dataset):
  def __init__(self, input_ids, attn_mask, labels=None,
        reason_feats=None, order_feats=None):
    self.input_ids = input_ids
    self.attn_mask = attn_mask
    if labels is not None:
        self.labels = labels
    if reason_feats is not None:
        self.reason_feats = reason_feats
    if order_feats is not None:
        self.order_feats = order_feats

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, index):
    item = {
        'input_ids': self.input_ids[index],
        'attention_mask': self.attn_mask[index]
    }   
    if hasattr(self, 'labels'):
        item['labels'] = self.labels[index]
    if hasattr(self, 'reason_feats'):
        item['reason_feats'] = int(self.reason_feats[index])
    if hasattr(self, 'order_feats'):
        item['order_feats'] = int(self.order_feats[index])

    return item


def build_dataset(texts, labels=None, reason_feats=None, order_feats=None):
    global MODEL_STR
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_STR, do_lower_case=True   
    )
    tokenized_data = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        max_length=150,
        return_attention_mask=True)

    dataset = MyDataset(
        tokenized_data['input_ids'], 
        tokenized_data['attention_mask'],
        labels,
        reason_feats=reason_feats,
        order_feats=order_feats)

    return dataset, tokenizer


def compute_metrics(pred):
    """p: EvalPrediction obj"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def finetune_model(model, train_dataset, tokenizer, working_dir,
        mlm=False, epochs=3, eval_dataset=None):
    if mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm_probability=0.15)
    else:
        data_collator = None

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        output_dir=working_dir)

    model = model.train()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator)

    train_result = trainer.train()

    # works but removing in favor of classification_eval() below for AUC support
    if eval_dataset is not None:
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    return train_result, model





def finetune_model_manual(model, train_dataset, tokenizer, working_dir,
        mlm=False, epochs=3, eval_dataset=None, attr=False, prefix=''):
    if mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm_probability=0.15)
    else:
        data_collator = None

    dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16, collate_fn=data_collator)
    model = model.cuda()
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-6)#, eps=1e-8)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)  
    accs = []
    for epoch in tqdm(range(epochs), total=epochs):
        avg_loss = 0.0
        for step, batch in enumerate(dataloader):
            if CUDA: 
                batch = {k: torch.stack([t.cuda() for t in v], dim=-1).cuda() for k, v in batch.items()}
            model.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # if isinstance(model, transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForSequenceClassification):
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.detach().cpu().item()
        if epoch + 5 >= epochs:
            model.eval()    
            acc = eval_standard(model, eval_dataset)["acc"]
            print(f'loss: {avg_loss/len(dataloader)}  acc: {acc}')
            accs.append(acc)
            model.train()

    if attr:
        model.eval()
        eval_results, vizs, t2a = eval_attribute(
            model, eval_dataset, tokenizer)

        print("WRITING VIZ to viz.html")
        with open(f'{prefix}-viz.html', 'w') as f:
            f.write('\n'.join(vizs))

        attr_rows = []
        for k in t2a:
            attr_rows.append({
                'token': k,
                '1': np.mean(t2a[k]),
                'N': len(t2a[k])
            })
        print('WRITING IMPORTANCE TO importance.tsv')
        df_attr = pd.DataFrame(attr_rows)
        df_attr = df_attr.sort_values(by=['1', 'N'], ascending=False)
        df_attr.to_csv(f'{prefix}-importance.tsv', sep='\t')
    with open(f'{prefix}-acc.tsv', 'a') as f:
        f.write(f"{accs[-1]}\t{eval_results['acc']}\n")

    return accs


def inference_manual(model, test_dataset):
    """Use a dataloader to run inference."""
    model.eval()
    out = []
    for step, batch in enumerate(dataloader):
        if CUDA: 
            batch = (x.cuda() for x in batch)
        input_ids, labels = batch
        with torch.no_grad():
            loss, logits = model(input_ids, labels=labels)            
        preds = scipy.special.softmax(logits.cpu().numpy(), axis=1)
        out += preds.tolist()
    return out






def binary_classification_report(y_true, y_prob):
    y_prob = np.array(y_prob)
    y_hat = (y_prob > 0.5).astype('int')
    return {
        # 'confusion-matrix': metrics.confusion_matrix(y_true, y_hat).astype(np.float32),
        # 'precision': metrics.precision_score(y_true, y_hat),
        # 'recall': metrics.recall_score(y_true, y_hat),
        # 'f1': metrics.f1_score(y_true, y_hat),
        'acc': metrics.accuracy_score(y_true, y_hat),
        # 'auc': metrics.roc_auc_score(y_true, y_prob),
        # 'loss': metrics.log_loss(y_true, y_prob)
    }





def transfer_parameters(from_model, to_model):
    to_dict = to_model.state_dict()
    from_dict = {k: v for k, v in from_model.state_dict().items() if k in to_dict}
    to_dict.update(from_dict)
    to_model.load_state_dict(to_dict)

    return to_model





class Attributor:
    def __init__(self, model, target_class, tokenizer):
        """ TODO generalize to multiclass """
        self.model = model
        self.target_class = target_class
        self.tokenizer = tokenizer
        
        self.fwd_fn = self.build_forward_fn(target_class)
        global MODEL_STR
        if MODEL_STR == 'microsoft/deberta-v3-base':
            self.lig = LayerIntegratedGradients(self.fwd_fn, self.model.deberta.embeddings)
        elif MODEL_STR == 'roberta-base':
            self.lig = LayerIntegratedGradients(self.fwd_fn, self.model.roberta.embeddings)
        else:
            raise Exception("unknown model str: " + MODEL_STR)

    def attribute(self, input_ids):

        ref_ids = [[x if x in [101, 102] else 0 for x in input_ids[0]]]

        attribution, delta = self.lig.attribute(
                inputs=torch.tensor(input_ids).cuda() if CUDA else torch.tensor(input_ids),
                baselines=torch.tensor(ref_ids).cuda() if CUDA else torch.tensor(ref_ids),
                n_steps=25,
                internal_batch_size=5,
                return_convergence_delta=True)

        attribution_sum = self.summarize(attribution)        

        return attribution_sum, delta

    def attr_and_visualize(self, input_ids, label):
        attr_sum, delta = self.attribute(input_ids)
        y_prob = self.fwd_fn(input_ids)
        pred_class = 1 if y_prob.data[0] > 0.5 else 0

        if CUDA:
            input_ids = input_ids.cpu().numpy()[0]
            label = label.cpu().item()
            attr_sum = attr_sum.cpu().numpy()
            y_prob = y_prob.cpu().item()
        else:
            input_ids = input_ids.numpy()[0]
            label = label.item()
            attr_sum = attr_sum.numpy()
            y_prob = y_prob.item()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        record = viz.VisualizationDataRecord(
            attr_sum, 
            y_prob,
            pred_class,
            label,
            self.target_class,
            attr_sum.sum(),
            tokens,
            delta)

        tok2attr = defaultdict(list)
        for tok, attr in zip(tokens, attr_sum):
            tok2attr[tok].append(attr)

        html = viz.visualize_text([record])

        return html.data, tok2attr, attr_sum, y_prob, pred_class


    def build_forward_fn(self, label_dim):

        def custom_forward(inputs):
            preds = self.model(inputs)[0]
            return torch.softmax(preds, dim=1)[:, label_dim]

        return custom_forward

    def summarize(self, attributions):
        """ sum across each embedding dim and normalize """
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

def eval_standard(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1, # has to be batch size 1 
        sampler=SequentialSampler(dataset),
        # group by datatype
        collate_fn=lambda data: {k: [x[k] for x in data] for k in data[0].keys()})
    
    model.eval()

    if CUDA:
        model = model.cuda()

    y_probs = []
    y_hats = []
    labels = []

    for batch in tqdm(dataloader):

        labels += batch['labels']

        if CUDA:
            batch = {k: torch.tensor(v).cuda() for k, v in batch.items()}
        else:
            batch = {k: torch.tensor(v) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            # if isinstance(model, transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForSequenceClassification):
            loss = outputs.loss
            preds= outputs.logits
            # else:
            #     loss, preds = outputs

        preds = torch.softmax(preds, dim=1)[:, 1].cpu().numpy().tolist()
        y_probs += preds
        # y_hat += [1] if preds[0] > 0.5 else [0]

    report = binary_classification_report(labels, y_probs)
    return report
    

def eval_attribute(model, dataset, tokenizer):
    dataloader = DataLoader(dataset, batch_size=1, # has to be batch size 1 
        sampler=SequentialSampler(dataset),
        # group by datatype
        collate_fn=lambda data: {k: [x[k] for x in data] for k in data[0].keys()})

    model.eval()

    if CUDA:
        model = model.cuda()

    attributor = Attributor(model, target_class=1, tokenizer=tokenizer)

    y_probs = []
    y_hats = []
    labels = []
    vizs = []
    tok2attr = None

    for batch in tqdm(dataloader):

        labels += batch['labels']

        if CUDA:
            batch = {k: torch.tensor(v).cuda() for k, v in batch.items()}
        else:
            batch = {k: torch.tensor(v) for k, v in batch.items()}

        viz, t2a, attrs, y_prob, y_hat = attributor.attr_and_visualize(
            batch['input_ids'], batch['labels'])

        if tok2attr is None:
            tok2attr = t2a
        else:
            for k, v in t2a.items():
                tok2attr[k] += v

        y_probs += [y_prob]
        y_hats += [y_hat]
        vizs.append(viz)

    report = binary_classification_report(labels, y_probs)
    return report, vizs, tok2attr

class DistilBertExtraFeats(nn.Module):
    def __init__(self, str, num_labs):
        super().__init__()
        # TODO hacky
        global N_REASON, N_ORDER

        # super().__init__(config)
        self.num_labels = num_labs #config.num_labels

        self.distilbert = AutoModel.from_pretrained(
            str,
            num_labels=num_labs,
            output_attentions=False,
            output_hidden_states=False
        )
        self.config = self.distilbert.config
        # print(self.config)
        if hasattr(self.config, 'dim'):
            hidden = self.config.dim
        else:
            hidden = self.config.hidden_size

        self.pre_classifier = nn.Linear(hidden + N_REASON + N_ORDER, hidden)

        self.classifier = nn.Linear(hidden, num_labs)
        self.dropout = nn.Dropout(0.2)

        # Initialize weights and apply final processing
        # self.post_init()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids=None,
        reason_feats=None,
        order_feats=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        # TODO hacky
        global N_REASON, N_ORDER

        # TODO HERE!!!
        classifier_input = torch.cat((
            pooled_output, 
            F.one_hot(reason_feats, num_classes=N_REASON),
            F.one_hot(order_feats, num_classes=N_ORDER)), dim=1)

        pooled_output = self.pre_classifier(classifier_input)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )



































from random import shuffle

def noise_seq(seq, drop_prob=0.25, shuf_dist=3, drop_set=None, keep_bigrams=False):
    def perm(i):
        return i[0] + (shuf_dist + 1) * np.random.random()
    
    if drop_set == None:
        dropped_seq = [x for x in seq if np.random.random() > drop_prob]
    else:
        dropped_seq = [x for x in seq if not (x in drop_set and np.random.random() < drop_prob)]

    if keep_bigrams:
        i = 0
        original = ' '.join(seq)
        tmp = []
        while i < len(dropped_seq)-1:
            if ' '.join(dropped_seq[i : i+2]) in original:
                tmp.append(dropped_seq[i : i+2])
                i += 2
            else:
                tmp.append([dropped_seq[i]])
                i += 1

        dropped_seq = tmp

    # global shuffle
    if shuf_dist == -1:
        shuffle(dropped_seq)
    # local shuffle
    elif shuf_dist > 0:
        dropped_seq = [x for _, x in sorted(enumerate(dropped_seq), key=perm)]
    # shuf_dist of 0 = no shuffle

    if keep_bigrams:
        dropped_seq = [z for y in dropped_seq for z in y]
    
    return dropped_seq


def augment_data(df, n=3):
    for index, row in df.iterrows():

        s = str(row['allwords'])
        for j in range(n):
            tmp = ' '.join(noise_seq(s.split()))
            df = df.append(pd.DataFrame({
              'uniqueid': f'{index}-{j}', 
              'allwords': tmp,
              '45words': ' '.join(tmp.split()[:45]), 
              'classifier': int(row['classifier']), 
              'escalation': int(row['escalation']), 
              'reason_feature': int(row['reason_feature']), 
              'order_feature': int(row['order_feature']),
              'index': [len(df) + 1]
            }))
    df = df.sample(frac=1)
    return df










def split_df(df, 
        selection_colname, 
        label_colname, 
        label_balance='natural', 
        test_proportion=0.25, 
        pretrain_all=False,
        fold='1/1'):

    df_finetune = df.loc[df[selection_colname] == 1]

    if label_balance == '50:50':
        esc = df_finetune.loc[df_finetune[label_colname] == 1]
        none = df_finetune.loc[df_finetune[label_colname] == 0].sample(len(esc))
        df_finetune = pd.concat([esc, none])
        # df_finetune = df_finetune.sample(frac=1)

    elif label_balance == 'natural':
        pass
        # df_finetune = df_finetune.sample(frac=1)

    if fold == '1/1':
        idx = int(len(df_finetune) * test_proportion)
        df_test = df_finetune[:idx]
        df_train = df_finetune[idx:]
    else: 
        num, denom = [int(x) for x in fold.split('/')]
        block_size = int(len(df_finetune) * 1.0 / denom)

        # for kfold xval
        if num == 1 and denom == 1:
            pass
        elif (num + 1) == denom: # last block
            df_test = df_finetune[-block_size:]
            df_train = df_finetune[: -block_size]
        else:
            df_test = df_finetune[num * block_size: num*block_size + block_size]
            df_train = pd.concat(
                (df_finetune[: num*block_size],
                df_finetune[num*block_size + block_size : ]),
                axis=0)

    if pretrain_all:
        df_pretrain = df
    else:
        df_pretrain = df[~df.isin(df_test)].dropna(how='all')
    df_pretrain = df_pretrain.sample(frac=1)
    return df_train, df_test, df_pretrain


# TODO sooo hacky...
N_REASON = 0
N_ORDER = 0
MODEL_STR = "distilbert-base-uncased"

def run_expt(args):
    global N_REASON, N_ORDER, MODEL_STR


    if args.seed == -1:
        seed = int(random.random() * 1000)
    else:
        seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 

    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)

    df = pd.read_csv(args.data, delimiter=',')

    black_df = df.loc[df['classifier'] == 1]
    black_0 = black_df.loc[black_df['escalation'] == 0]
    # black_0 = black_0.sample(frac=1) #shuffle
    black_1 = black_df.loc[black_df['escalation'] == 1]

    white_df = df.loc[df['classifier'] == 1]
    white_0 = white_df.loc[white_df['escalation'] == 0]
    white_1 = white_df.loc[white_df['escalation'] == 1]

    test_size = int(len(black_1) * 0.1)

    test_0 = random.sample(list(black_0.index), test_size)
    test_1 = random.sample(list(black_1.index), test_size)

    # hack for kfold
    i = 0
    # while i < len(black_1.index):
    #     print('STARTING FOLD ', i)
    #     test_0 = list(black_0.index)[i: i + test_size]
    #     test_1 = list(black_1.index)[i: i + test_size]
    #     i += test_size
    #     test_idxs = test_0 + test_1

    train_1 = list(set(list(black_1.index) + list(white_1.index)) - set(test_1))
    train_0 =  list(set(list(black_0.index) + list(white_1.index)) - set(test_0))

    if args.balance: 
        train_0 = random.sample(train_0, len(train_1))
    train_idxs = train_1 + train_0

    df_train = df.iloc[train_idxs]
    df_test = df.iloc[test_idxs]

    if args.augmentation > 0:
        df_train = augment_data(df_train, n=args.augmentation)

    N_REASON = int(np.max(df.reason_feature)) + 1
    N_ORDER = int(np.max(df.order_feature)) + 1

    if args.backbone in {'bert', 'roberta', 'deberta'}:
        if args.backbone == 'bert':
            MODEL_STR = 'bert-base-uncased'
        elif args.backbone == 'roberta':
            MODEL_STR = 'roberta-base'
        elif args.backbone == 'deberta':
            MODEL_STR = 'microsoft/deberta-v3-base'

        cls_model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_STR,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False)

        train_dataset, _ = build_dataset(
            texts=df_train[args.finetune_colname].tolist(),
            labels=df_train[args.label_colname].tolist(),
            reason_feats=df_train['reason_feature'].tolist() if args.extra_features else None,
            order_feats=df_train['order_feature'].tolist() if args.extra_features else None,
        )

        test_dataset, tokenizer = build_dataset(
            texts=df_test['45words'].tolist(),
            labels=df_test[args.label_colname].tolist(),
            reason_feats=df_train['reason_feature'].tolist() if args.extra_features else None,
            order_feats=df_train['order_feature'].tolist() if args.extra_features else None,
        )

        accs = finetune_model_manual(
            model=cls_model, 
            train_dataset=train_dataset, 
            eval_dataset=test_dataset,
            tokenizer=tokenizer, 
            working_dir=args.working_dir,
            mlm=False,
            epochs=args.finetuning_epochs,
            attr=args.attr_inf,
            prefix=str(i))
        


    else:
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk import word_tokenize
        
        corpus = list(df_train[args.finetune_colname])

        cv = CountVectorizer(
            input=corpus,
            ngram_range=(1,4),
            lowercase=True,
            tokenizer=lambda x: word_tokenize(x),
            min_df=2,
            binary=True
        )
        cv.fit(corpus)
        X_tr = cv.transform(corpus)
        Y_tr = df_train['escalation']
        
        X_te = cv.transform(list(df_test['45words']))
        Y_te = df_test['escalation']

        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        if args.backbone == 'logreg':
            model = LogisticRegression(penalty='l2')
        elif args.backbone == 'nb':
            model = MultinomialNB()
        elif args.backbone == 'forest':
            model = RandomForestClassifier()
        elif args.backbone == 'svm':
            model = SVC()
    
        model.fit(X_tr, Y_tr)
        preds = list(model.predict(X_te))
        acc = accuracy_score(Y_te, preds)

        return acc
    quit()
    ############################################################
    # LM pretraining
    if args.pretrain_epochs > 0:
        print('## LM PRETRAINING')
        pretrain_texts = df_pretrain[args.pretrain_colname].tolist()
        dataset, tokenizer = build_dataset(pretrain_texts)
        mlm_model = AutoModelForPreTraining.from_pretrained(MODEL_STR)
        result, mlm_model = finetune_model_manual(
            model=mlm_model, 
            train_dataset=dataset, 
            tokenizer=tokenizer, 
            working_dir=args.working_dir,
            mlm=True,
            epochs=args.pretrain_epochs)
    else:
        print('## SKIPPING PRETRAINING')


    ############################################################
    # Classification
    if args.extra_features:
        cls_model = DistilBertExtraFeats(
            MODEL_STR,
            num_labs=2)
    else:
        cls_model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_STR,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False)

    if args.pretrain_epochs > 0:
        cls_model = transfer_parameters(mlm_model, cls_model)

    train_dataset, _ = build_dataset(
        texts=df_train[args.finetune_colname].tolist(),
        labels=df_train[args.label_colname].tolist(),
        reason_feats=df_train['reason_feature'].tolist() if args.extra_features else None,
        order_feats=df_train['order_feature'].tolist() if args.extra_features else None,
    )

    test_dataset, tokenizer = build_dataset(
        texts=df_test[args.finetune_colname].tolist(),
        labels=df_test[args.label_colname].tolist(),
        reason_feats=df_train['reason_feature'].tolist() if args.extra_features else None,
        order_feats=df_train['order_feature'].tolist() if args.extra_features else None,
    )

    cls_model = finetune_model_manual(
        model=cls_model, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset,
        tokenizer=tokenizer, 
        working_dir=args.working_dir,
        mlm=False,
        epochs=args.finetuning_epochs)

    if args.save_model:
        torch.save(cls_model.state_dict(), args.working_dir + '/model.ckpt')

    ############################################################
    # Eval and dump
    print(eval_standard(cls_model, test_dataset)); quit()
    if args.extra_features:
        eval_results = eval_standard(cls_model, test_dataset)
    else:
        eval_results, vizs, t2a = eval_attribute(
            cls_model, test_dataset, tokenizer)

        importance_out = os.path.join(args.working_dir, 'out_importance.tsv')
        viz_out = os.path.join(args.working_dir, 'out_viz.html')

        with open(viz_out, 'w') as f:
            f.write('\n'.join(vizs))

        attr_rows = []
        for k in t2a:
            attr_rows.append({
                'token': k,
                '1': np.mean(t2a[k]),
                'N': len(t2a[k])
            })

        df_attr = pd.DataFrame(attr_rows)
        df_attr = df_attr.sort_values(by=['1', 'N'], ascending=False)
        df_attr.to_csv(importance_out, sep='\t')

    s = ''

    metrics_out = os.path.join(args.working_dir, 'out_metrics.tsv')
    for k, v in eval_results.items():
        if k == 'confusion-matrix':
            v = v.tolist()
        s += '%s\t%s\n' % (k, str(v))

    with open(metrics_out, 'w') as f:
        f.write(s)

    # wrote kpomt pit[it]
    if not os.path.exists(args.output):
        with open(args.output, 'a') as f:
            f.write('\t'.join(['accuracy', 'auc', 'f1', 'seed']) + '\n')

    with open(args.output, 'a') as f:
        f.write('\t'.join([str(x) for x in [eval_results['acc'], eval_results['auc'], eval_results['f1'], seed]]) + '\n')


if __name__ == '__main__':
    """
   
python train2.py --data data/police/data_all_45_850_577_rea_order.csv --pretrain_epochs 1 --finetuning_epochs 15 --seed 420 --backbone deberta --output accs --balance --augmentation 3

    """
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument('--data', default='escalation_data.csv', 
        help='path to input data csv.')
    parser.add_argument('--output', default='global_results.tsv',
        help='Global resultsfile for appending.')
    parser.add_argument('--working_dir', default='wd', 
        help='working dir')
    parser.add_argument('--save_model', action='store_true',
        help='If you set this flag than the system will save a model into the working directory.')
    # CSV management
    parser.add_argument('--id_colname', default='uniqueid', 
        help='name of column in csv corresponding to unique example id')
    parser.add_argument('--pretrain_colname', default='allwords', 
        help='name of column in csv corresponding to input text for pretraining')
    parser.add_argument('--finetune_colname', default='45words', 
        help='name of column in csv corresponding to input text for fine-tuning')
    parser.add_argument('--label_colname', default='escalation', 
        help='name of column in csv corresponding to output labels')

    # NEW
    parser.add_argument('--selection_colname', default='classifier', 
        help='name of column in csv that marks data which should be considered for fine-tuning')
    parser.add_argument('--balance', action='store_true', 
        help='use balanced training set')
    parser.add_argument('--augmentation', default=-1, type=int,
        help='data augmentation strategy')
    parser.add_argument('--attr_inf', action='store_true', 
        help='attribute inference')

    # Data args
    parser.add_argument('--label_balance', default='50:50', 
        help='0/1 balance of labels. Accepted values: [50:50, natural]. 50:50 will do 83 escalation + 83 nonescalation.')
    parser.add_argument('--kfold', default=4, type=int,
        help='xval splits.')
    parser.add_argument('--pretrain_all', action='store_true',
        help='Set this to pretrain on all of the data (including train/test). Otherwise it will leave out the test set')

    # Training args
    parser.add_argument('--pretrain_epochs', default=0, type=int,
        help='Number of pretraining epochs (0 to skip).')
    parser.add_argument('--finetuning_epochs', default=3, type=int,
        help='Number of finetuning epochs (0 to skip).')
    parser.add_argument('--seed', default='-1', type=str,
        help='Random seed (-1 to automatically generate).')

    # Modeling args
    parser.add_argument('--extra_features', action='store_true',
        help='Adds reason_feature and order_feature as inputs to bert prediction layer.')
    parser.add_argument('--backbone', default='distilbert', choices=[
        'forest', 'logreg', 'nb', 'svm', 'distilbert',
        'bert', 'deberta', 'roberta'],
        help='Backbone model.')

    args = parser.parse_args()
    seeds = args.seed
    for seed in tqdm(seeds.split(',')):
        args.seed = int(seed)
        acc = run_expt(args)
        with open(f'{args.output}-{args.backbone}-{args.balance}-{args.augmentation}-{args.finetune_colname}.txt', 'a') as f:
            f.write(str(acc) + '\n')

















































