# bert-of-theseus-tf
tensorflow version of bert-of-theseus

This repository contains implementation of bert-of-theseus in tensorflow version. original paper reference:["BERT-of-Theseus: Compressing BERT by Progressive Module Replacing"](http://arxiv.org/abs/2002.02925). 

The official pytorch implementation: https://github.com/JetRunner/BERT-of-Theseus

BERT-of-Theseus is a new compressed BERT by progressively replacing the components of the original BERT.

## instruction on usage

1. first stage:model replacing

   ```python
   import modeling_theseus
   import optimization_theseus
   #...code for define model graph...
   bert_model = modeling_theseus.BertModel(bert_config,is_training,input_ids,text_length,replace_rate_prob,suc_layers,finetune_suc,token_type_ids)
   #...code for other parts of model graph
   #generate replace prob
   global_step = tf.train.get_or_create_global_step()
   replace_rate_prob = replace_linear_k*tf.cast(global_ste,tf.float32)+base_replace_prob
   #init weight from bert pretrained checkpoints
   tvars = tf.trainable_variables()
   assignment_map,suc_assignment_map,_ = modeling_theseus.get_assignment_map_from_checkpoint_for_theseus(tvars,init_checkpoints)
   tf.train.init_from_checkpoint(init_checkpoints,assignment_map)
   tf.train_init_from_checkpoint(init_checkpoints,suc_assignment_map)
   #..code for call model definition
   #gen train_op for training
   train_op = optimization_theseus.create_optimizer_for_bert_theseus(loss,lr,decay_steps,clip_norm,False)
   
   ```

   2.second stage: scc modules finetune

   most of code is like stage 1 except for using **modeling_theseus.get_assignment_map_from_checkpoint()** to init, and pass **finetune_suc=True** to both **BertModel** and **create_optimizer_for_bert_theseus**

## evals on Chinese NER based on bert_mrc

implement three different kinds of methods:

1、two-stage training phase。constant replace prob=0.5

2、two-stage training phase。linear strategy of replace prob

3、one stagetraining phase，exclude scc finetuning.

```
|         method           | f1-micro-avg |
| :---------------------:  | :----------: |
| two-stage,const prob=0.5 |    0.9459    |
| two-stage,linear stratege|    0.9491    |
| one-stage                |    0.9342    |
| orig bert+mrc+focalloss  |    0.9580    |
```

more details of bert_mrc on chinese-ner task.please refer to :https://github.com/qiufengyuyi/sequence_tagging

