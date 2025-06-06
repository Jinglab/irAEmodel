rm(list = ls())

setwd("/home/xfan/irAE_model/1.1subsampling_training/1.proporation/")

library(TLSformer)
library(Seurat)
library(tidyverse)
library(MLmetrics)

sc_dat_blood_on_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_on_imm_irAE_related_filter5.rds")
print("sc_dat_blood_on_imm")
table(sc_dat_blood_on_imm$irAE_label)
sc_dat_blood_pre_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_imm_irAE_related_filter5.rds")
print("sc_dat_blood_pre_imm")
table(sc_dat_blood_pre_imm$irAE_label)
sc_dat_blood_pre_radio <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_radio_irAE_related_filter5.rds")
print("sc_dat_blood_pre_radio")
table(sc_dat_blood_pre_radio$irAE_label)

seed_list <- c(42,56,93)
proportion_list <- c(0.7,0.8,0.9)

sc_dat_list <- list(sc_dat_blood_on_imm,sc_dat_blood_pre_imm,sc_dat_blood_pre_radio)
names(sc_dat_list) <- c("on_imm","pre_imm","pre_radio")
colnames(sc_dat_blood_on_imm)

pred_metrics_df <- data.frame("time_point" = rep("a",27),
                              "proporation(for training)" = rep("a",27),
                              "seed" = rep("a",27),
                              "auc" = rep("a",27),
                              "acc" = rep("a",27),
                              "f1_score" = rep("a",27),
                              "recall" = rep("a",27),
                              "precision" = rep("a",27)
                              )

# split data
# training data -- proporation 70% 50% 30%

for(t in names(sc_dat_list)){
  print(t)
  if(t == names(sc_dat_list)[1]){
    n = 0
  }
  sc_dat_tmp <- sc_dat_list[[t]]
  for(p in proportion_list){
    print(p)
    set.seed(p*10)
    num_select_row <- sample(1:nrow(sc_dat_tmp),floor(nrow(sc_dat_tmp)*p))
    sc_dat_tmp_train <- sc_dat_tmp[num_select_row,]
    sc_dat_tmp_test <- sc_dat_tmp[-num_select_row,]
    for(i in seed_list){
      if(!dir.exists(paste0("irAE_CESC_immune_",t,"_",p,"_seed",i))){
        dir.create(paste0("irAE_CESC_immune_",t,"_p",p,"_seed",i))
      }
      print(paste0("Seed: ",i))
      train_episodes <- floor(nrow(sc_dat_tmp)/30)
      print(paste0("train_episodes: ",train_episodes))
      print("Start training")

      start_time <- Sys.time()

      sc_dat_tmp_trained <- run_tlsformer_train(seu_obj = sc_dat_tmp_train,
                                                pretrained_model = "TLSformer_BERT",
                                                sen_len = 260,
                                                pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                save_checkpoint_path = paste0("/home/xfan/irAE_model/1.1subsampling_training/1.proporation/irAE_CESC_immune_",t,"_p",p,"_seed",i),
                                                batch_size = 1,
                                                train_K = 15,
                                                train_Q = 15,
                                                train_episodes = train_episodes,
                                                val_episodes = 100,
                                                val_steps = 50,
                                                metadata = TRUE,
                                                reproduce = TRUE,
                                                set_seed = i,
                                                target_name = "irAE_label",
                                                envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                with_gpu = TRUE
                                              )
      saveRDS(sc_dat_tmp_trained,file = paste0("/home/xfan/irAE_model/1.1subsampling_training/1.proporation/irAE_CESC_immune_",t,"_p",p,"_seed",i,"/sc_dat_tmp_trained.rds"))
      sc_dat_tmp_preded <- run_tlsformer_pred(seu_obj = sc_dat_tmp_test,
                                              pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                              save_checkpoint_path = paste0("/home/xfan/irAE_model/1.1subsampling_training/1.proporation/irAE_CESC_immune_",t,"_p",p,"_seed",i),
                                              envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                              pretrained_model = "TLSformer_BERT",
                                              sen_len = 260,
                                              data_type = "sc_st",
                                              metadata = TRUE,
                                              class_num = 2,
                                              with_gpu = TRUE
                                              )
      saveRDS(sc_dat_tmp_preded,file = paste0("/home/xfan/irAE_model/1.1subsampling_training/1.proporation/irAE_CESC_immune_",t,"_p",p,"_seed",i,"/sc_dat_tmp_preded.rds"))

      end_time <- Sys.time()

      print(paste0("Cost time: ",end_time - start_time))

      n = n + 1

      sc_dat_tmp_preded$norm_relative_distance <- 1 - ((sc_dat_tmp_preded$relative_distance-min(sc_dat_tmp_preded$relative_distance))/(max(sc_dat_tmp_preded$relative_distance)-min(sc_dat_tmp_preded$relative_distance)))

      auc <- AUC(sc_dat_tmp_preded$norm_relative_distance, sc_dat_tmp_preded$irAE_label)
      acc <- Accuracy(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label)
      f1_score <- F1_Score(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label)
      recall <- Recall(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label,  positive = NULL)
      prec <- Precision(sc_dat_tmp_preded$irAE_label, sc_dat_tmp_preded$pred_label, positive = NULL)

      pred_metrics_df[n,1] <- t
      pred_metrics_df[n,2] <- p
      pred_metrics_df[n,3] <- i
      pred_metrics_df[n,4] <- auc
      pred_metrics_df[n,5] <- acc
      pred_metrics_df[n,6] <- f1_score
      pred_metrics_df[n,7] <- recall
      pred_metrics_df[n,8] <- prec

    }
  }
}

saveRDS(pred_metrics_df,file = "/home/xfan/irAE_model/1.subsampling_prediction/1.proporation/proporation_pred_sc_results_20250603.rds")

print("have done")
