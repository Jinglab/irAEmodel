rm(list = ls())

if(!dir.exists("/home/xfan/irAE_model/2.1important_subcluster_seed/")){
  dir.create("/home/xfan/irAE_model/2.1important_subcluster_seed/")
}

setwd("/home/xfan/irAE_model/2.1important_subcluster_seed/")

library(TLSformer)
library(Seurat)
library(tidyverse)
library(MLmetrics)

seed_list <- c(81,43,57)

sc_dat_blood_on_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_on_imm_irAE_related_filter5.rds")
print("sc_dat_blood_on_imm")
table(sc_dat_blood_on_imm$irAE_label)
unique(sc_dat_blood_on_imm$celltype_sub)

sc_dat_blood_pre_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_imm_irAE_related_filter5.rds")
print("sc_dat_blood_pre_imm")
table(sc_dat_blood_pre_imm$irAE_label)
unique(sc_dat_blood_pre_imm$celltype_sub)

sc_dat_blood_pre_radio <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_radio_irAE_related_filter5.rds")
sc_dat_blood_pre_radio[1:30,"celltype_sub"] <- "tmp_celltype"
sc_dat_blood_pre_radio[1:30,"irAE_label"] <- 0
colnames(sc_dat_blood_pre_radio)
print("sc_dat_blood_pre_radio")
table(sc_dat_blood_pre_radio$irAE_label)
unique(sc_dat_blood_pre_radio$celltype_sub)

sc_dat_list <- list(sc_dat_blood_on_imm,sc_dat_blood_pre_imm,sc_dat_blood_pre_radio)
names(sc_dat_list) <- c("on_imm","pre_imm","pre_radio")

# training data
for(sd in seed_list){
  if(!dir.exists(paste0("./seed_",sd))){
    dir.create(paste0("./seed_",sd))
  }
  for(i in c("on_imm","pre_imm")){
    print(i)
    if(!dir.exists(paste0("./seed_",sd,"/irAE_CESC_",i))){
      dir.create(paste0("./seed_",sd,"/irAE_CESC_",i))
    }
    
    sc_dat_tmp <- sc_dat_list[[i]]
    
    if(i == "pre_radio"){
      sub_celltypes <- unique(sc_dat_tmp$celltype_sub)[!unique(sc_dat_tmp$celltype_sub)%in%"tmp_celltype"]
    }else{
      sub_celltypes <- unique(sc_dat_tmp$celltype_sub)
    }
    
    if(i == "on_imm"&sd==seed_list[1]){
      n=0
      pred_metrics_df <- data.frame("time_point" = rep("a",length(sub_celltypes)),
                                    "celltype" = rep("a",length(sub_celltypes)),
                                    "seed" = rep("a",length(sub_celltypes)), 
                                    "auc" = rep("a",length(sub_celltypes)),
                                    "acc" = rep("a",length(sub_celltypes)),
                                    "f1_score" = rep("a",length(sub_celltypes)),
                                    "recall" = rep("a",length(sub_celltypes)),
                                    "precision" = rep("a",length(sub_celltypes))
      )
    }else{
      pred_metrics_df_tmp <- data.frame("time_point" = rep("a",length(sub_celltypes)),
                                        "celltype" = rep("a",length(sub_celltypes)),
                                        "seed" = rep("a",length(sub_celltypes)),
                                        "auc" = rep("a",length(sub_celltypes)),
                                        "acc" = rep("a",length(sub_celltypes)),
                                        "f1_score" = rep("a",length(sub_celltypes)),
                                        "recall" = rep("a",length(sub_celltypes)),
                                        "precision" = rep("a",length(sub_celltypes))
      )
      pred_metrics_df <- rbind(pred_metrics_df,pred_metrics_df_tmp)
    }
    
    for(c in sub_celltypes){
      
      if(!dir.exists(paste0("./seed_",sd,"/irAE_CESC_",i,"/",c))){
        dir.create(paste0("./seed_",sd,"/irAE_CESC_",i,"/",c))
      }
      
      print(c)
      
      tryCatch({
        sc_dat_train <- sc_dat_tmp[!(sc_dat_tmp$celltype_sub%in%c),]
        
        train_episodes <- floor((1.2*nrow(sc_dat_train)/30))
        
        start_time <- Sys.time()
        
        sc_dat_tmp_trained <- run_tlsformer_train(seu_obj = sc_dat_train,
                                                  pretrained_model = "TLSformer_BERT",
                                                  sen_len = 260,
                                                  pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                  save_checkpoint_path = paste0("/home/xfan/irAE_model/2.1important_subcluster_seed/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                  batch_size = 1,
                                                  train_K = 15,
                                                  train_Q = 15,
                                                  train_episodes = train_episodes,
                                                  val_episodes = 100,
                                                  val_steps = 50,
                                                  metadata = TRUE,
                                                  reproduce = TRUE,
                                                  set_seed = sd,
                                                  target_name = "irAE_label",
                                                  envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                  with_gpu = TRUE)
        saveRDS(sc_dat_tmp_trained,file = paste0("/home/xfan/irAE_model/2.1important_subcluster_seed/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_trained.rds"))
        
        sc_dat_tmp_preded <- run_tlsformer_pred(seu_obj = sc_dat_tmp,
                                                pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                save_checkpoint_path = paste0("/home/xfan/irAE_model/2.1important_subcluster_seed/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                pretrained_model = "TLSformer_BERT",
                                                sen_len = 260,
                                                data_type = "sc_st",
                                                metadata = TRUE,
                                                class_num = 2,
                                                with_gpu = TRUE)
        saveRDS(sc_dat_tmp_preded,file = paste0("/home/xfan/irAE_model/2.1important_subcluster_seed/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_preded.rds"))
        
        end_time <- Sys.time()
        
        print(paste0("Cost time: ",end_time - start_time))
        
        n = n + 1
        
        sc_dat_tmp_preded$norm_relative_distance <- 1 - ((sc_dat_tmp_preded$relative_distance-min(sc_dat_tmp_preded$relative_distance))/(max(sc_dat_tmp_preded$relative_distance)-min(sc_dat_tmp_preded$relative_distance)))
        
        auc <- AUC(sc_dat_tmp_preded$norm_relative_distance, sc_dat_tmp_preded$irAE_label)
        acc <- Accuracy(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label)
        f1_score <- F1_Score(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label)
        recall <- Recall(sc_dat_tmp_preded$irAE_label,sc_dat_tmp_preded$pred_label,  positive = NULL)
        prec <- Precision(sc_dat_tmp_preded$irAE_label, sc_dat_tmp_preded$pred_label, positive = NULL)
        
        pred_metrics_df[n,1] <- i
        pred_metrics_df[n,2] <- c
        pred_metrics_df[n,3] <- sd
        pred_metrics_df[n,4] <- auc
        pred_metrics_df[n,5] <- acc
        pred_metrics_df[n,6] <- f1_score
        pred_metrics_df[n,7] <- recall
        pred_metrics_df[n,8] <- prec
        
        print("done")
        
      }, error = function(e) {
        print(paste0("error: ",c))
      })
      
    }
    
    print(paste0("have done ",i))
    
  }
  
  print(paste0("have done ",sd))
  
}

saveRDS(pred_metrics_df,file = "/home/xfan/irAE_model/2.1important_subcluster_seed/irAE_CESC_diff_model_seed_celltype_delete.rds")