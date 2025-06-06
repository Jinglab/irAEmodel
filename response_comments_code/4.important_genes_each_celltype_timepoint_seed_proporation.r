rm(list = ls())

if(!dir.exists("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/")){
  dir.create("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/")
}

setwd("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/")

library(TLSformer)
library(Seurat)
library(tidyverse)
library(MLmetrics)

seed_list <- c(81,43,57)
prop_list <- c(0.7,0.8,0.9)

sc_dat_blood_on_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_on_imm_irAE_related_filter5.rds")
print("sc_dat_blood_on_imm")
table(sc_dat_blood_on_imm$irAE_label)
unique(sc_dat_blood_on_imm$celltype_sub)
sc_dat_blood_on_imm$cell_barcode <- rownames(sc_dat_blood_on_imm)

sc_dat_blood_pre_imm <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_imm_irAE_related_filter5.rds")
print("sc_dat_blood_pre_imm")
table(sc_dat_blood_pre_imm$irAE_label)
unique(sc_dat_blood_pre_imm$celltype_sub)
sc_dat_blood_pre_imm$cell_barcode <- rownames(sc_dat_blood_pre_imm)

sc_dat_blood_pre_radio <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_radio_irAE_related_filter5.rds")
sc_dat_blood_pre_radio[1:20,"celltype_sub"] <- "tmp_celltype"
sc_dat_blood_pre_radio[1:20,"irAE_label"] <- 0
print("sc_dat_blood_pre_radio")
table(sc_dat_blood_pre_radio$irAE_label)
unique(sc_dat_blood_pre_radio$celltype_sub)
sc_dat_blood_pre_radio$cell_barcode <- rownames(sc_dat_blood_pre_radio)

sc_dat_list <- list(sc_dat_blood_on_imm,sc_dat_blood_pre_imm,sc_dat_blood_pre_radio)
names(sc_dat_list) <- c("on_imm","pre_imm","pre_radio")

# training data
for(p in prop_list){
    print(paste0("propotation: ",p))
  if(!dir.exists(paste0("./prop_",p))){
    dir.create(paste0("./prop_",p))
  }
  for(sd in seed_list){
    if(!dir.exists(paste0("./prop_",p,"/seed_",sd))){
      dir.create(paste0("./prop_",p,"/seed_",sd))
    }
    
    for(i in names(sc_dat_list)){
      print(i)
      if(!dir.exists(paste0("./prop_",p,"/seed_",sd,"/irAE_CESC_",i))){
        dir.create(paste0("./prop_",p,"/seed_",sd,"/irAE_CESC_",i))
      }

      sc_dat_tmp <- sc_dat_list[[i]]
      if(i != "pre_radio"){
        sub_celltypes <- unique(sc_dat_tmp$celltype_sub)
      }else{
        sub_celltypes <- unique(sc_dat_tmp$celltype_sub)
        sub_celltypes <- sub_celltypes[!(sub_celltypes%in%c("tmp_celltype"))]
      }
      
      
      if(i == names(sc_dat_list)[1]&p==prop_list[1]){
        n=0
        pred_metrics_df <- data.frame("time_point" = rep("a",length(sub_celltypes)),
                                      "celltype" = rep("a",length(sub_celltypes)),
                                      "proporation" = rep("a",length(sub_celltypes)),
                                      "seed" = rep("a",length(sub_celltypes)), 
                                      
                                      "auc_train" = rep("a",length(sub_celltypes)),
                                      "acc_train" = rep("a",length(sub_celltypes)),
                                      "f1_score_train" = rep("a",length(sub_celltypes)),
                                      "recall_train" = rep("a",length(sub_celltypes)),
                                      "precision_train" = rep("a",length(sub_celltypes)),
                                      
                                      "auc_test" = rep("a",length(sub_celltypes)),
                                      "acc_test" = rep("a",length(sub_celltypes)),
                                      "f1_score_test" = rep("a",length(sub_celltypes)),
                                      "recall_test" = rep("a",length(sub_celltypes)),
                                      "precision_test" = rep("a",length(sub_celltypes)),
                                      
                                      "auc_all" = rep("a",length(sub_celltypes)),
                                      "acc_all" = rep("a",length(sub_celltypes)),
                                      "f1_score_all" = rep("a",length(sub_celltypes)),
                                      "recall_all" = rep("a",length(sub_celltypes)),
                                      "precision_all" = rep("a",length(sub_celltypes))
        )
      }else{
        pred_metrics_df_tmp <- data.frame("time_point" = rep("a",length(sub_celltypes)),
                                          "celltype" = rep("a",length(sub_celltypes)),
                                          "proporation" = rep("a",length(sub_celltypes)),
                                          "seed" = rep("a",length(sub_celltypes)),
                                          
                                          "auc_train" = rep("a",length(sub_celltypes)),
                                          "acc_train" = rep("a",length(sub_celltypes)),
                                          "f1_score_train" = rep("a",length(sub_celltypes)),
                                          "recall_train" = rep("a",length(sub_celltypes)),
                                          "precision_train" = rep("a",length(sub_celltypes)),
                                          
                                          "auc_test" = rep("a",length(sub_celltypes)),
                                          "acc_test" = rep("a",length(sub_celltypes)),
                                          "f1_score_test" = rep("a",length(sub_celltypes)),
                                          "recall_test" = rep("a",length(sub_celltypes)),
                                          "precision_test" = rep("a",length(sub_celltypes)),
                                          
                                          "auc_all" = rep("a",length(sub_celltypes)),
                                          "acc_all" = rep("a",length(sub_celltypes)),
                                          "f1_score_all" = rep("a",length(sub_celltypes)),
                                          "recall_all" = rep("a",length(sub_celltypes)),
                                          "precision_all" = rep("a",length(sub_celltypes))
        )
        pred_metrics_df <- rbind(pred_metrics_df,pred_metrics_df_tmp)
      }
      
      set.seed(sd)
      sc_dat_tmp_train <- sc_dat_tmp %>%
        group_by(celltype_sub) %>%
        slice_sample(prop = p)
      sc_dat_tmp_test <- sc_dat_tmp[!(sc_dat_tmp$cell_barcode%in%sc_dat_tmp_train$cell_barcode),]
      
      for(c in sub_celltypes){
        
        if(!dir.exists(paste0("./prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c))){
          dir.create(paste0("./prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c))
        }
        
        print(c)
        
        tryCatch({
          
          sc_dat_tmp_train_del <- sc_dat_tmp_train[!(sc_dat_tmp_train$celltype_sub%in%c),]
          
          train_episodes <- floor((1.2*nrow(sc_dat_tmp_train_del)/30))
          
          start_time <- Sys.time()
          
          sc_dat_tmp_trained <- run_tlsformer_train(seu_obj = sc_dat_tmp_train_del,
                                                    pretrained_model = "TLSformer_BERT",
                                                    sen_len = 260,
                                                    pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                    save_checkpoint_path = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                    batch_size = 1,
                                                    train_K = 15,
                                                    train_Q = 15,
                                                    train_episodes = train_episodes,
                                                    val_episodes = 100,
                                                    val_steps = 50,
                                                    metadata = TRUE,
                                                    reproduce = TRUE,
                                                    set_seed = 42,
                                                    target_name = "irAE_label",
                                                    envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                    with_gpu = TRUE)
          saveRDS(sc_dat_tmp_trained,file = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_trained.rds"))
          
          sc_dat_tmp_train_preded <- run_tlsformer_pred(seu_obj = sc_dat_tmp_train,
                                                       pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                       save_checkpoint_path = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                       envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                       pretrained_model = "TLSformer_BERT",
                                                       sen_len = 260,
                                                       data_type = "sc_st",
                                                       metadata = TRUE,
                                                       class_num = 2,
                                                       with_gpu = TRUE)
          saveRDS(sc_dat_tmp_train_preded,file = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_train_preded.rds"))
          
          
          sc_dat_tmp_test_preded <- run_tlsformer_pred(seu_obj = sc_dat_tmp_test,
                                                  pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                  save_checkpoint_path = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                  envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                  pretrained_model = "TLSformer_BERT",
                                                  sen_len = 260,
                                                  data_type = "sc_st",
                                                  metadata = TRUE,
                                                  class_num = 2,
                                                  with_gpu = TRUE)
          saveRDS(sc_dat_tmp_test_preded,file = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_test_preded.rds"))

          sc_dat_tmp_all_preded <- run_tlsformer_pred(seu_obj = sc_dat_tmp,
                                                  pretrained_model_path = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/",
                                                  save_checkpoint_path = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c),
                                                  envir_path = "/home/xfan/miniconda3/envs/TLSformer",
                                                  pretrained_model = "TLSformer_BERT",
                                                  sen_len = 260,
                                                  data_type = "sc_st",
                                                  metadata = TRUE,
                                                  class_num = 2,
                                                  with_gpu = TRUE)
          saveRDS(sc_dat_tmp_all_preded,file = paste0("/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/","prop_",p,"/seed_",sd,"/irAE_CESC_",i,"/",c,"/sc_dat_tmp_all_preded.rds"))
          
          end_time <- Sys.time()
          
          print(paste0("Cost time: ",end_time - start_time))
          
          n = n + 1
          
          # train
          
          sc_dat_tmp_train_preded$norm_relative_distance <- 1 - ((sc_dat_tmp_train_preded$relative_distance-min(sc_dat_tmp_train_preded$relative_distance))/(max(sc_dat_tmp_train_preded$relative_distance)-min(sc_dat_tmp_train_preded$relative_distance)))
          
          auc_train <- AUC(sc_dat_tmp_train_preded$norm_relative_distance, sc_dat_tmp_train_preded$irAE_label)
          acc_train <- Accuracy(sc_dat_tmp_train_preded$irAE_label,sc_dat_tmp_train_preded$pred_label)
          f1_score_train <- F1_Score(sc_dat_tmp_train_preded$irAE_label,sc_dat_tmp_train_preded$pred_label)
          recall_train <- Recall(sc_dat_tmp_train_preded$irAE_label,sc_dat_tmp_train_preded$pred_label,  positive = NULL)
          prec_train <- Precision(sc_dat_tmp_train_preded$irAE_label, sc_dat_tmp_train_preded$pred_label, positive = NULL)
          
          # test
          sc_dat_tmp_test_preded$norm_relative_distance <- 1 - ((sc_dat_tmp_test_preded$relative_distance-min(sc_dat_tmp_test_preded$relative_distance))/(max(sc_dat_tmp_test_preded$relative_distance)-min(sc_dat_tmp_test_preded$relative_distance)))
          
          auc <- AUC(sc_dat_tmp_test_preded$norm_relative_distance, sc_dat_tmp_test_preded$irAE_label)
          acc <- Accuracy(sc_dat_tmp_test_preded$irAE_label,sc_dat_tmp_test_preded$pred_label)
          f1_score <- F1_Score(sc_dat_tmp_test_preded$irAE_label,sc_dat_tmp_test_preded$pred_label)
          recall <- Recall(sc_dat_tmp_test_preded$irAE_label,sc_dat_tmp_test_preded$pred_label,  positive = NULL)
          prec <- Precision(sc_dat_tmp_test_preded$irAE_label, sc_dat_tmp_test_preded$pred_label, positive = NULL)

          # all
          sc_dat_tmp_all_preded$norm_relative_distance <- 1 - ((sc_dat_tmp_all_preded$relative_distance-min(sc_dat_tmp_all_preded$relative_distance))/(max(sc_dat_tmp_all_preded$relative_distance)-min(sc_dat_tmp_all_preded$relative_distance)))
          
          auc_all <- AUC(sc_dat_tmp_all_preded$norm_relative_distance, sc_dat_tmp_all_preded$irAE_label)
          acc_all <- Accuracy(sc_dat_tmp_all_preded$irAE_label,sc_dat_tmp_all_preded$pred_label)
          f1_score_all <- F1_Score(sc_dat_tmp_all_preded$irAE_label,sc_dat_tmp_all_preded$pred_label)
          recall_all <- Recall(sc_dat_tmp_all_preded$irAE_label,sc_dat_tmp_all_preded$pred_label,  positive = NULL)
          prec_all <- Precision(sc_dat_tmp_all_preded$irAE_label, sc_dat_tmp_all_preded$pred_label, positive = NULL)
          
          pred_metrics_df[n,1] <- i
          pred_metrics_df[n,2] <- c
          pred_metrics_df[n,3] <- p
          pred_metrics_df[n,4] <- sd
          
          pred_metrics_df[n,5] <- auc_train
          pred_metrics_df[n,6] <- acc_train
          pred_metrics_df[n,7] <- f1_score_train
          pred_metrics_df[n,8] <- recall_train
          pred_metrics_df[n,9] <- prec_train
          
          pred_metrics_df[n,10] <- auc_test
          pred_metrics_df[n,11] <- acc_test
          pred_metrics_df[n,12] <- f1_score_test
          pred_metrics_df[n,13] <- recall_test
          pred_metrics_df[n,14] <- prec_test
          
          pred_metrics_df[n,15] <- auc_all
          pred_metrics_df[n,16] <- acc_all
          pred_metrics_df[n,17] <- f1_score_all
          pred_metrics_df[n,18] <- recall_all
          pred_metrics_df[n,19] <- prec_all
          
          print("done")

        }, error = function(e) {

          print(paste0("error: ",c))
        
        })
        
      }
      
      print(paste0("have done ",i))
      
    }
    
    print(paste0("have done ",sd))
    
  }
  
}

saveRDS(pred_metrics_df,file = "/home/xfan/irAE_model/2.2important_subcluster_seed_proporation/irAE_CESC_diff_seed_proporation_celltype_delete.rds")

print("pred have done")