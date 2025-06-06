rm(list = ls())

library(tidyverse)
library(TLSformer)
library(Seurat)
library(reticulate)

reticulate::use_condaenv("/home/xfan/miniconda3/envs/TLSformer")

sc_dat <- readRDS("/home/xfan/processed_scRNAseq_dat_irAE/irAE_CESC_jinglab_70thousand/SceObj_v4_cellbender_merged_Annotated_NoUndet_res_3.42_TumorAnnt_ForModel_noRadio.rds")

#sc_dat@meta.data <- sc_dat@meta.data[,-ncol(sc_dat@meta.data)]
#sc_dat@meta.data$extract_cells <- ifelse((1:nrow(sc_dat@meta.data))%in%sample(1:nrow(sc_dat@meta.data),500),1,0)

#sc_dat_subset <- subset(sc_dat,subset = extract_cells%in%1)

# generate sentences

sc_dat <- generate_sentences(
  sc_dat,
  sen_len = 260,
  region_info = NULL,
  save_inseu = TRUE,
  save_path = "/home/xfan/irAE_model/0.2irAE_CESC_training_data/test_sentence.txt",
  #genes_representor = "/home/xfan/irAE_model/0.1pretrained_models/bc_pretrained_model/gene_list_bc_intersected_bulk_CESC_GSE186144_BERT.txt",
  data_type = "sc_st",
  envir_path = "/home/xfan/miniconda3/envs/TLSformer/"
)

saveRDS(sc_dat@meta.data,file = "/home/xfan/irAE_model/0.2irAE_CESC_training_data/irAE_CESC_metadata_sentence_top260_bc_ref.rds")

# after runing upper content
# input data prepare
dat <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/irAE_CESC_metadata_sentence_top260_bc_ref.rds")
dat$irAE_label <- ifelse(dat$irAEs=="Present",1,0)
dat <- dat[dat$biopsy == "PBMC",]

sc_dat_blood_pre_imm <- dat[dat$time=="pre-imm",]#subset(sc_dat_blood,subset = time == "pre-imm")
sc_dat_blood_on_imm <- dat[dat$time=="on-imm",]#subset(sc_dat_blood,subset = time == "on-imm")
sc_dat_blood_pre_radio <- dat[dat$time=="pre-radio",]#subset(sc_dat_blood,subset = time == "pre-radio")


# cell type diff
sc_meta <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/irAE_associated_cluster_in_samples_FDRadjust.rds")
sc_meta_sig <- sc_meta[sc_meta$sig!="nosig",]
sc_meta_sig$new_celltype_name <- paste0(sc_meta_sig$subtype,sc_meta_sig$biopsy)
View(sc_meta_sig)
cell_less5 <- readRDS("/home/xfan/irAE_model/0.2irAE_CESC_training_data/CellLessThan5Percent.rds")
View(cell_less5)
cell_less5$tissue <- str_replace(cell_less5$tissue,"on-imm","Post-imm")
cell_less5$tissue <- str_replace(cell_less5$tissue,"pre-imm","Pre-imm")
cell_less5$tissue <- str_replace(cell_less5$tissue,"pre-radio","Pre-radio")
cell_less5$new_celltype_name <- paste0(cell_less5$cell,cell_less5$tissue)

sc_meta_sig <- sc_meta_sig[!((sc_meta_sig$new_celltype_name)%in%cell_less5$new_celltype_name),]

View(sc_meta_sig)

sc_meta_sig_blood_pre_imm <- sc_meta_sig[sc_meta_sig$biopsy=="Pre-imm_PBMC",]
sc_meta_sig_blood_post_imm <- sc_meta_sig[sc_meta_sig$biopsy=="Post-imm_PBMC",]
sc_meta_sig_blood_pre_radio <- sc_meta_sig[sc_meta_sig$biopsy=="Pre-radio_PBMC",]

sc_dat_blood_pre_imm <-  sc_dat_blood_pre_imm[sc_dat_blood_pre_imm$celltype_sub %in% sc_meta_sig_blood_pre_imm$subtype,]  #subset(sc_dat_blood_pre_imm,subset = celltype_sub %in% sc_meta_sig_blood_pre_imm$subtype)
sc_dat_blood_on_imm <- sc_dat_blood_on_imm[sc_dat_blood_on_imm$celltype_sub %in% sc_meta_sig_blood_post_imm$subtype,] #subset(sc_dat_blood_on_imm,subset = celltype_sub %in% sc_meta_sig_blood_post_imm$subtype)
sc_dat_blood_pre_radio <- sc_dat_blood_pre_radio[sc_dat_blood_pre_radio$celltype_sub %in% sc_meta_sig_blood_pre_radio$subtype,] #subset(sc_dat_blood_pre_radio,subset = celltype_sub %in% sc_meta_sig_blood_pre_radio$subtype)

sc_dat_blood_pre_imm$irAE_related <- ifelse(sc_dat_blood_pre_imm$celltype_sub %in% sc_meta_sig_blood_pre_imm[sc_meta_sig_blood_pre_imm$sig=="up_sig",]$subtype,1,0)
sc_dat_blood_on_imm$irAE_related <- ifelse(sc_dat_blood_on_imm$celltype_sub %in% sc_meta_sig_blood_post_imm[sc_meta_sig_blood_post_imm$sig=="up_sig",]$subtype,1,0)
sc_dat_blood_pre_radio$irAE_related <- ifelse(sc_dat_blood_pre_radio$celltype_sub %in% sc_meta_sig_blood_pre_radio[sc_meta_sig_blood_pre_radio$sig=="up_sig",]$subtype,1,0)

sc_dat_blood_pre_imm$keep_consistent <- ifelse((sc_dat_blood_pre_imm$celltype_sub %in% sc_meta_sig_blood_pre_imm[sc_meta_sig_blood_pre_imm$sig=="up_sig",]$subtype & sc_dat_blood_pre_imm$irAEs == "Present")|(sc_dat_blood_pre_imm$celltype_sub %in% sc_meta_sig_blood_pre_imm[sc_meta_sig_blood_pre_imm$sig=="down_sig",]$subtype & sc_dat_blood_pre_imm$irAEs == "Absent"),1,0)
sc_dat_blood_pre_imm_irAE_related <- sc_dat_blood_pre_imm[sc_dat_blood_pre_imm$keep_consistent==1,] #subset(sc_dat_blood_pre_imm,subset = keep_consistent==1)
sum(sc_dat_blood_pre_imm$keep_consistent)
ncol(sc_dat_blood_pre_imm)

sc_dat_blood_on_imm$keep_consistent <- ifelse((sc_dat_blood_on_imm$celltype_sub %in% sc_meta_sig_blood_post_imm[sc_meta_sig_blood_post_imm$sig=="up_sig",]$subtype & sc_dat_blood_on_imm$irAEs == "Present")|(sc_dat_blood_on_imm$celltype_sub %in% sc_meta_sig_blood_post_imm[sc_meta_sig_blood_post_imm$sig=="down_sig",]$subtype & sc_dat_blood_on_imm$irAEs == "Absent"),1,0)
sc_dat_blood_on_imm_irAE_related <- sc_dat_blood_on_imm[sc_dat_blood_on_imm$keep_consistent==1,] #subset(sc_dat_blood_on_imm,subset = keep_consistent==1)
sum(sc_dat_blood_on_imm$keep_consistent)
ncol(sc_dat_blood_on_imm)

sc_dat_blood_pre_radio$keep_consistent <- ifelse((sc_dat_blood_pre_radio$celltype_sub %in% sc_meta_sig_blood_pre_radio[sc_meta_sig_blood_pre_radio$sig=="up_sig",]$subtype & sc_dat_blood_pre_radio$irAEs == "Present")|(sc_dat_blood_pre_radio$celltype_sub %in% sc_meta_sig_blood_pre_radio[sc_meta_sig_blood_pre_radio$sig=="down_sig",]$subtype & sc_dat_blood_pre_radio$irAEs == "Absent"),1,0)
sc_dat_blood_pre_radio_irAE_related <- sc_dat_blood_pre_radio[sc_dat_blood_pre_radio$keep_consistent==1,]#subset(sc_dat_blood_pre_radio,subset = keep_consistent==1)
sum(sc_dat_blood_pre_radio$keep_consistent)
ncol(sc_dat_blood_pre_radio_irAE_related)


ls()

table(sc_dat_blood_on_imm_irAE_related$celltype_sub)
table(sc_dat_blood_pre_imm_irAE_related$celltype_sub)
table(sc_dat_blood_pre_radio_irAE_related$celltype_sub)
View(sc_meta_sig_blood_pre_radio)
# save training data

saveRDS(sc_dat_blood_on_imm_irAE_related,file = "/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_on_imm_irAE_related_filter5.rds")
saveRDS(sc_dat_blood_pre_imm_irAE_related,file = "/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_imm_irAE_related_filter5.rds")
saveRDS(sc_dat_blood_pre_radio_irAE_related,file = "/home/xfan/irAE_model/0.2irAE_CESC_training_data/sc_dat_blood_pre_radio_irAE_related_filter5.rds")
