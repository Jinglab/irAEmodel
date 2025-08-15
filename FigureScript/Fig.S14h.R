rm(list=ls())
options(stringAsFactor=F)
library(Seurat)
library(tidyr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(clusterProfiler)
library(org.Hs.eg.db)
library(patchwork)
readxl::read_excel("SuppTables.xlsx",
                   sheet = "TableS3",skip=1) -> genename
list(genename$`model 3`,genename$`model 2`,genename$`model 1`) -> genefile

lapply(1:length(genefile), function(x){
  genefile[[x]] -> gene
  genes = bitr(gene, fromType="SYMBOL", toType=c("ENTREZID"), OrgDb="org.Hs.eg.db")
  ego_ALL <- enrichGO(gene = genes$ENTREZID, 
                      OrgDb = org.Hs.eg.db,
                      ont = "BP", #"ALL","BP","CC","MF"
                      pAdjustMethod = "fdr", #矫正方式 holm”, “hochberg”, “hommel”, “bonferroni”, “BH”, “BY”, “fdr”, “none”中的一种
                      pvalueCutoff = 0.05, 
                      qvalueCutoff = 0.2,
                      readable = TRUE,
                      keyType = 'ENTREZID') 
  clusterProfiler::simplify(ego_ALL)
}) -> deg.enrich
#save(deg.enrich,file = "enrich.rdataenrich.rdata")
#load(file = "/Users/yingjing/Library/CloudStorage/OneDrive-sjtu.edu.cn/MyFiles/Mylab/Projects/LeiZhang/radiotherapy_UC/Single_Cell/Manuscript/5.NI-10.NatCancer/R1/Revision_data/enrich.rdata")
lapply(deg.enrich,function(x){
  clusterProfiler::dotplot(x)
}) -> plist
wrap_plots(plist[1:2])
ggsave("enrich_model_gene.pdf",width =16,height = 6)


