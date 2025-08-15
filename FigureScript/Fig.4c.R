library(shazam)
library()
load("shazam_output.rdata"))#load your output data from shazam
library(alakazam)
curve.noirAE <- estimateAbundance(subset(shazam_output,group=="radio+ICI" & irAEs=="Absent"  ),
                                  group="tissue", ci=0.95, nboot=100, clone="clone_id")
curve.irAE <- estimateAbundance(subset(shazam_output,group=="radio+ICI" & irAEs=="Present"  ),
                                group="tissue", ci=0.95, nboot=100, clone="clone_id")
curve.norelapse <- estimateAbundance(subset(shazam_output,group=="radio+ICI" & Disease_relapse=="No relapse"  ),
                                     group="tissue", ci=0.95, nboot=100, clone="clone_id")
curve.relapse <- estimateAbundance(subset(shazam_output,group=="radio+ICI" & Disease_relapse=="Relapsed"  ),
                                   group="tissue", ci=0.95, nboot=100, clone="clone_id")
data.frame(cols=c("#B2A1D2", "#A6DBCC", "#DE9D74", "#ACE26D", "#BF57D1"),
           biopsy=c("on-imm_PBMC","on-radio_tumor","pre-imm_PBMC", "pre-radio_PBMC", "pre-radio_tumor")) -> tcoldf
plotAbundanceCurve(curve.noirAE,xlim = c(1,150),ylim = c(0,0.25),colors = tcoldf$cols, main_title = "non-irAE patients",legend_title="samples")|
  plotAbundanceCurve(curve.irAE,xlim = c(1,150),ylim = c(0,0.25),colors = tcoldf$cols, main_title = "irAE patients",legend_title="samples")|
  plotAbundanceCurve(curve.norelapse,xlim = c(1,150),ylim = c(0,0.25),colors = tcoldf$cols, main_title = "Non_relapsed patients",legend_title="samples")|
  plotAbundanceCurve(curve.relapse,xlim = c(1,150),ylim = c(0,0.25),colors = tcoldf$cols, main_title = "Relapsed patients",legend_title="samples")
ggsave(paste0(mypath,myfolder,savesub,"/ranked_clone_Res.pdf"),width = 12.5,height = 2)