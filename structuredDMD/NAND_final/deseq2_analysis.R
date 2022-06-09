library('DESeq2')
library(edgeR)
library(tidyverse)

# load count matrix
cts <- as.matrix(read_csv("data/count_matrix_no_circuit_genes_NAND.csv"))
genes <- cts[,1]
cts <- as.matrix(cts[,-1], row.names = cts[,1])
mode(cts) <- "integer"
rownames(cts) <- genes
head(cts,2)

# # TMM normalization on the counts with circuit genes present
# all_cts <- as.matrix(read_csv('data/count_matrix_NAND.csv'))
# all_genes <- all_cts[,1]
# all_cts <- as.matrix(all_cts[,-1], row.names = all_cts[,1])
# mode(all_cts) <- "integer"
# rownames(all_cts) <- all_genes
# DGEList <- DGEList(all_cts)
# DGEList.norm <- calcNormFactors(DGEList,method='TMM')
# cpm.norm <- cpm(DGEList.norm)
# head(cpm.norm,2)
# 
# # write.csv(cpm.norm,file='data/TMM_matrix_NAND.csv')

# # Gene length corrected TMM
# geneLengths <- as.matrix(read_csv("data/gene_lengths_NAND.csv"))
# geneLengths <- as.matrix(geneLengths[,-1], row.names = geneLengths[,1])
# mode(geneLengths) <- "integer"
# # # RPK's are calculated as 
# # # RPK <- counts / gene_length
# rpk <- all_cts / geneLengths[,]
# DGEList.rpk <- DGEList(rpk)
# DGEList.rpk.norm <- calcNormFactors(DGEList.rpk, method = "TMM")
# cpm.norm.rpk <- cpm(DGEList.rpk.norm)
# head(cpm.norm,2)
# 
# write.csv(cpm.norm.rpk,file='data/GeTMM_matrix_NAND.csv')

# loading metadata
colData <- data.frame(read_csv("data/metaData_NAND.csv"))
rownames(colData) <- colData[,1] 
all(rownames(colData) == colnames(cts))

# construct DESeqDataSet Object
dds <- DESeqDataSetFromMatrix(countData=cts,
                              colData=colData,
                              design=~condition)

# subset relevant samples
comp_samples = c('wt_00375','wt_003718') # first is control, second is treatment
dds <- dds[,dds$condition %in% comp_samples]

# set factor levels manually; the first factor is the reference to which other levels are compared
dds$condition <- factor(dds$condition, levels = comp_samples)
dds$condition

dds <- DESeq(dds)
resultsNames(dds)
# write.csv(counts(dds,normalized=TRUE),file='count_matrix_filtered_normalized.csv')

# NAME = 'condition_wt_00375_vs_wt_003718' # 'condition_M9LQ_vs_LB0', 'condition_M9SF_vs_LB0'
res <- results(dds)

# write.csv(as.data.frame(res), file="data/condition_nand003718_vs_wt003718_results.csv")

# http://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html
# resLFC <- lfcShrink(dds, coef='Condition_LB1_vs_LB0',type='apeglm')
# resLFC

# resOrdered <- res[order(res$log2FoldChange),]
# head(resOrdered,100)

# informative plots
plotCounts(dds,'aaeB',intgroup='condition')
plotCounts(dds,'csgD',intgroup='condition')
plotCounts(dds,'ymdF',intgroup='condition')
plotCounts(dds,'tauA',intgroup='condition')
plotCounts(dds,'tauB',intgroup='condition')
plotCounts(dds,'tauC',intgroup='condition')
plotCounts(dds,'napF',intgroup='condition')
plotCounts(dds,'alx',intgroup='condition')
plotCounts(dds,'ompC',intgroup='condition')
plotCounts(dds,'ompF',intgroup='condition')
plotCounts(dds,'ompR',intgroup='condition')
plotCounts(dds,'envZ',intgroup='condition')
plotCounts(dds,'phoP',intgroup='condition')
plotCounts(dds,gene=which.max(res$log2FoldChange),intgroup='condition')
plotCounts(dds,gene=which.min(res$log2FoldChange),intgroup='condition')
plotMA(res,colSig='blue',colLine='red')
plotDispEsts(dds, ylim = c(1e-6, 1e1) )

# volcano plot
topT <- as.data.frame(res)
#Adjusted P values (FDR Q values)
with(topT, plot(res$log2FoldChange, -log10(res$padj), pch=20, main="Volcano plot", cex=1.0, xlab=bquote(~Log[2]~fold~change), ylab=bquote(~-log[10]~Q~value)))
with(subset(topT, padj<0.05 & abs(log2FoldChange)>2), points(log2FoldChange, -log10(padj), pch=20, col="red", cex=0.5))
#Add lines for absolute FC>2 and P-value cut-off at FDR Q<0.05
abline(v=0, col="black", lty=3, lwd=1.0)
abline(v=-2, col="black", lty=4, lwd=2.0)
abline(v=2, col="black", lty=4, lwd=2.0)
abline(h=-log10(max(topT$pvalue[topT$padj<0.05], na.rm=TRUE)), col="black", lty=4, lwd=2.0)

