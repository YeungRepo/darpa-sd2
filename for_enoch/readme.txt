by: Rob Moseley

for_enoch/
└─── readme
└─── dchip/
│    └─── dchip_20180629_171325_readme.tsv
│    └─── rho0_r1_dchip_20180629_171325.tsv
│    └─── wt_r1_dchip_20180629_171325.tsv
│    └─── wt_r2_dchip_20180629_171325.tsv
│    └─── wt37d_r1_dchip_20180629_171325.tsv
└─── rma/
│    └─── rma_20180629_171227_readme.tsv
│    └─── rho0_r1_rma_20180629_171227.tsv
│    └─── wt_r1_rma_20180629_171227.tsv
│    └─── wt_r2_rma_20180629_171227.tsv
│    └─── wt37d_r1_rma_20180629_171227.tsv
└─── gene_lists/
     └─── Anastasia/
     │    └─── Anastasia_figures.png
     │    └─── de_dchip_20180626_annot_pv1e-40_fc4.0_mda2000.txt
     └─── Jed/
           └─── Jed_figures.png
           └─── de_dchip_20180626_reporters_jed.tsv

Descriptions:
	dchip/ : dchip normalized microarray data and a readme file. Files were normalized together.
	rma/ : rma normalized microarray data and a readme file. Files were normalized together.
	gene_lists/ : folders with the gene list and figure from Anastasia and Jed


Analyses:
- both used dchip data

	Anastasia: used all three conditions (i.e., rho0, wt, and wt37d) to make truth tables
		- Used both wt_rep1 and wt_rep2
		- I'm unsure if she dropped the first two time points for wt_rep1, wt_rep2, and wt37d before analysis
		- Anastasia made figures for each of her genes to visualize the variation between time points and conditions
		
	Jed: I'm very in the dark with what he did to find his genes but he seemed to only focus on wt_rep2 and wt37d
		- It seems he did not use wt_rep1 in his analysis
		- He did remove the first two timepoints from wt_rep2 and wt37d
		- I created similar plots as Anastasia's for Jed's gene list