% NAND 1.0 Growth curves


close all; clear; clc

load('NAND1_GCs.mat')

% columns 12,24,36,48,60,72,84,96 correspond to only media, no need to plot
% those

figure;
plot(A(:,[11 23	35	47	59	71	83	95]),'linewidth',2,'color',[0, 0.4470, 0.7410]) % Strain #1 (WT) 4 first Ara, last 4 Ara+IPTG
hold on
plot(A(:,[10 22	34	46	58	70	82	94]),'linewidth',2,'color',[0.8500, 0.3250, 0.0980]) % Strain #1 (WT) 4 first no induction, last 4 with IPTG
% plot(A(:,[1	13	25	37	49	61	73	85]),'linewidth',2,'color',[0.9290, 0.6940, 0.1250]) % Strain #2 (PhlF Genome) 4 first (A-D) no induction, last 4 (E-H) with IPTG
% plot(A(:,[2	14	26	38	50	62	74	86]),'linewidth',2,'color',[0.4940, 0.1840, 0.5560]) % Strain #3 (IcaR Genome) 4 first no induction, last 4 with IPTG
% plot(A(:,[3	15	27	39	51	63	75	87]),'linewidth',2,'color',[0.4660, 0.6740, 0.1880]) % Strain #4 (NAND Genome) 4 first no induction, last 4 with IPTG
% plot(A(:,[4	16	28	40	52	64	76	88]),'linewidth',2,'color',[0.3010, 0.7450, 0.9330]) % Strain #4 (NAND Genome) 4 first Ara, last 4 Ara+IPTG
% plot(A(:,[5	17	29	41	53	65	77	89]),'linewidth',2,'color',[0.6350, 0.0780, 0.1840]) % Strain #5, 4 first no induction, last 4 with IPTG
% plot(A(:,[6	18	30	42	54	66	78	90]),'linewidth',2,'color',[0, 0, 1]) % Strain #6, 4 first no induction, last 4 with IPTG (this strain caused loss of fitness) 
% plot(A(:,[8	20	32	44	56	68	80	92]),'linewidth',2,'color',[0, 0.5, 0]) % Strain #8 (PhlF plasmid) 4 first no induction, last 4 with IPTG
% plot(A(:,[9	21	33	45]),'linewidth',2,'color',[1, 0, 0]) % Strain #9 (IcaR plasmid) 4 first no induction
% plot(A(:,[57 69	81	93]),'linewidth',2,'color','k') % Strain #9 (IcaR plasmid) last 4 with IPTG (this strain caused loss of fitness) 


% 24 hour run




