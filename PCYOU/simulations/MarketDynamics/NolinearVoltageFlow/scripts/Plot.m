h=4;
% switch h
%     case 1
%         Delta_omega = X(:,1:g);
%         Delta_theta = X(:,g+1:g+n);
%         Delta_valve = X(:,g+n+1:2*g+n);
%         Delta_pmech = X(:,2*g+n+1:3*g+n);
%         
%     case 2
%         Delta_omega = X(:,1:g);
%         Delta_theta = X(:,g+1:g+n);
%         Delta_valve = X(:,g+n+1:2*g+n);
%         Delta_pmech = X(:,2*g+n+1:3*g+n);
%         Delta_psignal = X(:,end)*transpose(PF);
%         
%     case 5
%         Delta_omega = X(:,1:g);
%         Delta_theta = X(:,g+1:g+n);
%         Delta_valve = X(:,g+n+1:2*g+n);
%         Delta_pmech = X(:,2*g+n+1:3*g+n);
%         Delta_psignal = X(:,3*g+n+1:3*g+n+narea)*PF;   
%     
% end;


if h==3||h==4||h==6 ||h==7  % all cases of unified control
  % ------------     
    Delta_omega = X(:,1:g);
    Delta_theta = X(:,g+1:g+n);
    Delta_valve = X(:,g+n+1:2*g+n);
    Delta_pmech = X(:,2*g+n+1:3*g+n);
    Delta_load  = X(:,3*g+n+1:3*g+2*n);
    
    Delta_lambda = X(:,3*g+2*n+1:3*g+2*n+1);
    Delta_rhop = X(:,3*g + 2*n+2:3*g + 2*n+1+m);
    Delta_rhom = X(:,3*g + 2*n+2+m:3*g + 2*n+1+2*m);
    Delta_E    = X(:,3*g+2*n+2+2*m:3*g+3*n+1+2*m);
 % ------------    
%     Delta_psignal = - k_control*repmat(transpose(PF),length(T),1).*(Delta_lambda(:,G(G_controlled)) + Delta_omega(:,G_controlled)) ...
%     + 1./repmat(transpose(R(G_controlled)),length(T),1).*Delta_omega(:,G_controlled);   % cancel the oroginal droop control for UC generator

% actual flow, not deviation
Delta_edgev=ones(length(T),m);
for cnt1=1:m
    for cnt2=1:n
        if C(cnt2,cnt1) ~=0
            Delta_edgev(:,cnt1)=Delta_edgev(:,cnt1).*Delta_E(:,cnt2);
        end
    end
end
Delta_flow=Delta_edgev.*repmat(transpose(Bij_vf),length(T),1).*sin(Delta_theta*C); 
Delta_flowdeviation=Delta_flow - repmat(transpose(Plink_orig),length(T),1);

Delta_flownonlinear=repmat(transpose(Bij_nonlinear),length(T),1).*sin(Delta_theta*C);
Delta_flowlinear=repmat(transpose(Bij),length(T),1).*(Delta_theta*C);

      
Delta_omegafull=zeros(length(T),n);
Delta_omegafull(:,G)=Delta_omega;
Delta_omegafull(:,L)=(-(Delta_flow - repmat(transpose(Plink_orig),length(T),1))*transpose(C(L,:)) + repmat(transpose(deltaPm(L)),length(T),1)  - Delta_load(:,L)  )/D(L,L);

Delta_psignal = ( repmat(Delta_lambda,1,length(G_controlled)) - Delta_omega(:,G_controlled) + Delta_rhom * transpose(Hg(G_controlled,:)) - Delta_rhop * transpose(Hg(G_controlled,:)) )  -  1./(k_control*repmat(transpose(PF),length(T),1)).* Delta_pmech(:,G_controlled)  ...
    +1./repmat(transpose(R(G_controlled)),length(T),1).*Delta_omega(:,G_controlled)   +  Delta_pmech(:,G_controlled);   % cancel the oroginal droop control 

Delta_dsignal =1./(k_control*repmat(transpose(PFd),length(T),1)).* Delta_load(:,L_controlled)  - ( repmat(Delta_lambda,1,length(L_controlled)) - Delta_omegafull(:,L_controlled) + Delta_rhom * transpose(H(L_controlled-1,:)) - Delta_rhop * transpose(H(L_controlled-1,:)) )   ...
     +  Delta_load(:,L_controlled);   

Delta_lmp= repmat(Delta_lambda,1,n-1) - Delta_omegafull(:,rm) + Delta_rhom * transpose(H) - Delta_rhop * transpose(H) ;
end;


% if h==8  % unified control on a subset of buses
%     
%     Delta_omega = X(:,1:g);
%     Delta_theta = X(:,g+1:2*g);
%     Delta_valve = X(:,2*g+1:3*g);
%     Delta_pmech = X(:,3*g+1:4*g);
% 
%     Delta_lambda = X(:,4*g+1:4*g+n);
%     Delta_phi = X(:,4*g+n+1:4*g+2*n);
%     % Delta_f = X(:, 4*g+2*n+1:4*g+2*n+ltilde);
%     
%     Delta_psignal = - k_control*repmat(transpose(PF),length(T),1).*(Delta_lambda(:,G(G_controlled)) + Delta_omega(:,G_controlled)) ...
%     + 1./repmat(transpose(R(G_controlled)),length(T),1).*Delta_omega(:,G_controlled);   % cancel the oroginal droop control for UC generator
%        
% end;

%v = X(:,3*n+1:4*n);
%Pi = X(:,4*n+1:4*n+a);
%Rhop = X(:,4*n+a+1:4*n+l+a);
%Rhom = X(:,4*n+l+a+1:4*n+2*l+a);
% virtualFlow =  v*C*diag(Bij); 


% frequencies
figure(1);hold on;
plot(T, Delta_omegafull(:,:)/(2*pi) + 60,'lineWidth',1.5);
plot([T(1) T(end)], [60.0 60.0],'--k','lineWidth',1.0);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Frequency (Hz)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)

% figure(11);hold on;
% plot(T, Delta_omega,'lineWidth',1.5);
% %plot([T(1) T(end)], [60.0 60.0],'--k','lineWidth',1.0);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Frequency (Hz)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)

% phase angles
% figure(2);hold on;
% plot(T, Delta_theta-repmat(Delta_theta(:,end),1,n),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Relative Angle (rad)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)

figure(2);hold on;
plot(T, Delta_theta(:,:),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Phase Angle (rad)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)



% valve positions
figure(3);hold on;
plot(T, Delta_valve(:,G_controlled),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Valve Position (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)


% mechanic power
figure(4);hold on;
plot(T, Delta_pmech(:,G_controlled),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Generation (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)



if h==2 ||h==3 || h==4 
    % input signal
    figure(5);hold on;
    plot(T, Delta_psignal,'lineWidth',1.5);
    grid off;
    %axis([0 150 59.5 60.1]);
    xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
    ylabel('Generation Control Signal (pu)','fontname', 'Arial','fontsize',20);
    set(gca, 'fontname','Arial','fontsize',20)
    

end;

% line flow
figure(6);hold on;
plot(T, Delta_flow(:,limitlink),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Actual Line Flow (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)
% line flow
figure(16);hold on;
plot(T, Delta_flow(:,:),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Actual Line Flow (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)
%---------------------------------------------------
% figure(16);hold on;
% plot(T, Delta_flownonlinear(:,1:m),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Nonlinear Line Flow (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)
%---------------------------------------------------
% figure(26);hold on;
% plot(T, Delta_flowlinear(:,1:m),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Linear Line Flow (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)
%---------------------------------------------------
% figure(36);hold on;
% plot(T, Delta_flowdeviation(:,1:m),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Line Flow (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)



% load
figure(7);hold on;
plot(T, Delta_load(:,L_controlled),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Load (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)

% d_signal
figure(8);hold on;
plot(T, Delta_dsignal,'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Load Control Signal (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)

figure(9);hold on;
plot(T, Delta_E(:,G),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('Voltage Magnitude (pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)

figure(10);hold on;
plot(T, Delta_lmp(:,[L_controlled,bus_controlled']-1),'lineWidth',1.5);
grid off;
%axis([0 150 59.5 60.1]);
xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
ylabel('DLMP ($/pu)','fontname', 'Arial','fontsize',20);
set(gca, 'fontname','Arial','fontsize',20)



% ============
% figure(10);hold on;
% plot(T, Delta_lambda(:,1),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('lambda (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)
% 
% figure(11);hold on;
% plot(T, Delta_rhom(:,19),'lineWidth',1.5);
% grid off;
% %axis([0 150 59.5 60.1]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('rhom (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)


% if h==5 
%     % input signal
%     figure(5);hold on;
%     
%     for iig = 1:g
%         if ixg_controlled(1,iig)==1
%             plot(T, Delta_psignal(:,iig)/PF(1,iig),'r-','lineWidth',1.5);
%         end;
%         if ixg_controlled(2,iig)==1
%             plot(T, Delta_psignal(:,iig)/PF(2,iig),'b--','lineWidth',1.5);
%         end;
%     end;
%     
%         
%     grid off;
%     %axis([0 150 59.5 60.1]);
%     xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
%     ylabel('Control Signal (pu)','fontname', 'Arial','fontsize',20);
%     set(gca, 'fontname','Arial','fontsize',20)
% 
% 
% end;
% 
% 
% if  h==6 ||h==7 ||h==8
%     
%     % rescale PF
%     PF_rescaled = zeros(narea,g);
%     PF_rescaled(1,ixg_controlled(1,:)==1) = cost_alpha(ixg_controlled(1,:)==1)/sum(cost_alpha(ixg_controlled(1,:)==1));
%     PF_rescaled(2,ixg_controlled(2,:)==1) = cost_alpha(ixg_controlled(2,:)==1)/sum(cost_alpha(ixg_controlled(2,:)==1));
%     % input signal
%     figure(5);hold on;
%     
%     for iig = 1:g
%         tempindex = find(G_controlled==iig);
%         if ixg_controlled(1,iig)==1
%             plot(T, Delta_psignal(:,tempindex)/PF_rescaled(1,iig),'r-','lineWidth',1.5);
%         end;
%         if ixg_controlled(2,iig)==1
%             plot(T, Delta_psignal(:,tempindex)/PF_rescaled(2,iig),'b--','lineWidth',1.5);
%         end;
%     end;
%     
%         
%     grid off;
%     %axis([0 150 59.5 60.1]);
%     xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
%     ylabel('Control Signal (pu)','fontname', 'Arial','fontsize',20);
%     set(gca, 'fontname','Arial','fontsize',20)
% 
% 
% end;

  
% Read selected line power
% theta_l = (C(L,:)*diag(Bij)*transpose(C(L,:)))\( 0 + repmat(deltaPm(L),1,length(T)) - C(L,:)*diag(Bij)*transpose(C(G,:))*transpose(Delta_theta));   % Delta_theta is indeed Delta_theta_g
% theta = zeros(n,length(T));
% theta(G,:) = transpose(Delta_theta);
% theta(L,:) = theta_l;
% clear theta_l;


% theta = transpose(Delta_theta);
% 
% Delta_tieline_flow = diag(Bij(cutLinks))*sin(transpose(C(:,cutLinks))*theta) - repmat(Plink_orig(cutLinks),1,length(T)) ;
% Tieline_flow = repmat(Plink_orig(cutLinks),1,length(T)) + Delta_tieline_flow;
% Tieline_flow(1,:) = -Tieline_flow(1,:); % flip direction for convenience of plotting 
% 
% % plot tie-line flows. Total flow 7.4877 from area 1 to 2.
% figure(6); hold on;
% plot(T, transpose(Tieline_flow),'lineWidth',1.5);
% plot([T(1) T(end)], [barP(cutLinks(1)) barP(cutLinks(1))],'--k','lineWidth',1.0);
% grid off;
% %axis([0 150 0.0 6.0]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Line Flow (pu) full quantities','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)
% 
% 
% figure(7); hold on;
% plot(T, hatC(1,:)*(diag(Bij_nonlinear)*sin(transpose(C)*theta)-repmat(Plink_orig ,1,length(T))) ,'lineWidth',1.5);
% grid off;
% %axis([0 150 0.0 6.0]);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% ylabel('Deviation in inter-Area Power (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot in parallel
% figure(1);hold on;
% 
% load AGC
% 
% Omega = X(:,1:n);
% Theta = X(:,n+1:2*n);
% 
% 
% subplot(1,2,1);  hold on;
% plot(T, Omega(:,[ 3  13 23 26 33 ])/(2*pi) + 60,'lineWidth',1.0);
% plot([T(1) T(end)], [60.0 60.0],'--k','lineWidth',1.0);
% 
% grid off;axis([0 150 59.4 60.2]);
% ylabel('Frequency (Hz)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% title('(a) AGC','fontname', 'Arial','fontsize',20);
% 
% load SFC-CM
% 
% Omega = X(:,1:n);
% %Lambda = X(:,n+1:2*n);
% Theta = X(:,2*n+1:3*n);
% 
% 
% subplot(1,2,2); hold on;
% plot(T, Omega(:,[ 3 13 23 26 33 ])/(2*pi) + 60,'lineWidth',1.0);
% plot([T(1) T(end)], [60.0 60.0],'--k','lineWidth',1.0);
% grid off;axis([0 150 59.4 60.2]);
% set(gca, 'fontname','Arial','fontsize',20);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% title('(b) Unified Control','fontname', 'Arial','fontsize',20);
% 
% 
% % tie-line flows
% figure(2); hold on; 
% 
% load AGC
% 
% Omega = X(:,1:n);
% Theta = X(:,n+1:2*n);
% 
% subplot(1,2,1); hold on;
% plot(T, -(Plink_orig(1)*ones(length(T),1) +  Theta*C(:,1) *Bij(1)),'lineWidth',1.5, 'Color','g' );
% plot(T, (Plink_orig(3)*ones(length(T),1) +  Theta*C(:,3) *Bij(3)),'lineWidth',1.5, 'Color','b' );
% plot(T, (Plink_orig(41)*ones(length(T),1) +  Theta*C(:,41) *Bij(41)),'lineWidth',1.5, 'Color','r' );
% plot([T(1) T(end)], [2.6 2.6],'--k','lineWidth',1.0);
% 
% grid off;axis([0 150 -10.0 6.0]);
% ylabel('Line Flow (pu)','fontname', 'Arial','fontsize',20);
% set(gca, 'fontname','Arial','fontsize',20);
% xlabel('Time (sec)','fontname', 'Arial','fontsize',20);
% title('(a) AGC','fontname', 'Arial','fontsize',20);
%  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

