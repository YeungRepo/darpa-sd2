clear;
close all;
clc;


global k_control;
% executing main
datane_modified;
basmva = 100;

% dimensions
n = size(bus,1);    
m = size(line,1);  

% indices
G = mac_con(:,2);
g = length(G);
L = (1:n)'; L(G)=[]; 
l = length(L);

% ********************load power flow solutions as reference point********
pf_data = load('IEEE_39_pf.mat');
bus_sol = pf_data.bus_sol;
line_flow = pf_data.line_flow; 

% reference voltage magnitudes
Vm_orig = bus_sol(:,2);   % 39*1 vector
% reference voltage angles, in rad
theta_orig = bus_sol(:,3)/180*pi;

% net reference injections
Pm_orig= bus_sol(:,4)-bus_sol(:,6);        %Pgen - Pload

% Line power flows
% P1: inject from node 1; P2: inject from node 2
% Assume lossless: (P1 -P2)/2 is the real power flow on line
% Plink_orig = (line_flow(1:m,4) - line_flow(m+1:(2*m),4))/2;
%******************************************************************************

% generator inertias 
M=zeros(n,n);
M(G,G) = diag(2*mac_con(:,16)/(2*pi*60).*mac_con(:,3)/basmva);    %M:=2H/(2*pi*f_0)*mac_basmva/sys_basmva (Kundur pp. 131. Note: dot \omega in rad/(s^2))

% composite (load-)damping constant D
% (Kundur pp.598, Fig 11.18 : 1.0 pu for per unit freq; Not provided explicitly in the original datane file)
D=zeros(n,n);
D(G,G) = 5.0*diag(bus_sol(G,4))/(2*pi*60)*eye(g);   % D omega for omega in rad/s
% D(L,L) = 0.00/(2*pi*60)*eye(l); % zero D for load buses
D(L,L) = 0.2*mean(diag(D(G,G)))*eye(l); % construct nonzero D for load buses (required for nonlinear flow)
D=D*2;


% Generator governor, turbine parameters 
% (Kundur pp.598, Fig. 11.18)
Tg = 0.2*ones(g,1);
Tb = 7.0*ones(g,1);  % orignal 7
R = 1*0.05*(2*pi*60)./bus_sol(G,4);  % converting to pu power (system base) / (rad/s); assuming approximately the set point of each generator is the total generation contributing to regulation (example, Kundur 597)

% controllable load paramters---------
L_controlled=[25 26];
Td=5.0*ones(n,1);

% Line parameters (B_ij = |V_i||V_j| / X_ij cos(\theta_ij^0) for linearized PF); network topology
Bij = zeros(m,1);
Bij_nonlinear = zeros(m,1);
Bij_vf=zeros(m,1);
Plink_orig = zeros(m,1);     %reference branch flow, used to calculate deviation of power flow under nonlinear model
C= zeros(n,m);   %incidence matrix
for k=1:m
    ixp = line(k,1);
    ixn = line(k,2);
    C([ixp ixn],k)=[1;-1]; 
    CQ([ixp ixn],k)=[1;1]; 
    Bij(k)=Vm_orig(ixp)*Vm_orig(ixn)/(line(k,4)) *cos(theta_orig(ixp)-theta_orig(ixn));
    Bij_nonlinear(k)=Vm_orig(ixp)*Vm_orig(ixn)/(line(k,4));   % used for nonlinear flow
    Bij_vf(k)=1/line(k,4);   % used for voltage dynamics and nonlinear flow
    Plink_orig(k) = Bij_nonlinear(k)*sin(theta_orig(ixp)-theta_orig(ixn));
end

% Bjj
Bjj=-bus(:,9);
for cnt1=1:n
   for cnt2=1:m
       if C(cnt1,cnt2) ~= 0
      Bjj(cnt1)= Bjj(cnt1)-Bij_vf(cnt2);
       end
   end
end

% synchronous generator data
Tv = mac_con(:,9);
xd = mac_con(:,6);
xdt= mac_con(:,7);

Ef=1;

% edgev_orig=ones(m,1);
% for cnt1=1:m
%    for cnt2=1:n
%        if C(cnt2,cnt1) ~=0
%        edgev_orig(cnt1)=edgev_orig(cnt1)*Vm_orig(cnt2);
%        end
%    end
% end
% Eff=(1-(xd-xdt).*Bjj(G)).*Vm_orig(G) - (xd-xdt).*( (CQ(G,:)*(edgev_orig.*(diag(Bij_vf)*cos(transpose(C)*theta_orig))))./Vm_orig(G) ) ;
% Eff



% control parameters for AGC and UC
% G_controlled = 1:1:(g-1);         % generators that participate in secondary frequency control/ AGC/ UC. Case 1: All are controlled except the big one
G_controlled = [ 1 3 5  ];         % Case 2: Five out of 10 generators are controlled, 27.98/61.9822 = 45% of total generation
cost_alpha = bus_sol(G,4); % coefficient for UC cost functions proportional to nominal generation (approximate capacity). 
                           %c(p) := 1/(2*cost_alpha) p^2   ==> p = cost_alpha*(-omega)
% controllable buses
bus_controlled = mac_con(G_controlled,2);

% Lines with  both ends controllable

% line_both_ends_controlled = zeros(m,1);
% for mm = 1:m
%     if ismember(line(mm,1), bus_controlled) && ismember(line(mm,2), bus_controlled)
%         line_both_ends_controlled(mm) =1;
%     end;
% end;
% line_one_end_uncontrolled = 1:1:m;
% line_one_end_uncontrolled(line_both_ends_controlled>0.5) = [];
% ltilde = length(line_one_end_uncontrolled);

% define control areas, if ever used in any case below
% Edges in cut 1(1->2) 3(2->3) 41(26->27)  A1=(2,26)  A2=(1,3,27)
% Interface direction A1->A2

% narea = 2;
% hatC=zeros(narea,m); % 2 areas
% hatC(1,1)=-1;hatC(1,3)=1;hatC(1,41)=1;
% hatC(2,1)=1;hatC(2,3)=-1;hatC(2,41)=-1;
% cutLinks = [1 3 41];          
% 
% % buses and generators index for areas
% ixa1=[2 25 26 28 29 30 37 38];
% ixa2=1:n;
% ixa2(ixa1)=[];
% ixga1 = [1 8 9]; % generator index for generators in area 1
% ixga2 = 1:1:g;
% ixga2(ixga1) = [];
% ixa1_g = [30 37 38]; % generator bus index for generators in area 1
% ixa2_g = [31 32 33 34 35 36 39];
% nga1 = length(ixga1);
% nga2 = length(ixga2);
% ixg_controlled = zeros(narea,g);
% ixg_controlled(1,intersect(ixga1,G_controlled)) = 1;
% ixg_controlled(2,intersect(ixga2,G_controlled)) = 1;

%limitlink=[4, 19, 26, 33];
limitlink=[4, 19, 26, 33];
% specify line limits, if used in any case
barP = 100*ones(m,1);                % Thermal limits
barP(limitlink)= 3; %this makes tie-lines thermal limits active
underP = -barP;
% thermal limits for deviations in line flows
barDeltaP = barP - Plink_orig;
underDeltaP = underP - Plink_orig;



%----------------
% remove the first bus as reference bus
% n=n-1;
% G=G-1;
% l=l-1;
% L(1)=[];
% L=L-1;
% theta_orig=theta_orig-theta_orig(1);
% theta_orig(1)=[];
% M(1,:)=[];
% M(:,1)=[];
% D(1,:)=[];
% D(:,1)=[];
% C(1,:)=[];
% bus_controlled=bus_controlled-1;
rm=(2:n)';
Ct=C(rm,:);
Lt=L(2:l)-1;
% ------

% shift matrix transpose
H=(diag(Bij)*Ct'/(Ct*diag(Bij)*Ct'))';
Hg=H(G-1,:);
Hl=H(Lt,:);


                           
% a step change disturbance
dtb=[30];
deltaPm=zeros(n,1);
deltaPm(dtb)= 0;  

% transient distrubance on initial conditions
distb_g=3*rand(g,1)-1.5;
distb_d=3*rand(n,1)-1.5;
distb_lambda=2*rand(1,1)-1;
distb_rhom=2*rand(m,1);
distb_rhop=2*rand(m,1);


%cases  = 4;     % case 1: droop control only

                % case 2: AGC (secondary frequency control only);  
                % case 3: UC w/o Congestion Management, no inter-area
                % case 4: UC w Congestion management, no inter-area
                
                % case 5: AGC + ACE (secondary freq + inter-area constraint)
                % case 6: UC w congestion management + inter-area constraint
                % case 7: UC no congestion management, w inter-area constraint
                % case 8: (Champery extension. Still working on it. )UC on a subset. No ACE, No congestion management
                    
%store_xo = false;
%for h = cases
    
%     switch h
%         case 1
%             disp('droop only');
%         case 2
%             disp('AGC, no inter-area constraint');
%         case 3
%             disp('UC, no Congestion Management, no inter-area constraint');
%         case 4
%             disp('UC, with Congestion Management, no inter-area constraint');
%         case 5
%             disp('AGC + ACE');
%         case 6
%             disp('UC, with Congestion Management and inter-area constraint');
%         case 7
%             disp('UC, no Congestion Management, with inter-area constraint');
%         case 8
%             disp('UC on a subset of buses, no Congestion Management, no inter-area constraint');
%     end;
    
    %generic setup
    Tmax=150;
        
    %initial condition, all deviations under linear model. Will be used in
    %all cases below
    
    omegao = zeros(g,1);
    thetao = theta_orig;
    valveo = zeros(g,1);   % valve position
    pmecho = zeros(g,1) + distb_g ;   % mechanic power
    loado  = zeros(n,1) + distb_d ;  %-----
    
%    switch h 
        
%         case 1 %droop only
%                     
% %             A = zeros(4*g,4*g);
% %             A(1:g, 1:g)= -M(G,G)\D(G,G); 
% %             A(1:g, g+1:2*g) = M(G,G)\...
% %                 (-C(G,:)*diag(Bij)*transpose(C(G,:)) + (C(G,:)*diag(Bij)*transpose(C(L,:))/(C(L,:)*diag(Bij)*transpose(C(L,:))))*C(L,:)*diag(Bij)*transpose(C(G,:)));
% %             A(1:g,2*g+1:3*g) = 0;
% %             A(1:g,3*g+1:4*g) = inv(M(G,G));
% %             
% %             A(g+1:2*g, 1:g)= eye(g);
% %             A(g+1:2*g, g+1:2*g)=0;
% %             A(g+1:2*g, 2*g+1:3*g)=0;
% %             A(g+1:2*g, 3*g+1:4*g)=0;
% %             
% %             A(2*g+1:3*g, 1:g)= -inv(diag(Tg)).*inv(diag(R));
% %             A(2*g+1:3*g, g+1:2*g)=0;
% %             A(2*g+1:3*g, 2*g+1:3*g)=-inv(diag(Tg));
% %             A(2*g+1:3*g, 3*g+1:4*g)=0;
% %             
% %             A(3*g+1:4*g, 1:g)= 0;
% %             A(3*g+1:4*g, g+1:2*g)=0;
% %             A(3*g+1:4*g, 2*g+1:3*g)=inv(diag(Tb));
% %             A(3*g+1:4*g, 3*g+1:4*g)=-inv(diag(Tb));
% %             
% %             [V_eig,D_eig]=eig(A);
% %             figure(5); hold on; grid on;
% %             for ii = 1:4*g
% %                 plot(real(D_eig(ii,ii)),imag(D_eig(ii,ii)),'ko');
% %             end
% %             
% %             positive_real_parts_there_are = sum(sum(real(D_eig)>1e-5))
%              
%             xo = [omegao;thetao;valveo;pmecho];
%             txf = @(t, x) droop(t,x,n,deltaPm,M,D,C, Bij_nonlinear, Plink_orig,  R, Tg, Tb, G,L);
%             [T,X] = ode23s(txf,[0 Tmax],xo);    %run ode
%         
%         case 2 %AGC (no inter-area constraints)
%              
%             % AGC parameters - whole system as one area
%             % Kp = 0.5; % proportional area control gain
%             Ki = 0.03;   % integral area control gain
%             beta = sum(D*ones(n,1)) + sum(1./R);  % frequency weight in ACE
%             PF = cost_alpha(G_controlled)/sum(cost_alpha(G_controlled)); % normalize such that sum(PF) = 1;
%             measurement_point = 5;             % where (#generator) we measure the frequency; (can be a vector - take average of several locations)
%             
%             p_areao = 0;
%             xo = [omegao;thetao;valveo;pmecho; p_areao];
%             txf = @(t, x) agc(t,x,n,deltaPm,M,D,C, Bij_nonlinear, Plink_orig, R, Tg, Tb, G,G_controlled, L,  Ki, beta, PF, measurement_point);
%             [T,X] = ode23s(txf,[0 Tmax],xo);    %run ode
%         
%             
%         case 5 %AGC + ACE
%             
%             % AGC parameters - whole system as one area
% %             Ki = [sum(cost_alpha(ixg_controlled(1,:)==1));
% %                       sum(cost_alpha(ixg_controlled(2,:)==1))];
% %             Ki = 0.05*Ki/sum(Ki);                              % distribute integral gain according to the participation capacity
%             Ki = [0.03; 0.03];
%             beta = [sum(D(ixa1_g)*ones(nga1,1)) + sum(1./R(ixga1));
%                     sum(D(ixa2_g)*ones(nga2,1)) + sum(1./R(ixga2))];
%             PF = zeros(narea,g);
%             PF(1,ixg_controlled(1,:)==1) = cost_alpha(ixg_controlled(1,:)==1)/sum(cost_alpha(ixg_controlled(1,:)==1));
%             PF(2,ixg_controlled(2,:)==1) = cost_alpha(ixg_controlled(2,:)==1)/sum(cost_alpha(ixg_controlled(2,:)==1));
%             measurement_point = [9;5];
%            
%             p_areao = zeros(narea,1);
%             xo = [omegao;thetao;valveo;pmecho; p_areao];
%             
%             txf = @(t, x) agc_ace(t,x,n,deltaPm,M,D,C, Bij_nonlinear, Plink_orig, R, Tg, Tb, G,  L,  Ki, beta, PF, measurement_point, ixg_controlled, hatC);
%             [T,X] = ode23s(txf,[0 Tmax],xo);    %run ode
%             
%         case 3 %UC, no Congestion Management
%             inter_area =0;
%             congestion_management=0;
%             
%         case 4 %UC, with Congestion Management, no inter-area constraint
%             inter_area =0;
%             congestion_management=1;
%            
%         case 6 %UC, with Congestion Management and inter-area constraint
%             inter_area = 1;
%             congestion_management= 1;
%             
%         case 7 %UC, with inter-area constraint, no congestion management
%             inter_area = 1;
%             congestion_management= 0;
%             
% %         case 8 % New algorithm added 1/30/2017: UC implementable on a subset of buses, no ACE. no congestion management   
% %             inter_area = 0;
% %             congestion_management= 0;
%     end
    %end all cases
    
%    if h==3 || h==4 || h==6 ||h==7  % unified control with different selections of objectives
        
        PF = cost_alpha(G_controlled)/sum(cost_alpha(G_controlled)); % normalize such that sum(PF) = 1;
        % ------
        PFd=sort(PF);
        PFd(3:end)=[];
        PFd=-(PFd+(rand(1)-0.5)*0.1);
        
        lambdao = zeros(1,1);
%         phio=zeros(n,1);
%         pio = zeros(narea,1);
        rhopo=zeros(m,1);
        rhomo=zeros(m,1);
        Eo=Vm_orig;
        xo = [omegao;thetao;valveo;pmecho;loado;lambdao;rhopo;rhomo;Eo];
%       xo = [omegao;thetao;valveo;pmecho;lambdao;phio;pio;rhopo;rhomo];
        
%         mass1=ones(length([omegao;thetao;valveo;pmecho;loado;lambdao;rhopo;rhomo]),1);
%         mass2=zeros(length(Eo),1);
%         mass2(G)=1;
%         mass=diag([mass1;mass2]);
        opts = odeset('Stats','on');
        %opts.MaxStep = [ 6.2972e-02]; 
        %opts.MinStep = [1e1];
%        global thistory 
%        thistory = [];
         txf = @(t, x) md(t,x,n,m,deltaPm,barDeltaP,underDeltaP, M, D, C, Bij, Bij_nonlinear,Plink_orig, R, Tg, Tb,Td, G, G_controlled,  L,L_controlled,PF,PFd, H, Hg, Hl, rm, Bjj, Tv, xd, xdt, Ef,Bij_vf,CQ); 
 %       txf = @(t, x) md(t,x,n,m,deltaPm,barDeltaP,underDeltaP, M, D, C, Bij, Bij_nonlinear,Plink_orig, R, Tg, Tb,Td, G, G_controlled,  L,L_controlled,PF,PFd, H, Hg, Hl, rm, Bjj, Tv, xd, xdt, Ef,Bij_vf,CQ,opts); 
        
        
        [T,X] = ode23t(txf,[0:0.2:Tmax],xo,opts);    
            
%    end;
    
%     if h==8  % unified control on a subset of buses
%         
%         PF = cost_alpha(G_controlled)/sum(cost_alpha(G_controlled)); % normalize such that sum(PF) = 1;
%         
%         lambdao = zeros(n,1);
%         phio=zeros(n,1);
% %         pio = zeros(narea,1);
% %         rhopo=zeros(m,1);
% %         rhomo=zeros(m,1);
%         fo = zeros(ltilde,1);
%         xo = [omegao;thetao;valveo;pmecho;lambdao;phio;fo];
% 
% 
% 
%         % internal parameters of UC: set here
% 
%         k_control= sum(1./R(G_controlled));
% 
%         k_lambda = zeros(n,n);
%         k_lambda(G,G) = inv(M(G,G));
%         k_lambda(L,L) = inv(0.2*eye(l));   % mimic 0.2s virtual inertia for load buses
%         k_lambda = 1*0.8*k_lambda;
% 
%         k_phi = 1*2000*diag(1./(abs(C*diag(Bij)*transpose(C))*ones(n,1))); 
%         % k_pi = 0.08*eye(narea);
%         % k_rho = 0.5*eye(m);
%         k_f = 1*eye(ltilde);
% 
%         Btilde = diag(Bij);
%         line_both_ends_controlled = 1:1:m;
%         line_both_ends_controlled(line_one_end_uncontrolled)=[];
%         Btilde(line_both_ends_controlled,:)=[];
%         
%         txf = @(t, x) unified_control_f(t,x,n,m,ltilde, deltaPm, M, D, C, Bij_nonlinear, Btilde, R, Tg, Tb, G, G_controlled, line_one_end_uncontrolled,  L,g,l,PF, k_control, k_lambda, k_phi, k_f); 
%         [T,X] = ode113(txf,[0 Tmax],xo);    %run ode
%             
%     end;
        
%end





Plot;

% X(:,1:g)=[];
% X=[Delta_omegafull,X];





% clear intermediate variables
% clear pf_data;