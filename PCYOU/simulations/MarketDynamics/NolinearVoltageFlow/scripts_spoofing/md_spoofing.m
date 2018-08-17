

function y = md_spoofing(t,x,n,m,deltaPm,barDeltaP,underDeltaP, M, D, C,Bij,  Bij_nonlinear,Plink_orig, R, Tg, Tb,Td, G, G_controlled,  L,L_controlled, PF,PFd, H, Hg, Hl, rm, Bjj, Tv, xd, xdt, Ef, Bij_vf,CQ,price0,bus_controlled,price_sup,gain)
%global thistory
%narea = size(hatC,1);
g = length(G);
%l = length(L);

% internal parameters of UC: set here
global k_control
k_control= 0.7*sum(1./R(G_controlled));

k_lambda =1/mean(diag(M(G,G)))*0.4;
% k_lambda = zeros(n,n);
% k_lambda(G,G) = inv(M(G,G));
% k_lambda(L,L) = inv(0.2*eye(l));   % mimic 0.2s virtual inertia for load buses
% k_lambda = 1*0.8*k_lambda;

% k_phi = 1*2000*diag(1./(abs(C*diag(Bij_nonlinear)*transpose(C))*ones(n,1))); 
% k_phi = 0.1*eye(n);
% 
% k_pi = 0.08*eye(narea);
k_rho = 25*0.5*eye(m);
k_rho = k_rho*1;

% read from current state
omega_g = x(1:g);
theta = x(g+1:g+n);
valve = x(g+n+1:2*g+n);
pmech = x(2*g+n+1:3*g+n);
load=x(3*g+n+1:3*g+2*n);
lambda = x(3*g+2*n+1:3*g+2*n+1);
% phi = x(3*g+2*n+1:3*g+3*n);
% pi_area = x(3*g+3*n+1:3*g+3*n+narea);
rhop = x(3*g+2*n+2:3*g+2*n+1+m);
rhom = x(3*g+2*n+2+m:3*g+2*n+1+2*m);
E=x(3*g+2*n+2+2*m:3*g+3*n+1+2*m);

% edge corresponding two-end voltage
edgev=ones(m,1);
for cnt1=1:m
   for cnt2=1:n
       if C(cnt2,cnt1) ~=0
       edgev(cnt1)=edgev(cnt1)*E(cnt2);
       end
   end
end

% calculate full phase angles
pgen = zeros(n,1);
pgen(G) = pmech; 

omega = zeros(n,1);
omega(G) = omega_g;
omega(L) = D(L,L)\( pgen(L) + deltaPm(L) - load(L) - C(L,:)*(edgev.*(diag(Bij_vf)*sin(transpose(C)*theta)) -Plink_orig) );


% obtain control signal
p_signal = zeros(g,1);
% p_signal(G_controlled) =price0(bus_controlled) + 0.5./(lambda - omega_g(G_controlled) + Hg(G_controlled,:)*rhom - Hg(G_controlled,:)*rhop )  -  1./(k_control*PF).* pmech(G_controlled)  ...
%    + 1./R(G_controlled) .*omega_g(G_controlled)  + pmech(G_controlled);   % cancel the oroginal droop control 

% p_signal(G_controlled) =price0(bus_controlled) - 10./(lambda - omega_g(G_controlled) + Hg(G_controlled,:)*rhom - Hg(G_controlled,:)*rhop )  -  1./(k_control*PF).* pmech(G_controlled)  ...
%     + 1./R(G_controlled) .*omega_g(G_controlled)  + pmech(G_controlled);   % need to know the equilibrium prices


p_signal(G_controlled) =(lambda - omega_g(G_controlled) + Hg(G_controlled,:)*rhom - Hg(G_controlled,:)*rhop ) + gain./( price_sup(bus_controlled)  - (lambda - omega_g(G_controlled) + Hg(G_controlled,:)*rhom - Hg(G_controlled,:)*rhop ) )   -  1./(k_control*PF).* pmech(G_controlled)  ...
    + 1./R(G_controlled) .*omega_g(G_controlled)  + pmech(G_controlled);   % self design

% load control signal
d_signal=zeros(n,1);
% d_signal(L_controlled)= 1./(k_control*PFd).* load(L_controlled) -  price0(L_controlled)- 0.5./(lambda - omega(L_controlled) + H(L_controlled-1,:)*rhom - H(L_controlled-1,:)*rhop ) ...
%    +load(L_controlled); 

% d_signal(L_controlled)= 1./(k_control*PFd).* load(L_controlled) -  price0(L_controlled) + 10./(lambda - omega(L_controlled) + H(L_controlled-1,:)*rhom - H(L_controlled-1,:)*rhop ) ...
%     +load(L_controlled); % need to know the equilibrium prices

% 
d_signal(L_controlled)= 1./(k_control*PFd).* load(L_controlled) -  (lambda - omega(L_controlled) + H(L_controlled-1,:)*rhom - H(L_controlled-1,:)*rhop ) - gain./( price_sup(L_controlled) -(lambda - omega(L_controlled) + H(L_controlled-1,:)*rhom - H(L_controlled-1,:)*rhop ))  ...
    +load(L_controlled); % self design

% simulate the system. Connecting with controller by p_signal
domega_g  = M(G,G)\(deltaPm(G) + pgen(G) -load(G) - D(G,G)*omega_g - C(G,:)*(edgev.*(diag(Bij_vf)*sin(transpose(C)*theta)) -Plink_orig));
dtheta  = omega;                     % theta rad, omega: rad/(s)
dvalve =  diag(Tg)\(-valve - 1./R .* omega_g + p_signal);
dpmech  = diag(Tb)\(- pmech + valve);
dload= diag(Td)\(-load + d_signal); 



% controller/cyber/computation and communication. Output: p_signal
dlambda = k_lambda*( - sum(deltaPm) - sum(pgen) + sum(load) );

% dphi    = k_phi*(C*diag(Bij)*transpose(C)*lambda - C*diag(Bij)*hatC'*pi_area - C*diag(Bij)*(rhop-rhom));
% dpi     = inter_area*k_pi*(hatC*diag(Bij)*transpose(C)*phi);


% if sum(ismember(thistory,t))==0
%     thistory = [thistory, t];
%     disp(thistory)
% end
% 
% if numel(thistory)>1
%     delta_t = thistory(end)-thistory(end-1);
% 
%     if delta_t< 1e0  
%         k_rho =1.0/delta_t ; 
%         disp(k_rho);
%     end
% end
drhop   = k_rho*(transpose(H)*(pgen(rm)+deltaPm(rm)-load(rm)) - barDeltaP);
drhom   = k_rho*(underDeltaP - transpose(H)*(pgen(rm)+deltaPm(rm)-load(rm))); 

%projection
ix=find(rhop<=0);
if ~isempty(ix)
    drhop(ix) = max(0*drhop(ix), drhop(ix));
end
ix=find(rhom<=0);
if ~isempty(ix)
    drhom(ix) = max(0*drhom(ix), drhom(ix));
end

% if (rhom(19)>0.0)
%     disp([drhom(19),rhom(19)]) 
% end


dE=zeros(n,1);
dE(G)=diag(Tv)\(repmat(Ef,g,1)-(1-(xd-xdt).*Bjj(G)).*E(G) + (xd-xdt).*( (CQ(G,:)*(edgev.*(diag(Bij_vf)*cos(transpose(C)*theta))))./E(G) )   ); 

%dE(L)= pgen(L) + deltaPm(L) - load(L) - C(L,:)*(edgev.*(diag(Bij_vf)*sin(transpose(C)*theta)) -Plink_orig)  -  D(L,L)*omega(L);

% size(dE(L))
% size(domega_g)
% size(dtheta)
% size(dvalve)
% size(dpmech)
% size(dlambda)
% size(drhop)
% size(drhom)


y=[domega_g;dtheta;dvalve;dpmech;dload;dlambda;drhop;drhom;dE];

return 