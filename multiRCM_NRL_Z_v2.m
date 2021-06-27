npick=200;
folder='F:\Shukai\NRL data\NRL - Connected Cavity Data\NRL - Connected Cavity Data\';

%folder='F:\Shukai\NRL data\higherloss\Connected Cavity Sparam Data - Higher Alpha\';
%load([folder '1p6a.mat']);
load([folder '1cavity_ports_on_same_face.mat']);

for i=1:npick
    j=round(i*(200/npick));
    S11a(i,:)=reshape(SCf(:,1,j),16001,1);
    S12a(i,:)=reshape(SCf(:,2,j),16001,1);
    S21a(i,:)=reshape(SCf(:,3,j),16001,1);
    S22a(i,:)=reshape(SCf(:,4,j),16001,1);

end

%load([folder '2p6a.mat']);
load([folder '2cavities_port2_in_side_location.mat']);
for i=1:npick
    j=round(i*(200/npick));
    S11b(i,:)=reshape(SCf(:,1,j),16001,1);
    S12b(i,:)=reshape(SCf(:,2,j),16001,1);
    S21b(i,:)=reshape(SCf(:,3,j),16001,1);
    S22b(i,:)=reshape(SCf(:,4,j),16001,1);

end

%load([folder '3p6a.mat']);
load([folder '3cavities_asymmetric_port_location.mat']);
for i=1:npick
    j=round(i*(200/npick));
    S11c(i,:)=reshape(SCf(:,1,j),16001,1);
    S12c(i,:)=reshape(SCf(:,2,j),16001,1);
    S21c(i,:)=reshape(SCf(:,3,j),16001,1);
    S22c(i,:)=reshape(SCf(:,4,j),16001,1);

end
clear SCf
%% S to Z, set Z0=50
[n,numpoints]=size(S11a);
freq_start=3.95*10^9;
freq_end=5.85*10^9;
freq=linspace(freq_start, freq_end, numpoints);
Z0=50;
[Z11a,Z12a,Z21a,Z22a]=S2Z_Bo(S11a,S12a,S21a,S22a,Z0,Z0); % 50 is the Z for transmission line
[Z11b,Z12b,Z21b,Z22b]=S2Z_Bo(S11b,S12b,S21b,S22b,Z0,Z0); % 50 is the Z for transmission line
[Z11c,Z12c,Z21c,Z22c]=S2Z_Bo(S11c,S12c,S21c,S22c,Z0,Z0); % 50 is the Z for transmission line
% %%
% mk=7;
% fig1=plot(freq/10^9,imag(Z11a(mk,:)),'-','DisplayName','1cav'); hold on;
% fig1=plot(freq/10^9,imag(Z11b(mk,:)),'-','DisplayName','2cav'); hold on;
% fig1=plot(freq/10^9,imag(Z11c(mk,:)),'-','DisplayName','3cav'); hold on;
% title('Z_{11} imag part compare');
% xlabel('Frequency (GHz)');
% ylabel('Ime(Z) (\Omega)');
% box on;
% xlim([3.95 5.85]);
% legend;
% hold off
%% binary sort
n1_trn=180;
n2_trn=180;
n3_trn=180;
n_trn=n1_trn+n2_trn+n3_trn;
s1=0; s2=0;
while (s1~=180 || s2~=180)
    for i=1:n_trn
       a=rand();
       if a<n1_trn/n_trn
           y_trn(i)=0;        % 0 is 1cav
       elseif a<(n1_trn+n2_trn)/n_trn
           y_trn(i)=1;        % 1 is 2cav
       else
           y_trn(i)=2;
       end
    end
    [~,s1]=size(find(y_trn==0));
    [~,s2]=size(find(y_trn==2));
end
size(find(y_trn==0))  %check random
size(find(y_trn==2))  %check random
%% arrange test
npck=20;
n1_tst=npck;
n2_tst=npck;
n3_tst=npck;
n_tst=n1_tst+n2_tst+n3_tst;
s1=0; s2=0;
while (s1~=npck || s2~=npck)
    for i=1:n_tst
       a=rand();
       if a<n1_tst/n_tst
           y_tst(i)=0;        % 0 is 1cav
       elseif a<(n1_tst+n2_tst)/n_tst
           y_tst(i)=1;        % 1 is 2cav
       else
           y_tst(i)=2;
       end
    end
    [~,s1]=size(find(y_tst==0));
    [~,s2]=size(find(y_tst==2));
end
size(find(y_tst==0))  %check random
size(find(y_tst==2))  %check random
%%
npick=10000;
ntot=16001;
clear x_tst x_trn ind
ia=1; ib=ia; ic=ia;

for i=1:npick
   ind(i)=floor(ntot/npick)*(i-1)+1; 
end


for i=1:n_tst
    if y_tst(i)==0
        x_tst(:,i)=Z11a(ia,ind);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst(:,i)=Z11b(ib,ind);
        ib=ib+1;
    else
        x_tst(:,i)=Z11c(ic,ind);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn(:,i)=Z11a(ia,ind);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn(:,i)=Z11b(ib,ind);
        ib=ib+1;
    else
        x_trn(:,i)=Z11c(ic,ind);
        ic=ic+1;
    end
end

% %%
% plot(1:npick,x_tst(:,1));
% %%
% plot(1:npick,x_trn(:,1));
%% h5 write, train
[a,b]=size(x_trn);
h5out=['train_rcm_nrl_3_Z11.h5'];
h5create(h5out,'/train_set_x',[a b]);
h5write(h5out,'/train_set_x',real(x_trn));

[a,b]=size(y_trn);
h5create(h5out,'/train_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/train_set_y',y_trn);
%
for i=1:2
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);
% h5 write, test
[a,b]=size(x_tst);
h5out=['test_rcm_nrl_3_Z11.h5'];
h5create(h5out,'/test_set_x',[a b]);
h5write(h5out,'/test_set_x',real(x_tst));

[a,b]=size(y_tst);
h5create(h5out,'/test_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/test_set_y',y_tst);
%
for i=1:2
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);